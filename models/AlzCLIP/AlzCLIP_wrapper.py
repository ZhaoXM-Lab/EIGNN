"""
AlzCLIP wrapper to integrate two-stage training into the framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import nibabel as nib
from models.AlzCLIP.model import CLIPModel_simple, ClassificationHead
from models.AlzCLIP.model import ImageEncoder_linear, SNPEncoder_linear

BASEDIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import opts for patch size and center matrix configuration
try:
    from utils.opts import FULL_PATCH_SIZE, FULL_CENTER_MAT
except ImportError:
    # Fallback values if import fails
    FULL_PATCH_SIZE = (117, 141, 117)
    FULL_CENTER_MAT = np.array([[60], [72], [60]])


class AlzCLIPWrapper(nn.Module):
    """
    AlzCLIP wrapper compatible with the existing framework interface.
    """

    def __init__(self,
                 embedding_dim: int = 256,
                 projection_dim: int = 128,
                 num_classes: int = 1,  # Default to 1 for binary classification like EIGN
                 dropout: float = 0.1,
                 temperature: float = 1.0,
                 pretrain_epochs: int = 100,
                 pretrain_lr: float = 0.001,  # Learning rate for pretraining stage
                 finetune_lr: float = 0.0001,  # Learning rate for fine-tuning stage
                 weight_decay: float = 0.0001,  # Weight decay for both stages
                 pretrain_checkpoint_save_path: str = "/tmp/best_pretrain_alzclip.pt",
                 img_feature_dim: int = 170,  # Changed to 170 for AAL3v1 brain regions
                 snp_input_dim: int = 234,  # SNP input dimension for AlzCLIP
                 freeze_clip_in_finetune: bool = True,
                 **kwargs):
        super().__init__()

        # Store hyperparameters
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.temperature = temperature
        self.pretrain_epochs = pretrain_epochs
        self.pretrain_lr = pretrain_lr
        self.finetune_lr = finetune_lr
        self.weight_decay = weight_decay
        self.pretrain_checkpoint_path = pretrain_checkpoint_save_path
        self.img_feature_dim = img_feature_dim
        self.snp_input_dim = snp_input_dim
        self.freeze_clip_in_finetune = freeze_clip_in_finetune
        # Create args object for CLIPModel_simple
        self.args = type('Args', (), {
            'embedding_dim': embedding_dim,
            'projection_dim': projection_dim,
            'dropout': dropout,
            'temperature': temperature,
            'num_classes': max(num_classes, 2)  # CLIP model needs at least 2 classes
        })()

        # Initialize the CLIP model with correct dimensions
        self.clip_model = CLIPModel_simple(self.args)
        # Override encoder dimensions to match our data

        self.clip_model.image_encoder = ImageEncoder_linear(in_dim=img_feature_dim, out_dim=embedding_dim)
        self.clip_model.snp_encoder = SNPEncoder_linear(in_dim=snp_input_dim, out_dim=embedding_dim)

        # Initialize classification head
        if num_classes == 1:
            # Binary classification
            self.classifier = nn.Linear(projection_dim * 2, 1)
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            # Multi-class classification
            self.classifier = ClassificationHead(
                input_dim=projection_dim * 2,
                hidden_dim=128,
                num_classes=num_classes,
                dropout=dropout
            )
            self.criterion = nn.CrossEntropyLoss()

        # Training state
        self.current_epoch = 0
        self.stage = 'pretrain'  # 'pretrain' or 'finetune'
        self.best_pretrain_loss = float('inf')
        self.pretrain_checkpoint_loaded = False

        # Load AAL3v1 atlas for brain region extraction
        self.aal3v1_atlas = None
        self.aal3v1_atlas_tensor = None  # GPU tensor version for fast processing
        self.aal3v1_roi_masks = None  # Pre-computed ROI masks for maximum efficiency
        self.aal3v1_roi_mapping = None
        self._load_aal3v1_atlas()

    def _load_aal3v1_atlas(self):
        """Load AAL3v1 atlas template and ROI mapping, crop it to match input patch size"""
        try:
            # Load the AAL3v1 atlas template
            atlas_path = os.path.join(BASEDIR, "data/templates/AAL3v1_1.5mm_resample.nii")
            roi_path = os.path.join(BASEDIR, "data/templates/AAL3v1_ROI.txt")

            atlas_full = nib.load(atlas_path).get_fdata().astype(np.int32)

            # Crop atlas using the same logic as batch_sampling
            # Calculate margin based on FULL_PATCH_SIZE
            margin = [int(np.floor((i - 1) / 2.0)) for i in FULL_PATCH_SIZE]

            # Get center coordinates
            x_cor, y_cor, z_cor = FULL_CENTER_MAT[0, 0], FULL_CENTER_MAT[1, 0], FULL_CENTER_MAT[2, 0]

            # Apply same cropping logic as batch_sampling function
            self.aal3v1_atlas = atlas_full[
                                max(x_cor - margin[0], 0): x_cor + margin[0] + 1,
                                max(y_cor - margin[1], 0): y_cor + margin[1] + 1,
                                max(z_cor - margin[2], 0): z_cor + margin[2] + 1
                                ]

            # Load ROI mapping from the text file
            self.aal3v1_roi_mapping = {}
            with open(roi_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 3:
                            roi_id = int(parts[0])
                            roi_name = parts[1]
                            # Some lines might have additional numeric values, we just need the ID and name
                            self.aal3v1_roi_mapping[roi_id] = roi_name

            print(f"Loaded AAL3v1 atlas with {len(self.aal3v1_roi_mapping)} ROIs")
            print(f"Cropped atlas shape: {self.aal3v1_atlas.shape}")

            # Convert atlas to tensor for GPU processing (will be moved to device later)
            self.aal3v1_atlas_tensor = torch.from_numpy(self.aal3v1_atlas.astype(np.int32))

            # Pre-compute ROI masks for maximum efficiency
            self._precompute_roi_masks()

        except Exception as e:
            print(f"Warning: Could not load AAL3v1 atlas: {e}")
            # Fallback to None - will use simple feature extraction
            self.aal3v1_atlas = None
            self.aal3v1_atlas_tensor = None
            self.aal3v1_roi_masks = None
            self.aal3v1_roi_mapping = None

    def _precompute_roi_masks(self):
        """Pre-compute ROI masks for all 170 regions to maximize performance"""
        if self.aal3v1_atlas_tensor is None:
            raise NotImplementedError

        # Flatten atlas for efficient mask computation
        atlas_flat = self.aal3v1_atlas_tensor.contiguous().view(-1)  # [H*W*D]

        # Create ROI IDs tensor
        roi_ids = torch.arange(1, 171, dtype=self.aal3v1_atlas_tensor.dtype)  # [170]

        # Pre-compute all ROI masks: [170, H*W*D]
        # Broadcasting: [170, 1] == [1, H*W*D] -> [170, H*W*D]
        self.aal3v1_roi_masks = atlas_flat.unsqueeze(0) == roi_ids.unsqueeze(1)
        self.aal3v1_roi_masks = self.aal3v1_roi_masks.to(torch.float32)

        print(f"Pre-computed ROI masks: {self.aal3v1_roi_masks.shape} (170 ROIs)")
        print(
            f"Memory usage for masks: {self.aal3v1_roi_masks.numel() * self.aal3v1_roi_masks.element_size() / 1024 / 1024:.1f} MB")

    def _setup_optimizer_for_stage(self, optimizer, stage):

        if stage == 'pretrain':
            target_lr = self.pretrain_lr
            # Ensure CLIP model parameters are trainable, classifier frozen
            for param in self.clip_model.parameters():
                param.requires_grad = True
            for param in self.classifier.parameters():
                param.requires_grad = False
            print(f"Setting up optimizer for PRETRAIN stage:")

        else:  # finetune stage
            target_lr = self.finetune_lr
            # Ensure CLIP model parameters are frozen, classifier trainable
            for param in self.clip_model.parameters():
                param.requires_grad = False if self.freeze_clip_in_finetune else True
            for param in self.classifier.parameters():
                param.requires_grad = True
            print(f"Setting up optimizer for FINETUNE stage:")

        # Update learning rate for all parameter groups
        for param_group in optimizer.param_groups:
            param_group['lr'] = target_lr
            param_group['weight_decay'] = self.weight_decay

    def get_model_parameters(self):
        """
        Get all model parameters for optimizer initialization.
        This should be used when creating the optimizer to ensure it contains all parameters.
        
        Returns:
            iterator: All model parameters (CLIP model + classifier)
        """
        # Return iterator over all model parameters
        import itertools
        return itertools.chain(self.clip_model.parameters(), self.classifier.parameters())

    def _validate_optimizer_type(self, optimizer):
        """
        Check if optimizer is Adam-like and supports the required parameters
        
        Args:
            optimizer: The optimizer to validate
        """
        optimizer_name = optimizer.__class__.__name__
        supported_optimizers = ['Adam', 'AdamW']

        if optimizer_name not in supported_optimizers:
            print(f"Warning: Optimizer {optimizer_name} may not be fully supported. "
                  f"Recommended optimizers: {supported_optimizers}")
        else:
            print(f"✓ Using supported optimizer: {optimizer_name}")

        # Check if optimizer has required parameter groups structure
        if not hasattr(optimizer, 'param_groups') or not isinstance(optimizer.param_groups, list):
            raise ValueError(f"Optimizer {optimizer_name} does not support param_groups structure")

    def _extract_img_features(self, img):
        """
        Extract brain region features from 3D image data using AAL3v1 atlas.
        GPU-accelerated version for high performance.
        """
        if img.dim() == 6:  # EIGN format: [batch, patches, C, H, W, D]
            img = img[:, 0]  # Take first patch: [batch, C, H, W, D]

        batch_size = img.shape[0]
        device = img.device
        dtype = img.dtype

        if self.aal3v1_roi_masks is None:
            # Fallback to simple global average if atlas not available
            raise NotImplementedError("Error: AAL3v1 atlas not available")

        # Move pre-computed masks to same device as input if needed
        if self.aal3v1_roi_masks.device != device:
            self.aal3v1_roi_masks = self.aal3v1_roi_masks.to(device)

        # Get image data: [batch, C, H, W, D] -> [batch, H, W, D]
        img_data = img[:, 0]  # Remove channel dimension

        # Verify dimensions match (check against original atlas shape)
        expected_shape = self.aal3v1_atlas_tensor.shape if self.aal3v1_atlas_tensor is not None else self.aal3v1_atlas.shape
        if img_data.shape[1:] != expected_shape:
            raise ValueError(
                f"Image dimension mismatch: expected {expected_shape}, "
                f"got {img_data.shape[1:]}")

        # Highly optimized vectorized processing using pre-computed masks
        # Flatten spatial dimensions for efficient processing
        img_flat = img_data.view(batch_size, -1)  # [batch_size, H*W*D]

        # Use pre-computed masks (already on correct device)
        # [batch_size, H*W*D] @ [H*W*D, 170] -> [batch_size, 170]
        region_features = img_flat @ self.aal3v1_roi_masks.T

        return region_features

    def _prepare_snp_features(self, snp):
        """Handle SNP dimension mismatch"""
        # If SNP data dimension doesn't match AlzCLIP expectation, adapt it
        if snp.shape[1] != self.snp_input_dim:
            raise ValueError(f"SNP dimension mismatch: expected {self.snp_input_dim}, got {snp.shape[1]}")
        return snp

    def forward(self, img, snp, std_out=False):
        """
        Forward pass compatible with framework interface.
        
        Args:
            img: Image data - tensor of shape (batch_size, num_patches, C, H, W, D) - EIGN format
            snp: SNP data - tensor of shape (batch_size, snp_dim)
            std_out: If True, return only standard output (compatibility with EIGN)
        
        Returns:
            output: Model predictions
        """
        # Extract and prepare features
        img_features = self._extract_img_features(img)
        snp_features = self._prepare_snp_features(snp)

        # Create batch dictionary for CLIP model
        batch = {
            'img': img_features,
            'snp': snp_features,
            'gt': torch.zeros(img_features.shape[0], dtype=torch.long, device=img.device)  # Dummy for inference
        }

        # Get embeddings from CLIP model
        img_emb, snp_emb = self.clip_model.get_embeddings(batch)

        # Combine embeddings
        combined_emb = torch.cat([img_emb, snp_emb], dim=1)

        # Classification output (only used during fine-tuning)
        if self.stage == 'finetune':
            output = self.classifier(combined_emb)
        else:
            # During pretraining, return embeddings for contrastive learning
            output = combined_emb

        if std_out:
            return output
        else:
            return output,

    def save_best_pretrain_model(self, val_loss):
        """
        Save the best pretrained model if current validation loss is better
        """
        if val_loss < self.best_pretrain_loss:
            self.best_pretrain_loss = val_loss
            torch.save({
                'clip_model_state_dict': self.clip_model.state_dict(),
                'best_loss': val_loss,
                'epoch': self.current_epoch
            }, self.pretrain_checkpoint_path)
            return True
        return False

    def load_best_pretrain_model(self):
        """
        Load the best pretrained model
        """
        if os.path.exists(self.pretrain_checkpoint_path):
            checkpoint = torch.load(self.pretrain_checkpoint_path, map_location=next(self.parameters()).device)
            self.clip_model.load_state_dict(checkpoint['clip_model_state_dict'])
            self.pretrain_checkpoint_loaded = True
            print(
                f"Loaded best pretrained model with loss: {checkpoint['best_loss']:.4f} from epoch {checkpoint['epoch']}")
        else:
            raise FileNotFoundError("Warning: No pretrained checkpoint found!")

    def evaluate_data(self, val_loader, device, dtype='float32'):
        """
        Evaluate the model on validation data (compatible with EIGN interface)
        For pretrain stage: calculate contrastive loss and save best model
        For finetune stage: calculate classification metrics
        """
        if self.stage == 'pretrain':
            pred, groundtruths, group_labels, val_loss = self._evaluate_pretrain_stage(val_loader, device, dtype)
            print(f'Pretraining val loss: {val_loss:.4f}')
            val_loss = np.inf  # must return inf in pretraining, otherwise the framework may not save the best model in finetuning
        else:
            pred, groundtruths, group_labels, val_loss = self._evaluate_finetune_stage(val_loader, device, dtype)
        return pred, groundtruths, group_labels, val_loss

    def _evaluate_pretrain_stage(self, val_loader, device, dtype='float32'):
        """
        Evaluate during pretrain stage: only calculate contrastive loss
        """
        self.clip_model.eval()
        total_loss = 0
        num_batches = 0
        groundtruths = []
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                inputs, aux_labels, labels, dis_label = data

                # Handle dimension consistency
                if inputs.dim() == 5:
                    inputs = inputs.unsqueeze(1)
                    aux_labels = aux_labels.unsqueeze(1)
                    labels = labels.unsqueeze(1)
                    dis_label = dis_label.unsqueeze(1)

                groundtruths.append(labels[:, 0, :])
                inputs = inputs.to(device=device, dtype=dtype)
                aux_labels = aux_labels.to(device=device, dtype=dtype)
                labels = labels.to(device=device, dtype=dtype)

                # Extract and prepare features
                img_features = self._extract_img_features(inputs)
                snp_features = self._prepare_snp_features(aux_labels[:, 0])

                # Create batch for contrastive learning
                batch = {
                    'img': img_features,
                    'snp': snp_features,
                    'gt': labels[:, 0, :].squeeze(-1).long()
                }

                # Calculate contrastive loss
                loss = self.clip_model(batch)
                total_loss += loss.item()
                num_batches += 1

        # Calculate average validation loss
        avg_val_loss = total_loss / num_batches if num_batches > 0 else float('inf')

        # Save best model if improved
        self.save_best_pretrain_model(avg_val_loss)

        # Return dummy values that match EIGN interface requirements
        # During pretrain, we don't need actual predictions
        groundtruths = torch.cat(groundtruths, dim=0).squeeze(-1).to(dtype).cpu().numpy()
        batch_size = len(val_loader.dataset)
        dummy_pred = np.zeros((batch_size, 1, 1))  # Dummy predictions
        dummy_gt = groundtruths.ravel().reshape((batch_size, 1, 1))  # Dummy ground truth
        dummy_group = np.zeros((batch_size, 1))  # Dummy group labels

        return dummy_pred, dummy_gt, dummy_group, avg_val_loss

    def _evaluate_finetune_stage(self, val_loader, device, dtype='float32'):
        """
        Evaluate during finetune stage: calculate classification metrics
        """
        predicts = []
        groundtruths = []
        group_labels = []

        with torch.no_grad():
            self.train(False)
            for i, data in enumerate(val_loader, 0):
                inputs, aux_labels, labels, dis_label = data

                # Handle dimension consistency like MADDi
                if inputs.dim() == 5:  # If squeezed: [batch, C, H, W, D] -> [batch, 1, C, H, W, D]
                    inputs = inputs.unsqueeze(1)
                    aux_labels = aux_labels.unsqueeze(1)
                    labels = labels.unsqueeze(1)
                    dis_label = dis_label.unsqueeze(1)

                inputs = inputs.to(device=device, dtype=dtype)
                aux_labels = aux_labels.to(device=device, dtype=dtype)

                # Use framework interface
                outputs = self(inputs, aux_labels[:, 0])
                predicts.append(outputs)
                groundtruths.append(labels[:, 0, :])
                group_labels.append(dis_label)

            device = next(self.parameters()).device
            # Handle tuple output from forward (outputs, ) or single output
            pred = [i[0] if isinstance(i, tuple) else i for i in predicts]
            pred = torch.cat(pred, 0)

            # Apply appropriate activation
            if self.num_classes == 1:
                pred = torch.sigmoid(pred)
            else:
                pred = torch.softmax(pred, dim=1)

            groundtruths = torch.cat(groundtruths, dim=0).squeeze(-1).to(dtype)
            group_labels = torch.cat(group_labels, dim=0).to(torch.long)

            # Calculate loss
            if self.num_classes == 1:
                val_loss = self.criterion(pred.to(device), groundtruths.to(device=device))
            else:
                val_loss = self.criterion(pred.to(device), groundtruths.squeeze(-1).long().to(device=device))

            # Return in EIGN format
            pred = pred.unsqueeze(-1).cpu().numpy()
            groundtruths = groundtruths.unsqueeze(-1).cpu().numpy()
            group_labels = group_labels.cpu().numpy()
            val_loss = val_loss.cpu().item()

        return pred, groundtruths, group_labels, val_loss

    def fit(self, train_loader, optimizer, device, dtype):
        """
        Train the model for one epoch (compatible with EIGN interface)
        """
        self.train(True)
        losses = torch.zeros(1, dtype=dtype, device=device)

        # At epoch 0, reload external optimizer settings and check optimizer type
        if self.current_epoch == 0:
            # Check if we should skip pretraining (pretrain_epochs = 0)
            if self.pretrain_epochs == 0:
                print(f"\n{'=' * 50}")
                print("Skipping pretraining (pretrain_epochs=0), starting with finetune stage")
                print(f"{'=' * 50}")

                # Set stage to finetune immediately
                self.stage = 'finetune'

                # Setup optimizer for finetune stage (will handle parameter freezing)
                self._setup_optimizer_for_stage(optimizer, 'finetune')

                print("CLIP model parameters frozen for fine-tuning")
                print(f"Classification head ready with input dim: {self.projection_dim * 2}")
                print(f"{'=' * 50}\n")
            else:
                # Normal pretraining start
                self._setup_optimizer_for_stage(optimizer, 'pretrain')

        # Check if optimizer supports Adam-like parameters
        self._validate_optimizer_type(optimizer)

        c = 0
        batch_size = train_loader.batch_size
        inputs_buf = torch.Tensor()
        aux_labels_buf = torch.Tensor()
        labels_buf = torch.Tensor()

        for n, data in enumerate(train_loader, 0):
            inputs, aux_labels, labels, dis_label = data

            if inputs.dim() == 5:  # If squeezed: [batch, C, H, W, D] -> [batch, 1, C, H, W, D]
                inputs = inputs.unsqueeze(1)
                aux_labels = aux_labels.unsqueeze(1)
                labels = labels.unsqueeze(1)
                dis_label = dis_label.unsqueeze(1)

            inx = ~torch.isnan(labels.view(labels.shape[0], -1)[:, 0])
            inx = inx & (~torch.isnan(inputs.view(inputs.shape[0], -1)[:, 0]))
            inx = inx & (~torch.isnan(aux_labels.view(aux_labels.shape[0], -1)[:, 0]))

            inputs_buf = torch.cat([inputs_buf, inputs[inx]], 0)
            aux_labels_buf = torch.cat([aux_labels_buf, aux_labels[inx]], 0)
            labels_buf = torch.cat([labels_buf, labels[inx]], 0)

            if (n + 1) < len(train_loader):
                if inputs_buf.shape[0] < batch_size + 2:  # batch norm needs more than 1 sample
                    continue
                else:
                    inputs = inputs_buf[:batch_size]
                    aux_labels = aux_labels_buf[:batch_size]
                    labels = labels_buf[:batch_size]

                    inputs_buf = inputs_buf[batch_size:]
                    aux_labels_buf = aux_labels_buf[batch_size:]
                    labels_buf = labels_buf[batch_size:]
            else:
                inputs = inputs_buf
                aux_labels = aux_labels_buf
                labels = labels_buf
            c += 1

            # Process multi patch format
            labels = labels[:, 0, :].to(device=device, dtype=dtype)
            aux_labels = aux_labels.to(device=device, dtype=dtype)
            inputs = inputs.to(device=device, dtype=dtype)

            optimizer.zero_grad()

            # Calculate loss based on training stage
            loss = self._calculate_stage_loss(inputs, aux_labels[:, 0], labels)

            loss.backward(retain_graph=True)
            losses += loss.detach()
            optimizer.step()

        # Update epoch and stage
        self.current_epoch += 1
        # Only switch from pretrain to finetune if we actually did pretraining
        if self.current_epoch >= self.pretrain_epochs and self.stage == 'pretrain' and self.pretrain_epochs > 0:
            print(f"\n{'=' * 50}")
            print("Switching from pretrain to finetune stage")
            print(f"{'=' * 50}")

            # Load best pretrained model before switching to finetune
            if not self.pretrain_checkpoint_loaded:
                self.load_best_pretrain_model()

            self.stage = 'finetune'

            # Reset optimizer for finetune stage (will handle parameter freezing)
            self._setup_optimizer_for_stage(optimizer, 'finetune')

            print("CLIP model parameters frozen for fine-tuning")
            print(f"Classification head ready with input dim: {self.projection_dim * 2}")
            print(f"{'=' * 50}\n")

        return losses / c

    def _calculate_stage_loss(self, inputs, aux_labels, labels):
        """Calculate loss based on current training stage"""

        if self.stage == 'pretrain':
            # During pretraining: ONLY contrastive loss

            # Extract and prepare features using shared methods
            img_features = self._extract_img_features(inputs)
            snp_features = self._prepare_snp_features(aux_labels)

            # Create batch for contrastive learning
            batch = {
                'img': img_features,
                'snp': snp_features,
                'gt': labels.squeeze(-1).long()
            }

            # Contrastive loss only
            contrastive_loss = self.clip_model(batch)
            return contrastive_loss

        else:
            # During fine-tuning: ONLY classification loss
            # Use forward function to get predictions, then calculate loss

            outputs = self.forward(inputs, aux_labels, std_out=True)

            assert labels.shape[1] == 1
            if self.num_classes == 1:
                # For binary classification, labels should be shape [batch, 1]
                classification_loss = self.criterion(outputs, labels[:, 0, :])
            else:
                # For multi-class, labels should be shape [batch] with class indices
                classification_loss = self.criterion(outputs, labels[:, 0, :].squeeze(-1).long())

            return classification_loss


if __name__ == "__main__":
    # Generate random test data compatible with the framework
    batch_size = 4
    img_feature_dim = 170  # AAL3v1 brain regions
    snp_input_dim = 937  # SNP input dimension for AlzCLIP
    img_size = (117, 141, 117)  # Full patch size from FULL_PATCH_SIZE

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing AlzCLIP on device: {device}")

    # Generate random data matching the expected format
    # SNP data: 234-dimensional vectors
    snp_data = np.random.randn(batch_size, snp_input_dim).astype(np.float32)

    # Image data: EIGN format (batch_size, num_patches, C, H, W, D)
    # For AlzCLIP, we simulate the format that comes from EIGN
    img_data = np.random.randn(batch_size, 1, 1, *img_size).astype(np.float32)

    # Convert to PyTorch tensors
    snp_tensor = torch.FloatTensor(snp_data).to(device)
    img_tensor = torch.FloatTensor(img_data).to(device)

    print(f"SNP tensor shape: {snp_tensor.shape}")
    print(f"Image tensor shape: {img_tensor.shape}")

    # Test different configurations
    test_configs = [
        {"num_classes": 1, "pretrain_epochs": 2},  # Binary classification
        {"num_classes": 3, "pretrain_epochs": 2},  # Multi-class classification
        {"num_classes": 1, "pretrain_epochs": 0},  # Binary classification with no pretraining (direct finetune)
    ]

    for i, config in enumerate(test_configs):
        print(f"\n{'=' * 60}")
        print(f"Testing configuration {i + 1}: {config}")
        print(f"{'=' * 60}")

        # Create model
        model = AlzCLIPWrapper(
            embedding_dim=256,
            projection_dim=128,
            num_classes=config["num_classes"],
            pretrain_epochs=config["pretrain_epochs"],
            pretrain_lr=0.001,  # Test pretrain learning rate
            finetune_lr=0.0001,  # Test finetune learning rate
            weight_decay=0.0001,  # Test weight decay
            img_feature_dim=img_feature_dim,
            snp_input_dim=snp_input_dim,
        ).to(device)

        print(f"Model created successfully!")
        print(f"Current stage: {model.stage}")
        print(f"Current epoch: {model.current_epoch}")
        print(f"Pretrain LR: {model.pretrain_lr}")
        print(f"Finetune LR: {model.finetune_lr}")
        print(f"Weight decay: {model.weight_decay}")

        # Test the new unified parameter approach
        print(f"\n--- Testing unified parameter optimization ---")
        # Create optimizer with all model parameters
        import torch.optim as optim

        test_optimizer = optim.Adam(model.get_model_parameters(), lr=0.001)
        print(f"Created optimizer with {len(test_optimizer.param_groups)} parameter groups")
        print(f"Total parameters in optimizer: {sum(len(group['params']) for group in test_optimizer.param_groups)}")

        # Test stage setup
        model._setup_optimizer_for_stage(test_optimizer, 'pretrain')
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters in pretrain setup: {trainable_params}/{total_params}")

        model._setup_optimizer_for_stage(test_optimizer, 'finetune')
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters in finetune setup: {trainable_params}/{total_params}")

        # Test pretrain stage
        print(f"\n--- Testing PRETRAIN stage ---")
        model.eval()
        with torch.no_grad():
            # Test forward pass during pretraining
            outputs = model(img_tensor, snp_tensor, std_out=True)
            print(f"Pretrain output shape: {outputs.shape}")
            print(f"Expected pretrain output shape: [batch_size, projection_dim * 2] = [{batch_size}, {128 * 2}]")

            # Test contrastive loss calculation
            try:
                labels = torch.randint(0, max(config["num_classes"], 2), (batch_size, 1, 1)).to(device)
                loss = model._calculate_stage_loss(img_tensor, snp_tensor, labels)
                print(f"Contrastive loss: {loss.item():.4f}")
                print("✓ Pretrain stage test passed!")
            except Exception as e:
                print(f"✗ Pretrain stage test failed: {e}")

        # Simulate switching to finetune stage
        print(f"\n--- Simulating stage transition ---")
        model.current_epoch = config["pretrain_epochs"]
        model.stage = 'finetune'

        # Freeze CLIP model parameters (simulate the transition)
        for param in model.clip_model.parameters():
            param.requires_grad = False

        print(f"Switched to stage: {model.stage}")

        # Test finetune stage
        print(f"\n--- Testing FINETUNE stage ---")
        model.eval()
        with torch.no_grad():
            # Test forward pass during fine-tuning
            outputs = model(img_tensor, snp_tensor, std_out=True)

            if config["num_classes"] == 1:
                expected_shape = (batch_size, 1)
                print(f"Finetune output shape: {outputs.shape}")
                print(f"Expected finetune output shape: {expected_shape}")

                # Test sigmoid activation
                probs = torch.sigmoid(outputs)
                print(f"Sigmoid probabilities range: [{probs.min().item():.4f}, {probs.max().item():.4f}]")
            else:
                expected_shape = (batch_size, config["num_classes"])
                print(f"Finetune output shape: {outputs.shape}")
                print(f"Expected finetune output shape: {expected_shape}")

                # Test softmax activation  
                probs = torch.softmax(outputs, dim=1)
                print(f"Softmax probabilities shape: {probs.shape}")
                print(f"Softmax probabilities sum: {probs.sum(dim=1)}")

            # Test classification loss calculation
            try:
                if config["num_classes"] == 1:
                    labels = torch.rand(batch_size, 1, 1).to(device)  # Binary labels
                else:
                    labels = torch.randint(0, config["num_classes"], (batch_size, 1, 1)).to(device)

                loss = model._calculate_stage_loss(img_tensor, snp_tensor, labels)
                print(f"Classification loss: {loss.item():.4f}")
                print("✓ Finetune stage test passed!")
            except Exception as e:
                print(f"✗ Finetune stage test failed: {e}")

        print(f"\n--- Testing evaluation interface ---")
        try:
            # Create dummy data loader format
            dummy_data = [(img_tensor, snp_tensor.unsqueeze(1),
                           labels.to(device), torch.zeros(batch_size, 1, 1))]


            class DummyLoader:
                def __init__(self, data):
                    self.data = data
                    self.dataset = list(range(batch_size))

                def __iter__(self):
                    return iter(self.data)

                def __len__(self):
                    return len(self.data)


            dummy_loader = DummyLoader(dummy_data)

            # Test evaluation method
            pred, gt, group_labels, val_loss = model.evaluate_data(dummy_loader, device)
            print(f"Evaluation predictions shape: {pred.shape}")
            print(f"Evaluation ground truth shape: {gt.shape}")
            print(f"Evaluation group labels shape: {group_labels.shape}")
            print(f"Evaluation loss: {val_loss:.4f}")
            print("✓ Evaluation interface test passed!")

        except Exception as e:
            print(f"✗ Evaluation interface test failed: {e}")

        print(f"\n✓ Configuration {i + 1} test completed successfully!")
        print(f"{'=' * 60}")
