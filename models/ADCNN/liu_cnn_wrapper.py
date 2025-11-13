"""
Liu CNN model wrapper for the Brain_IMGEN framework.

This module integrates the Liu CNN model into the framework using the original
prepare_model function, maintaining the original architecture while providing
framework-compatible interface.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Liu_CNN.build_model import prepare_model


class LiuCNNWrapper(nn.Module):
    """
    Wrapper class to integrate Liu CNN model into the framework.
    
    Uses the original prepare_model function to maintain consistency with
    the paper's implementation while providing framework-compatible interface.
    """

    def __init__(self, num_classes=1, img_size=(96, 96, 96), snp_dim=937,
                 in_channel=1, feat_dim=1024, n_hid_main=512, expansion=8,
                 type_name='conv3x3x3', norm_type='Instance', **kwargs):
        """
        Initialize Liu CNN model wrapper using original prepare_model function.
        
        Args:
            num_classes: Number of output classes (1 for binary classification)
            img_size: Input image size (H, W, D) - default (121, 145, 121)
            snp_dim: SNP feature dimension (not used in Liu CNN, kept for compatibility)
            in_channel: Input channels (default 1)
            feat_dim: Feature dimension for CNN output (default 1024)
            n_hid_main: Hidden layer size in classifier (default 512)
            expansion: Channel expansion factor for CNN (default 8)
            type_name: CNN type name (default 'conv3x3x3')
            norm_type: Normalization type (default 'Instance')
            **kwargs: Additional arguments for compatibility
        """

        super(LiuCNNWrapper, self).__init__()

        self.num_classes = num_classes
        self.img_size = img_size
        self.snp_dim = snp_dim
        self.feat_dim = feat_dim

        # Determine n_label based on num_classes
        # prepare_model expects n_label (output dimension), not num_classes
        n_label = num_classes if num_classes > 1 else 1

        # Use original prepare_model function to build the model
        self.liu_model = prepare_model(
            in_channel=in_channel,
            feat_dim=feat_dim,
            n_hid_main=n_hid_main,
            n_label=n_label,
            out_dim=12,  # This parameter is not used in the current prepare_model
            expansion=expansion,
            type_name=type_name,
            norm_type=norm_type
        )

        # Loss function
        if num_classes == 1:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        # SNP dummy layer for compatibility (Liu CNN doesn't use SNP data)
        self.snp_dummy = nn.Identity()

    def forward(self, img, snp=None, std_out=False):
        """
        Forward pass compatible with framework interface.
        
        Args:
            img: Image data - tensor of shape (batch_size, num_patches, C, H, W, D) or (batch_size, C, H, W, D)
            snp: SNP data - tensor of shape (batch_size, snp_dim) (ignored in Liu CNN)
            std_out: If True, return only standard output (compatibility with EIGN)
        
        Returns:
            output: Model predictions
        """
        # Handle EIGN format: extract single patch if multi-patch format
        if img.dim() == 6:  # EIGN format: [batch, patches, C, H, W, D]
            img = img[:, 0]  # Take first (only) patch: [batch, C, H, W, D]

        # Ensure correct input format for Liu CNN (expects single channel)
        if img.shape[1] != 1:
            # If multi-channel, take first channel
            img = img[:, :1]

        # Liu CNN doesn't use age information in our adaptation
        # Set age_id to None for compatibility
        age_id = None

        # Forward through Liu CNN model
        output = self.liu_model(img, age_id)

        # Ensure output shape compatibility
        if self.num_classes == 1 and output.dim() == 1:
            output = output.unsqueeze(-1)
        elif self.num_classes == 1 and output.shape[-1] != 1:
            # If output has wrong shape, take first column for binary classification
            output = output[:, :1]

        if std_out:
            return output
        else:
            return output,

    def evaluate_data(self, val_loader, device, dtype='float32'):
        """
        Evaluate the model on validation data (compatible with framework interface).
        
        Args:
            val_loader: DataLoader for validation data
            device: Device to run evaluation on
            dtype: Data type for tensors
            
        Returns:
            tuple: (predictions, ground_truths, group_labels, validation_loss)
        """
        predicts = []
        groundtruths = []
        group_labels = []

        with torch.no_grad():
            self.train(False)
            for i, data in enumerate(val_loader, 0):
                inputs, aux_labels, labels, dis_label = data

                # Handle data format compatibility
                if inputs.dim() == 5:  # If squeezed: [batch, C, H, W, D] -> [batch, 1, C, H, W, D]
                    inputs = inputs.unsqueeze(1)
                    aux_labels = aux_labels.unsqueeze(1)
                    labels = labels.unsqueeze(1)
                    dis_label = dis_label.unsqueeze(1)

                inputs = inputs.to(device=device, dtype=dtype)
                aux_labels = aux_labels.to(device=device, dtype=dtype)

                # Forward pass (Liu CNN ignores SNP data)
                outputs = self(inputs, aux_labels[:, 0])
                predicts.append(outputs)
                groundtruths.append(labels[:, 0, :])
                group_labels.append(dis_label)

        device = next(self.parameters()).device
        pred = [i[0] for i in predicts]
        pred = torch.cat(pred, 0)

        # Apply appropriate activation function
        if self.num_classes == 1:
            pred = torch.sigmoid(pred)
        else:
            pred = torch.softmax(pred, dim=1)

        groundtruths = torch.cat(groundtruths, dim=0).squeeze(-1).to(dtype)
        group_labels = torch.cat(group_labels, dim=0).to(torch.long)

        # Calculate validation loss
        if self.num_classes == 1:
            # Binary classification: BCEWithLogitsLoss expects same shape
            val_loss = self.criterion(pred.to(device), groundtruths.to(device=device))
        else:
            # Multi-class classification: CrossEntropyLoss expects [batch] target
            val_loss = self.criterion(pred.to(device), groundtruths.squeeze(-1).long().to(device=device))

        # Convert to numpy for compatibility
        pred = pred.unsqueeze(-1).cpu().numpy()
        groundtruths = groundtruths.unsqueeze(-1).cpu().numpy()
        group_labels = group_labels.cpu().numpy()
        val_loss = val_loss.cpu().item()

        return pred, groundtruths, group_labels, val_loss

    def fit(self, train_loader, optimizer, device, dtype):
        """
        Train the model for one epoch (compatible with framework interface).
        
        Args:
            train_loader: DataLoader for training data
            optimizer: Optimizer for training
            device: Device to run training on
            dtype: Data type for tensors
            
        Returns:
            torch.Tensor: Average training loss for the epoch
        """
        self.train(True)
        losses = torch.zeros(1, dtype=dtype, device=device)

        c = 0
        batch_size = train_loader.batch_size
        inputs_buf = torch.Tensor()
        aux_labels_buf = torch.Tensor()
        labels_buf = torch.Tensor()

        for n, data in enumerate(train_loader, 0):
            inputs, aux_labels, labels, dis_label = data

            # Handle data format compatibility
            if inputs.dim() == 5:  # If squeezed: [batch, C, H, W, D] -> [batch, 1, C, H, W, D]
                inputs = inputs.unsqueeze(1)
                aux_labels = aux_labels.unsqueeze(1)
                labels = labels.unsqueeze(1)
                dis_label = dis_label.unsqueeze(1)

            # Filter out NaN values using framework logic
            inx = ~torch.isnan(labels.view(labels.shape[0], -1)[:, 0])
            inx = inx & (~torch.isnan(inputs.view(inputs.shape[0], -1)[:, 0]))
            inx = inx & (~torch.isnan(aux_labels.view(aux_labels.shape[0], -1)[:, 0]))

            inputs_buf = torch.cat([inputs_buf, inputs[inx]], 0)
            aux_labels_buf = torch.cat([aux_labels_buf, aux_labels[inx]], 0)
            labels_buf = torch.cat([labels_buf, labels[inx]], 0)

            if (n + 1) < len(train_loader):
                if inputs_buf.shape[0] < batch_size + 2:  # batch norm must use more than 1 sample
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

            # Prepare data for training
            labels = labels[:, 0, :].to(device=device, dtype=dtype)
            aux_labels = aux_labels.to(device=device, dtype=dtype)
            inputs = inputs.to(device=device, dtype=dtype)

            optimizer.zero_grad()
            outputs = self(inputs, aux_labels[:, 0])

            assert labels.shape[1] == 1
            if self.num_classes == 1:
                # Binary classification: BCEWithLogitsLoss expects same shape
                loss = self.criterion(outputs[0], labels[:, 0, :])
            else:
                # Multi-class classification: CrossEntropyLoss expects [batch] target
                loss = self.criterion(outputs[0], labels[:, 0, :].squeeze(-1).long())

            loss.backward(retain_graph=True)
            losses += loss.detach()
            optimizer.step()

        return losses / c


if __name__ == "__main__":
    # Test the Liu CNN integration
    batch_size = 2
    img_size = (96, 96, 96)
    # img_size = (121, 145, 121)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test data
    snp_data = np.random.randint(0, 2, (batch_size, 937)).astype(np.float32)
    img_data = np.random.randn(batch_size, 1, *img_size).astype(np.float32)

    # Convert to PyTorch tensors
    snp_tensor = torch.FloatTensor(snp_data).to(device)
    img_tensor = torch.FloatTensor(img_data).to(device)

    # Test model
    print("Testing Liu CNN integration...")
    model = LiuCNNWrapper(num_classes=1, img_size=img_size).to(device)
    model.eval()

    with torch.no_grad():
        # Test framework-compatible interface
        outputs = model(img_tensor, snp_tensor, std_out=True)
        print(f"Output shape: {outputs.shape}")
        print("Liu CNN integration test passed!")
