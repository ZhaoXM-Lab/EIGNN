import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def extract_orthogonal_slices(img_3d, img_size=(121, 145, 121)):
    """
    Extract three orthogonal slices from a 3D image and resize them to 72x72.
    
    Args:
        img_3d: tensor of shape (batch_size, 1, H, W, D) - full 3D image
        img_size: size of the 3D image (default (121, 145, 121))
    
    Returns:
        tensor of shape (batch_size, 3, 72, 72) representing the three orthogonal slices
    """
    batch_size = img_3d.shape[0]
    H, W, D = img_size

    # Extract center slices from the 3D image
    center_h, center_w, center_d = H // 2, W // 2, D // 2

    # Get the three orthogonal center slices
    # Axial slice (xy plane) - slice through depth dimension
    axial_slice = img_3d[:, 0, :, :, center_d]  # (batch_size, H, W)

    # Coronal slice (xz plane) - slice through width dimension  
    coronal_slice = img_3d[:, 0, :, center_w, :]  # (batch_size, H, D)

    # Sagittal slice (yz plane) - slice through height dimension
    sagittal_slice = img_3d[:, 0, center_h, :, :]  # (batch_size, W, D)

    # Resize each slice to 72x72 using interpolation
    target_size = 72
    axial_resized = F.interpolate(axial_slice.unsqueeze(1), size=(target_size, target_size), mode='bilinear',
                                  align_corners=False)
    coronal_resized = F.interpolate(coronal_slice.unsqueeze(1), size=(target_size, target_size), mode='bilinear',
                                    align_corners=False)
    sagittal_resized = F.interpolate(sagittal_slice.unsqueeze(1), size=(target_size, target_size), mode='bilinear',
                                     align_corners=False)

    # Stack the three slices to create a 3-channel image
    combined_slices = torch.cat([axial_resized, coronal_resized, sagittal_resized], dim=1)  # (batch_size, 3, 72, 72)

    return combined_slices


class SNPModel(nn.Module):
    def __init__(self, input_dim):
        super(SNPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 200)
        self.bn1 = nn.BatchNorm1d(200, momentum=0.01, eps=0.001)  # Keras equivalent: momentum=0.99
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(200, 100)
        self.bn2 = nn.BatchNorm1d(100, momentum=0.01, eps=0.001)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(100, 50)
        self.bn3 = nn.BatchNorm1d(50, momentum=0.01, eps=0.001)
        self.dropout3 = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)

        x = F.relu(self.fc3(x))
        x = self.bn3(x)
        x = self.dropout3(x)

        return x


class ImageModel(nn.Module):
    def __init__(self, input_channels=1):
        super(ImageModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 72, kernel_size=3)
        self.conv2 = nn.Conv2d(72, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3)

        # Calculate the flattened size after convolutions
        # Input: 72x72, after conv1 (3x3): 70x70, after conv2 (3x3): 68x68, after conv3 (3x3): 66x66
        # So flattened size = 32 * 66 * 66 = 139,392
        self.fc = nn.Linear(32 * 66 * 66, 50)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the tensor
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads=4, key_dim=50, value_dim=None, use_bias=True,
                 kernel_initializer="glorot_uniform", bias_initializer="zeros"):
        super(MultiHeadAttention, self).__init__()

        # Match Keras MultiHeadAttention parameter structure exactly
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim if value_dim is not None else key_dim
        self.use_bias = use_bias

        # Keras uses key_dim for scaling, NOT per-head dimension
        self.inverse_sqrt_key_dim = 1.0 / math.sqrt(float(key_dim))

        # In the original TF code: MultiHeadAttention(num_heads=4, key_dim=50)
        # Input feature dimension is 50 (from dense_img, dense_clinical, dense_snp)
        input_dim = 50

        # Keras MultiHeadAttention creates these projections:
        # - Query projection: input_dim -> key_dim * num_heads
        # - Key projection: input_dim -> key_dim * num_heads  
        # - Value projection: input_dim -> value_dim * num_heads
        # - Output projection: value_dim * num_heads -> input_dim (default)
        self.w_q = nn.Linear(input_dim, key_dim * num_heads, bias=use_bias)
        self.w_k = nn.Linear(input_dim, key_dim * num_heads, bias=use_bias)
        self.w_v = nn.Linear(input_dim, self.value_dim * num_heads, bias=use_bias)

        # Output projection - Keras defaults to projecting back to input dimension
        self.w_o = nn.Linear(self.value_dim * num_heads, input_dim, bias=use_bias)

        # Initialize weights to match Keras defaults
        self._init_weights(kernel_initializer, bias_initializer)

    def _init_weights(self, kernel_initializer, bias_initializer):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if kernel_initializer == "glorot_uniform":
                    nn.init.xavier_uniform_(m.weight)
                else:
                    nn.init.xavier_uniform_(m.weight)  # Default fallback

                if m.bias is not None:
                    if bias_initializer == "zeros":
                        nn.init.zeros_(m.bias)
                    else:
                        nn.init.zeros_(m.bias)  # Default fallback

    def scaled_dot_product_attention(self, Q, K, V):
        # Keras uses sqrt(key_dim), not sqrt(d_k per head)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.inverse_sqrt_key_dim
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output

    def forward(self, query, value=None, key=None):
        # Match Keras MultiHeadAttention call signature: forward(query, value, key=None)
        # When key is None, key defaults to value
        if key is None:
            key = value if value is not None else query
        if value is None:
            value = query

        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        seq_len_v = value.size(1)

        # Linear projections and reshape for multi-head attention
        # Q: (batch, seq_len_q, key_dim * num_heads) -> (batch, num_heads, seq_len_q, key_dim)
        Q = self.w_q(query).view(batch_size, seq_len_q, self.num_heads, self.key_dim).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len_k, self.num_heads, self.key_dim).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len_v, self.num_heads, self.value_dim).transpose(1, 2)

        # Apply scaled dot-product attention
        attention_output = self.scaled_dot_product_attention(Q, K, V)

        # Concatenate heads: (batch, num_heads, seq_len_q, value_dim) -> (batch, seq_len_q, value_dim * num_heads)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.value_dim * self.num_heads)

        # Final linear projection
        output = self.w_o(attention_output)
        return output


class CrossModalAttention(nn.Module):
    def __init__(self):
        super(CrossModalAttention, self).__init__()
        # Match exact Keras call: MultiHeadAttention(num_heads=4, key_dim=50)
        self.mha1 = MultiHeadAttention(num_heads=4, key_dim=50)
        self.mha2 = MultiHeadAttention(num_heads=4, key_dim=50)

    def forward(self, x, y):
        # Add sequence dimension: (batch, 50) -> (batch, 1, 50)
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)

        # Match original TF: MultiHeadAttention()(x, y) means query=x, value=y, key=y
        a1 = self.mha1(x, y)  # Equivalent to query=x, value=y, key=y (default)
        # Match original TF: MultiHeadAttention()(y, x) means query=y, value=x, key=x  
        a2 = self.mha2(y, x)  # Equivalent to query=y, value=x, key=x (default)

        # Remove sequence dimension: (batch, 1, 50) -> (batch, 50)
        a1 = a1.squeeze(1)
        a2 = a2.squeeze(1)

        # Concatenate: (batch, 50) + (batch, 50) -> (batch, 100)
        return torch.cat([a1, a2], dim=1)


class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()
        # Match exact Keras call: MultiHeadAttention(num_heads=4, key_dim=50)
        self.mha = MultiHeadAttention(num_heads=4, key_dim=50)

    def forward(self, x):
        # Add sequence dimension: (batch, 50) -> (batch, 1, 50)
        x = x.unsqueeze(1)

        # Match original TF: MultiHeadAttention()(x, x) means query=x, value=x, key=x
        attention = self.mha(x, x)  # Equivalent to query=x, value=x, key=x (default)

        # Remove sequence dimension: (batch, 1, 50) -> (batch, 50)
        attention = attention.squeeze(1)

        return attention


class MultiModalModel(nn.Module):
    def __init__(self, mode='MM_SA_BA', snp_dim=937, img_channels=3, num_classes=1, img_size=(121, 145, 121), **kwargs):
        super(MultiModalModel, self).__init__()
        self.mode = mode
        self.num_classes = num_classes
        self.img_size = img_size

        # Sub-models (removed clinical_model)
        self.snp_model = SNPModel(snp_dim)
        self.img_model = ImageModel(img_channels)

        # Attention modules (only keep SNP-Image interactions)
        self.cross_attention_av = CrossModalAttention()  # SNP-Image attention

        self.self_attention_v = SelfAttention()  # Image self-attention
        self.self_attention_a = SelfAttention()  # SNP self-attention

        # Calculate merged dimension based on mode
        base_dim = 50 + 50  # snp + img outputs (removed clinical)
        if mode == 'MM_BA':
            merged_dim = base_dim + 100  # 1 cross-modal attention pair, outputs 100
        elif mode == 'MM_SA':
            merged_dim = base_dim + 50 * 2  # 2 self-attention, each outputs 50
        elif mode == 'MM_SA_BA':
            merged_dim = base_dim + 100  # same as MM_BA
        elif mode == 'None':
            merged_dim = base_dim
        else:
            raise ValueError("Mode must be one of 'MM_SA', 'MM_BA', 'MM_SA_BA' or 'None'.")

        # Output layer - adapt for binary classification like EIGN
        if num_classes == 1:
            self.output = nn.Linear(merged_dim, 1)  # Binary classification
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.output = nn.Linear(merged_dim, num_classes)  # Multi-class
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, img, snp, std_out=False):
        """
        Forward pass adapted for the framework's interface
        
        Args:
            img: Image data - tensor of shape (batch_size, num_patches, C, H, W, D) - EIGN format
            snp: SNP data - tensor of shape (batch_size, snp_dim)
            std_out: If True, return only standard output (compatibility with EIGN)
        
        Returns:
            output: Model predictions
        """
        # MADDi uses single patch, extract it from EIGN format
        if img.dim() == 6:  # EIGN format: [batch, patches, C, H, W, D]
            img = img[:, 0]  # Take first (only) patch: [batch, C, H, W, D]

        # Extract orthogonal slices from 3D image for image processing
        img_2d = extract_orthogonal_slices(img, self.img_size)

        # Get features from sub-models
        dense_snp = self.snp_model(snp)
        dense_img = self.img_model(img_2d)

        ########### Attention Layer ############

        ## Cross Modal Bi-directional Attention ##
        if self.mode == 'MM_BA':
            # Only SNP-Image cross attention
            av_att = self.cross_attention_av(dense_snp, dense_img)
            merged = torch.cat([av_att, dense_img, dense_snp], dim=1)

        ## Self Attention ##
        elif self.mode == 'MM_SA':
            vv_att = self.self_attention_v(dense_img)
            aa_att = self.self_attention_a(dense_snp)
            merged = torch.cat([aa_att, vv_att, dense_img, dense_snp], dim=1)

        ## Self Attention and Cross Modal Bi-directional Attention##
        elif self.mode == 'MM_SA_BA':
            vv_att = self.self_attention_v(dense_img)
            aa_att = self.self_attention_a(dense_snp)
            # Cross attention between self-attended features
            av_att = self.cross_attention_av(aa_att, vv_att)
            merged = torch.cat([av_att, dense_img, dense_snp], dim=1)

        ## No Attention ##
        elif self.mode == 'None':
            merged = torch.cat([dense_img, dense_snp], dim=1)

            ########### Output Layer ############
        output = self.output(merged)

        if std_out:
            return output
        else:
            return output,

    def evaluate_data(self, val_loader, device, dtype='float32'):
        """
        Evaluate the model on validation data (compatible with EIGN interface)
        """
        predicts = []
        groundtruths = []
        group_labels = []

        with torch.no_grad():
            self.train(False)
            for i, data in enumerate(val_loader, 0):
                inputs, aux_labels, labels, dis_label = data

                # MADDi uses single patch, so data is squeezed in dataloader
                # Unsqueeze to maintain consistency with EIGN interface
                if inputs.dim() == 5:  # If squeezed: [batch, C, H, W, D] -> [batch, 1, C, H, W, D]
                    inputs = inputs.unsqueeze(1)
                    aux_labels = aux_labels.unsqueeze(1)
                    labels = labels.unsqueeze(1)
                    dis_label = dis_label.unsqueeze(1)

                inputs = inputs.to(device=device, dtype=dtype)
                aux_labels = aux_labels.to(device=device, dtype=dtype)

                # Now use standard EIGN interface
                outputs = self(inputs, aux_labels[:, 0])
                predicts.append(outputs)
                groundtruths.append(labels[:, 0, :])  # Standard multi-patch interface
                group_labels.append(dis_label)

            device = next(self.parameters()).device
            pred = [i[0] for i in predicts]
            pred = torch.cat(pred, 0)

            if self.num_classes == 1:
                pred = torch.sigmoid(pred)
            else:
                pred = torch.softmax(pred, dim=1)

            groundtruths = torch.cat(groundtruths, dim=0).squeeze(-1).to(dtype)
            group_labels = torch.cat(group_labels, dim=0).to(torch.long)


            # Calculate loss based on num_classes
            if self.num_classes == 1:
                # Binary classification: BCEWithLogitsLoss expects same shape
                val_loss = self.criterion(pred.to(device),
                                          groundtruths.to(device=device))
            else:
                # Multi-class classification: CrossEntropyLoss expects [batch] target
                val_loss = self.criterion(pred.to(device),
                                          groundtruths.squeeze(-1).long().to(device=device))

            # Use standard EIGN return format
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
        losses = torch.zeros(1, dtype=dtype, device=device, )

        c = 0
        batch_size = train_loader.batch_size
        inputs_buf = torch.Tensor()
        aux_labels_buf = torch.Tensor()
        labels_buf = torch.Tensor()

        for n, data in enumerate(train_loader, 0):
            inputs, aux_labels, labels, dis_label = data

            # MADDi uses single patch, so data is squeezed in dataloader
            # Unsqueeze to maintain consistency with EIGN interface
            if inputs.dim() == 5:  # If squeezed: [batch, C, H, W, D] -> [batch, 1, C, H, W, D]
                inputs = inputs.unsqueeze(1)
                aux_labels = aux_labels.unsqueeze(1)
                labels = labels.unsqueeze(1)
                dis_label = dis_label.unsqueeze(1)

            # Filter out nan values using EIGN logic
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

            # multi patch
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
    # Generate random test data compatible with the framework
    batch_size = 4
    img_size = (121, 145, 121)  # Full image size like ResNet

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # SNP data: 937-dimensional 0/1 vectors
    snp_data = np.random.randint(0, 2, (batch_size, 937)).astype(np.float32)

    # Image data: full 3D image format (batch_size, 1, H, W, D)
    img_data = np.random.randn(batch_size, 1, *img_size).astype(np.float32)

    # Convert to PyTorch tensors
    snp_tensor = torch.FloatTensor(snp_data).to(device)
    img_tensor = torch.FloatTensor(img_data).to(device)

    # Test all model modes
    modes = ['None', 'MM_SA', 'MM_BA', 'MM_SA_BA']

    for mode in modes:
        print(f"Testing mode: {mode}")
        model = MultiModalModel(mode=mode, snp_dim=937, img_channels=3, num_classes=1, img_size=img_size).to(device)
        model.eval()
        with torch.no_grad():
            # Test framework-compatible interface
            outputs = model(img_tensor, snp_tensor, std_out=True)
            print(f"Output shape: {outputs.shape}")

            print(f"Model {mode} test passed!")
            print("-" * 40)
