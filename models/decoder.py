import torch
import torch.nn as nn
import torch.nn.functional as F

class UpsampleBlock(nn.Module):
    """
    Foundational building block that increases the spatial resolution (width and height) 
    of a feature map while simultaneously refining its feature channels.
    
    Instead of using Transposed Convolutions (which often cause "checkerboard" artifacts 
    in generated images), this block uses Bilinear Interpolation for smooth scaling, 
    followed by standard convolutions to learn the refined features.

    Args:
        in_channels (int): Number of feature channels in the input tensor.
        out_channels (int): Number of feature channels produced by the convolutions.
        scale_factor (float): The multiplier for the spatial dimensions (e.g., 2.0 doubles the size).
    """
    def __init__(self, in_channels, out_channels, scale_factor=2.0):
        super().__init__()
        self.scale_factor = scale_factor
        
        # A standard feature extraction pipeline: Conv -> BatchNorm -> ReLU
        # We use a sequence of two to allow the network to learn more complex relationships 
        # after the spatial upsampling occurs.
        self.conv = nn.Sequential(
            # First convolution: maps input channels to output channels
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), # Stabilizes training by normalizing activations
            nn.ReLU(inplace=True),        # Non-linear activation
            
            # Second convolution: refines the features within the new channel dimension
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass for the UpsampleBlock.
        
        Args:
            x (Tensor): Input tensor of shape [Batch, Channels, Height, Width]
            
        Returns:
            Tensor: Upsampled and refined tensor of shape [Batch, out_channels, Height * scale, Width * scale]
        """
        # 1. Spatially resize the image/feature map using smooth bilinear interpolation
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        
        # 2. Pass the resized feature map through the convolutional layers to compute new features
        return self.conv(x)


class MultiTaskDecoder(nn.Module):
    """
    A multi-task neural network decoder designed to take deep feature embeddings 
    (specifically from DINOv2) and predict two distinct visual outputs simultaneously:
      1. An Affordance Mask (Where can an object be interacted with?)
      2. Surface Normals (What is the 3D geometry/orientation of the surface?)
      
    The network uses a "Shared Trunk" to do the heavy lifting of upsampling and 
    feature processing, before splitting off into two specialized "Task Heads".

    Args:
        dino_embed_dim (int): The number of channels output by the DINOv2 encoder. 
                              Defaults to 384 (which is standard for DINOv2 ViT-Small).
    """
    def __init__(self, dino_embed_dim=384):
        super().__init__()
        
        # =====================================================================
        # 1. SHARED TRUNK
        # Purpose: Gradually scale up the 32x32 DINOv2 features to 224x224.
        # Sharing these layers saves memory and computation, and allows the network 
        # to learn generalized features that are useful for *both* downstream tasks.
        # =====================================================================
        
        # Input shape: [Batch, 384, 32, 32] -> Output shape: [Batch, 256, 56, 56]
        self.shared_up1 = UpsampleBlock(dino_embed_dim, 256, scale_factor=1.75) 
        
        # Input shape: [Batch, 256, 56, 56] -> Output shape: [Batch, 128, 112, 112]
        self.shared_up2 = UpsampleBlock(256, 128, scale_factor=2.0)             
        
        # Input shape: [Batch, 128, 112, 112] -> Output shape: [Batch, 64, 224, 224]
        self.shared_up3 = UpsampleBlock(128, 64, scale_factor=2.0)              


        # =====================================================================
        # 2. TASK HEAD A: Affordance Mask
        # Purpose: Predict a 2D map showing interaction probabilities (0 to 1).
        # =====================================================================
        self.mask_head = nn.Sequential(
            # Final upsample: [Batch, 64, 224, 224] -> [Batch, 32, 448, 448]
            UpsampleBlock(64, 32, scale_factor=2.0),                            
            
            # Collapse the 32 feature channels down into a single prediction channel
            # Shape becomes: [Batch, 1, 448, 448]
            nn.Conv2d(32, 1, kernel_size=1)                                     
        )


        # =====================================================================
        # 3. TASK HEAD B: Surface Normals
        # Purpose: Predict the 3D surface orientation (X, Y, Z) for every pixel.
        # =====================================================================
        self.normal_head = nn.Sequential(
            # Final upsample: [Batch, 64, 224, 224] -> [Batch, 32, 448, 448]
            UpsampleBlock(64, 32, scale_factor=2.0),                            
            
            # Collapse the 32 feature channels down into 3 channels representing 
            # the X, Y, and Z spatial vectors for the surface normal.
            # Shape becomes: [Batch, 3, 448, 448]
            nn.Conv2d(32, 3, kernel_size=1)                                     
        )

    def forward(self, x):
        """
        Forward pass defining the flow of data through the entire decoder.
        
        Args:
            x (Tensor): Feature map from DINOv2. Expected shape: [Batch, 384, 32, 32]
            
        Returns:
            tuple: (mask_pred, normal_pred)
                - mask_pred: Tensor of shape [Batch, 1, 448, 448] containing probabilities (0.0 to 1.0).
                - normal_pred: Tensor of shape [Batch, 3, 448, 448] containing unit vectors (-1.0 to 1.0).
        """
        # --- Shared Processing ---
        # Pass the input through the shared trunk to get a high-resolution, generalized feature map
        shared_features = self.shared_up1(x)
        shared_features = self.shared_up2(shared_features)
        shared_features = self.shared_up3(shared_features)
        
        # --- Head A: Mask Prediction Pipeline ---
        # Generate raw mask scores (logits)
        mask_logits = self.mask_head(shared_features)
        
        # Apply Sigmoid activation. This mathematically squishes all raw numbers to be 
        # strictly between 0 and 1, allowing us to treat them as probabilities.
        mask_pred = torch.sigmoid(mask_logits) 
        
        # --- Head B: Normal Prediction Pipeline ---
        # Generate raw X, Y, Z vector values
        normal_raw = self.normal_head(shared_features)
        
        # Apply L2 Normalization across the channel dimension (dim=1, which holds X, Y, Z).
        # A surface normal must be a "unit vector" (meaning its total length in 3D space is exactly 1).
        # This division ensures the math holds true for physical 3D representations.
        normal_pred = F.normalize(normal_raw, p=2, dim=1) 
        
        return mask_pred, normal_pred