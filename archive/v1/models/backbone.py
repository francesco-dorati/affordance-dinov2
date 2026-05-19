import torch
import torch.nn as nn

class DINOv2Backbone(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()
        
        # Load the smallest DINOv2 model (ViT-Small) to save GPU memory
        # 'vits14' means Vision Transformer Small, with a patch size of 14x14 pixels.
        self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        
        # We freeze the backbone because DINOv2 already knows how to extract 
        # world-class semantic features. We don't want to accidentally ruin its weights!
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        # DINOv2 ViT-Small outputs embeddings of size 384
        self.embed_dim = 384

    def forward(self, x):
        """
        Args:
            x: RGB image tensor of shape [Batch, 3, H, W]
               Note: DINOv2 requires H and W to be multiples of 14.
        Returns:
            Patch embeddings of shape [Batch, Channels, Height/14, Width/14]
        """
        # Get the spatial dimensions of the input
        B, C, H, W = x.shape
        
        # Ensure input dimensions are divisible by the patch size (14)
        assert H % 14 == 0 and W % 14 == 0, \
            f"Input height and width must be divisible by 14. Got {H}x{W}."
        
        # DINOv2 expects inputs normalized using ImageNet stats, 
        # but for a frozen feature extractor on our dataset, raw [0,1] is often an okay start.
        # (We will add formal ImageNet normalization in the DataLoader later if needed).
        
        # Forward pass through DINOv2 to get the patch tokens
        # We use forward_features to get the dense spatial grid, not just the global class token.
        features = self.encoder.forward_features(x)
        
        # DINOv2 outputs a flat sequence of tokens: [Batch, Num_Patches, Embed_Dim]
        # We need to reshape this back into a 2D spatial grid (like an image feature map)
        patch_tokens = features['x_norm_patchtokens']
        
        # Calculate the new grid size (since patch size is 14)
        h_grid = H // 14
        w_grid = W // 14
        
        # Reshape to [Batch, Embed_Dim, h_grid, w_grid]
        spatial_features = patch_tokens.reshape(B, h_grid, w_grid, self.embed_dim).permute(0, 3, 1, 2)
        
        return spatial_features