import torch
import torch.nn as nn
import torch.nn.functional as F

class AgriVisionNet(nn.Module):
    """
    Enterprise edge vision network engineered for deterministic static graph compilation.
    Resolves vanishing gradients and spatial dimension mismatches.
    """
    def __init__(self, use_optimization: bool = True):
        super(AgriVisionNet, self).__init__()
        self.use_optimization = use_optimization
        
        # Spatial Feature Extraction Layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        
        # Adaptive pooling replaces brittle hardcoded layers to allow flexible image sizing
        self.pool = nn.AdaptiveAvgPool2d((8, 8)) 
        self.fc = nn.Linear(16 * 8 * 8, 2)
        
        if self.use_optimization:
            self.bn1 = nn.BatchNorm2d(16)
            self.dropout = nn.Dropout2d(p=0.3)
            
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Applies explicit Kaiming/He normal initialization to stabilize the convergence vector."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Enforce strict 4D Tensor shape alignment checking at runtime
        if x.dim() != 4:
            raise ValueError(f"Invalid execution tensor rank. Expected [B, C, H, W], received: {x.shape}")

        if self.use_optimization:
            # Fused operator simulation: Conv2D + BatchNorm + ReLU
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = self.dropout(x)
        else:
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            
        x = torch.flatten(x, 1)
        return self.fc(x)

def export_to_onnx_pipeline(model: nn.Module, save_path: str = "agrivision_edge.onnx") -> None:
    """Compiles the static computation graph into an ONNX binary for low-latency edge targets."""
    model.eval()
    dummy_input = torch.randn(1, 3, 64, 64)
    
    torch.onnx.export(
        model, 
        dummy_input, 
        save_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input_tensor'],
        output_names=['output_logits'],
        dynamic_axes={'input_tensor': {0: 'batch_size'}, 'output_logits': {0: 'batch_size'}}
    )
