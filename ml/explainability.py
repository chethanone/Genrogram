import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) for CustomCNN.
    Target Layer: conv4 (final feature extraction layer).
    """
    def __init__(self, model: nn.Module, target_layer_name: str = "conv4"):
        self.model = model
        self.model.eval()
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        
        self._register_hooks()

    def _register_hooks(self):
        target_layer = getattr(self.model, self.target_layer_name, None)
        if target_layer is None:
            raise ValueError(f"Layer {self.target_layer_name} not found in model.")

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        Generates 2D normalized Grad-CAM heatmap overlay matching input spectrogram spatial dimensions.
        input_tensor: [1, 1, H, W]
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()
            
        score = logits[0, target_class]
        score.backward()
        
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        h_in, w_in = input_tensor.shape[2], input_tensor.shape[3]
        cam_resized = F.interpolate(cam, size=(h_in, w_in), mode='bilinear', align_corners=False)
        
        cam_np = cam_resized.squeeze().cpu().numpy()
        
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-6:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)
            
        return cam_np

def test_gradcam_on_example(model_path: str = "models/best_model.pt", output_plot_path: str = "audit_outputs/dataset/gradcam_single_file_test.png"):
    from ml.custom_cnn import CustomCNN
    from ml.dataset import GTZANSegmentDataset, IDX_TO_GENRE
    
    full_model_path = project_root / model_path
    ckpt = torch.load(full_model_path, map_location="cpu")
    model = CustomCNN(num_classes=10)
    model.load_state_dict(ckpt["model_state_dict"])
    
    val_ds = GTZANSegmentDataset(str(project_root / "data/split_manifest.json"), str(project_root / "data/gtzan"), split="val")
    spec_tensor, label_idx = val_ds[0]
    input_batch = spec_tensor.unsqueeze(0)
    
    gradcam = GradCAM(model, target_layer_name="conv4")
    heatmap = gradcam.generate_heatmap(input_batch, target_class=label_idx)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    mel_np = spec_tensor.squeeze(0).numpy()
    
    im0 = axes[0].imshow(mel_np, aspect='auto', origin='lower', cmap='inferno')
    axes[0].set_title(f"Original Mel Spectrogram ({IDX_TO_GENRE[label_idx]})")
    fig.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(heatmap, aspect='auto', origin='lower', cmap='jet')
    axes[1].set_title("Grad-CAM Activation Heatmap")
    fig.colorbar(im1, ax=axes[1])
    
    axes[2].imshow(mel_np, aspect='auto', origin='lower', cmap='inferno')
    axes[2].imshow(heatmap, aspect='auto', origin='lower', cmap='jet', alpha=0.5)
    axes[2].set_title("Overlay (Spectrogram + Activation)")
    
    plt.tight_layout()
    full_output_path = project_root / output_plot_path
    full_output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(full_output_path, dpi=150)
    plt.close()
    
    print(f"SUCCESS: Grad-CAM heatmap generated and saved to {full_output_path}")

if __name__ == "__main__":
    test_gradcam_on_example()
