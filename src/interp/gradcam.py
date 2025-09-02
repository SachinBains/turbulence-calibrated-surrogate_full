import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
import cv2

class GradCAM:
    """Gradient-weighted Class Activation Mapping for 3D velocity fields."""
    
    def __init__(self, model: nn.Module, target_layers: List[str]):
        """
        Initialize GradCAM.
        
        Args:
            model: PyTorch model
            target_layers: List of layer names to hook
        """
        self.model = model
        self.target_layers = target_layers
        self.gradients = {}
        self.activations = {}
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook
        
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                self.hooks.append(module.register_forward_hook(forward_hook(name)))
                self.hooks.append(module.register_backward_hook(backward_hook(name)))
    
    def generate_cam(self, input_tensor: torch.Tensor, target_class: Optional[int] = None,
                    layer_name: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Generate Class Activation Maps.
        
        Args:
            input_tensor: Input tensor (1, C, D, H, W)
            target_class: Target class index (None for regression)
            layer_name: Specific layer to generate CAM for
            
        Returns:
            Dictionary of CAMs for each target layer
        """
        # Forward pass
        input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        
        # For regression, use mean of output as target
        if target_class is None:
            target = output.mean()
        else:
            target = output[0, target_class]
        
        # Backward pass
        self.model.zero_grad()
        target.backward(retain_graph=True)
        
        cams = {}
        target_layers = [layer_name] if layer_name else self.target_layers
        
        for layer in target_layers:
            if layer in self.gradients and layer in self.activations:
                gradients = self.gradients[layer]  # (1, C, D, H, W)
                activations = self.activations[layer]  # (1, C, D, H, W)
                
                # Global average pooling of gradients
                weights = gradients.mean(dim=(2, 3, 4), keepdim=True)  # (1, C, 1, 1, 1)
                
                # Weighted combination of activation maps
                cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, D, H, W)
                
                # Apply ReLU
                cam = F.relu(cam)
                
                # Normalize
                cam = self._normalize_cam(cam)
                
                cams[layer] = cam
        
        return cams
    
    def _normalize_cam(self, cam: torch.Tensor) -> torch.Tensor:
        """Normalize CAM to [0, 1]."""
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam
    
    def generate_velocity_cam(self, input_tensor: torch.Tensor, 
                            velocity_component: int = 0) -> Dict[str, torch.Tensor]:
        """
        Generate CAM for specific velocity component.
        
        Args:
            input_tensor: Input tensor
            velocity_component: Which velocity component (0=u, 1=v, 2=w)
            
        Returns:
            CAMs for velocity component
        """
        input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        
        # Target specific velocity component
        if velocity_component < output.shape[1]:
            target = output[0, velocity_component].mean()
        else:
            target = output.mean()
        
        # Backward pass
        self.model.zero_grad()
        target.backward(retain_graph=True)
        
        cams = {}
        for layer in self.target_layers:
            if layer in self.gradients and layer in self.activations:
                gradients = self.gradients[layer]
                activations = self.activations[layer]
                
                weights = gradients.mean(dim=(2, 3, 4), keepdim=True)
                cam = (weights * activations).sum(dim=1, keepdim=True)
                cam = F.relu(cam)
                cam = self._normalize_cam(cam)
                
                cams[layer] = cam
        
        return cams
    
    def cleanup(self):
        """Remove hooks."""
        for hook in self.hooks:
            hook.remove()

class VelocityFieldGradCAM:
    """Specialized GradCAM for turbulence velocity fields."""
    
    def __init__(self, model: nn.Module):
        """Initialize with automatic layer detection."""
        self.model = model
        self.target_layers = self._find_target_layers()
        self.gradcam = GradCAM(model, self.target_layers)
    
    def _find_target_layers(self) -> List[str]:
        """Automatically find suitable layers for GradCAM."""
        target_layers = []
        
        for name, module in self.model.named_modules():
            # Look for convolutional layers in decoder path
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                if 'dec' in name or 'up' in name or 'out' in name:
                    target_layers.append(name)
        
        # If no decoder layers found, use all conv layers
        if not target_layers:
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                    target_layers.append(name)
        
        return target_layers[-3:]  # Use last 3 layers
    
    def analyze_velocity_importance(self, input_tensor: torch.Tensor) -> Dict:
        """
        Analyze importance of different regions for velocity prediction.
        
        Args:
            input_tensor: Input velocity field (1, C, D, H, W)
            
        Returns:
            Analysis results with CAMs for each velocity component
        """
        results = {}
        velocity_names = ['u', 'v', 'w']
        
        for i, vel_name in enumerate(velocity_names):
            print(f"Generating GradCAM for {vel_name} component...")
            
            cams = self.gradcam.generate_velocity_cam(input_tensor, i)
            results[vel_name] = cams
        
        return results
    
    def visualize_cams(self, input_tensor: torch.Tensor, cams: Dict, 
                      save_dir: Optional[str] = None) -> None:
        """
        Visualize GradCAM results.
        
        Args:
            input_tensor: Original input tensor
            cams: CAM results from analyze_velocity_importance
            save_dir: Directory to save visualizations
        """
        velocity_names = ['u', 'v', 'w']
        
        for vel_name in velocity_names:
            if vel_name in cams:
                vel_cams = cams[vel_name]
                
                for layer_name, cam in vel_cams.items():
                    self._plot_cam_slices(input_tensor, cam, 
                                        f"{vel_name}_{layer_name}", save_dir)
    
    def _plot_cam_slices(self, input_tensor: torch.Tensor, cam: torch.Tensor,
                        title: str, save_dir: Optional[str] = None):
        """Plot CAM overlaid on input slices."""
        # Get middle slices
        input_np = input_tensor[0, 0].cpu().numpy()  # First channel
        cam_np = cam[0, 0].cpu().numpy()  # CAM
        
        D, H, W = input_np.shape
        
        # Create figure with 3 slices (XY, XZ, YZ)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # XY slice (middle Z)
        z_mid = D // 2
        xy_slice = input_np[z_mid]
        cam_xy = cam_np[z_mid]
        
        axes[0].imshow(xy_slice, cmap='viridis', alpha=0.7)
        axes[0].imshow(cam_xy, cmap='Reds', alpha=0.5)
        axes[0].set_title(f'{title} - XY slice (z={z_mid})')
        axes[0].axis('off')
        
        # XZ slice (middle Y)
        y_mid = H // 2
        xz_slice = input_np[:, y_mid, :]
        cam_xz = cam_np[:, y_mid, :]
        
        axes[1].imshow(xz_slice, cmap='viridis', alpha=0.7)
        axes[1].imshow(cam_xz, cmap='Reds', alpha=0.5)
        axes[1].set_title(f'{title} - XZ slice (y={y_mid})')
        axes[1].axis('off')
        
        # YZ slice (middle X)
        x_mid = W // 2
        yz_slice = input_np[:, :, x_mid]
        cam_yz = cam_np[:, :, x_mid]
        
        axes[2].imshow(yz_slice, cmap='viridis', alpha=0.7)
        axes[2].imshow(cam_yz, cmap='Reds', alpha=0.5)
        axes[2].set_title(f'{title} - YZ slice (x={x_mid})')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/{title}_gradcam.png", dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def get_importance_statistics(self, cams: Dict) -> Dict:
        """Compute statistics from GradCAM results."""
        stats = {}
        
        for vel_name, vel_cams in cams.items():
            vel_stats = {}
            
            for layer_name, cam in vel_cams.items():
                cam_np = cam.cpu().numpy()
                
                vel_stats[layer_name] = {
                    'mean_importance': float(np.mean(cam_np)),
                    'max_importance': float(np.max(cam_np)),
                    'std_importance': float(np.std(cam_np)),
                    'active_regions': float(np.sum(cam_np > 0.5) / cam_np.size)
                }
            
            stats[vel_name] = vel_stats
        
        return stats
    
    def cleanup(self):
        """Clean up hooks."""
        self.gradcam.cleanup()
