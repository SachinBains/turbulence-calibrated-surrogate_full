#!/usr/bin/env python3
"""
Gradio Demo Interface for Turbulence Surrogate Modeling
Interactive web interface for turbulence prediction and uncertainty quantification.
"""

import os
import sys
import gradio as gr
import numpy as np
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import tempfile
from PIL import Image
import io

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import load_config
from src.utils.devices import pick_device
from src.dataio.hit_dataset import HITDataset
from src.models.unet3d import UNet3D
from src.interp.shap_analysis import TurbulenceSHAP
from src.interp.ale import TurbulenceALE
from src.interp.gradcam import VelocityFieldGradCAM
from src.interp.sobol import TurbulenceSobolAnalyzer
from src.physics.q_criterion import TurbulenceVisualization
from src.uq.swag import SWAGModel
from torch.utils.data import DataLoader

class TurbulenceDemo:
    """Main demo class for turbulence surrogate modeling."""
    
    def __init__(self):
        self.models = {}
        self.datasets = {}
        self.device = pick_device()
        self.current_sample = None
        self.current_predictions = {}
        
        # Initialize analyzers
        self.shap_analyzer = None
        self.ale_analyzer = None
        self.gradcam_analyzer = None
        self.sobol_analyzer = None
        self.q_viz = TurbulenceVisualization()
        
        # Load available experiments
        self.load_available_experiments()
    
    def load_available_experiments(self):
        """Load available experiment configurations."""
        config_dir = Path('configs')
        self.available_experiments = {}
        
        if config_dir.exists():
            for config_file in config_dir.glob('*.yaml'):
                exp_name = config_file.stem
                self.available_experiments[exp_name] = str(config_file)
    
    def load_experiment(self, experiment_name: str) -> str:
        """Load a specific experiment."""
        if experiment_name not in self.available_experiments:
            return f"Experiment {experiment_name} not found!"
        
        try:
            config_path = self.available_experiments[experiment_name]
            cfg = load_config(config_path)
            exp_id = cfg['experiment_id']
            
            # Load model
            results_dir = Path(cfg['paths']['results_dir']) / exp_id
            ckpts = sorted(results_dir.glob('best_*.pth'))
            
            if not ckpts:
                return f"No checkpoint found for {experiment_name}"
            
            # Build model
            mcfg = cfg['model']
            model = UNet3D(
                mcfg['in_channels'], 
                mcfg['out_channels'], 
                base_ch=mcfg['base_channels']
            )
            
            # Load weights
            state = torch.load(ckpts[-1], map_location=self.device)
            model.load_state_dict(state['model'])
            model = model.to(self.device)
            model.eval()
            
            self.models[experiment_name] = model
            
            # Load dataset
            dataset = HITDataset(cfg, 'test', eval_mode=True)
            self.datasets[experiment_name] = dataset
            
            # Initialize analyzers
            self.shap_analyzer = TurbulenceSHAP(model, self.device)
            self.ale_analyzer = TurbulenceALE(model, self.device)
            self.gradcam_analyzer = VelocityFieldGradCAM(model)
            self.sobol_analyzer = TurbulenceSobolAnalyzer(model, self.device)
            
            return f"Successfully loaded experiment: {experiment_name}"
            
        except Exception as e:
            return f"Error loading experiment {experiment_name}: {str(e)}"
    
    def get_sample_data(self, experiment_name: str, sample_idx: int) -> Tuple[str, Dict]:
        """Get sample data for visualization."""
        if experiment_name not in self.datasets:
            return "Experiment not loaded!", {}
        
        try:
            dataset = self.datasets[experiment_name]
            if sample_idx >= len(dataset):
                return f"Sample index {sample_idx} out of range (max: {len(dataset)-1})", {}
            
            input_data, target_data = dataset[sample_idx]
            
            # Store current sample
            self.current_sample = {
                'input': input_data,
                'target': target_data,
                'experiment': experiment_name,
                'index': sample_idx
            }
            
            # Get model prediction
            model = self.models[experiment_name]
            with torch.no_grad():
                input_tensor = input_data.unsqueeze(0).to(self.device)
                prediction = model(input_tensor)
                prediction = prediction.cpu().numpy()[0]
            
            self.current_predictions = {
                'prediction': prediction,
                'target': target_data.numpy(),
                'input': input_data.numpy()
            }
            
            # Compute basic statistics
            pred_stats = {
                'mean': float(np.mean(prediction)),
                'std': float(np.std(prediction)),
                'min': float(np.min(prediction)),
                'max': float(np.max(prediction))
            }
            
            target_stats = {
                'mean': float(np.mean(target_data.numpy())),
                'std': float(np.std(target_data.numpy())),
                'min': float(np.min(target_data.numpy())),
                'max': float(np.max(target_data.numpy()))
            }
            
            mse = float(np.mean((prediction - target_data.numpy())**2))
            
            return f"Loaded sample {sample_idx} from {experiment_name}", {
                'prediction_stats': pred_stats,
                'target_stats': target_stats,
                'mse': mse
            }
            
        except Exception as e:
            return f"Error loading sample: {str(e)}", {}
    
    def create_velocity_slice_plot(self, axis: str = 'z', slice_idx: int = 16) -> go.Figure:
        """Create velocity field slice visualization."""
        if not self.current_predictions:
            return go.Figure().add_annotation(text="No data loaded", showarrow=False)
        
        prediction = self.current_predictions['prediction']
        target = self.current_predictions['target']
        
        # Extract slice
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        ax_idx = axis_map[axis]
        
        if ax_idx == 0:
            pred_slice = prediction[slice_idx, :, :]
            target_slice = target[slice_idx, :, :]
        elif ax_idx == 1:
            pred_slice = prediction[:, slice_idx, :]
            target_slice = target[:, slice_idx, :]
        else:
            pred_slice = prediction[:, :, slice_idx]
            target_slice = target[:, :, slice_idx]
        
        # Create subplots
        fig = go.Figure()
        
        # Add prediction heatmap
        fig.add_trace(go.Heatmap(
            z=pred_slice,
            name='Prediction',
            colorscale='RdBu',
            visible=True
        ))
        
        # Add target heatmap (initially hidden)
        fig.add_trace(go.Heatmap(
            z=target_slice,
            name='Ground Truth',
            colorscale='RdBu',
            visible=False
        ))
        
        # Add error heatmap (initially hidden)
        error = np.abs(pred_slice - target_slice)
        fig.add_trace(go.Heatmap(
            z=error,
            name='Absolute Error',
            colorscale='Reds',
            visible=False
        ))
        
        # Add dropdown menu
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=list([
                        dict(label="Prediction",
                             method="update",
                             args=[{"visible": [True, False, False]}]),
                        dict(label="Ground Truth",
                             method="update",
                             args=[{"visible": [False, True, False]}]),
                        dict(label="Absolute Error",
                             method="update",
                             args=[{"visible": [False, False, True]}])
                    ]),
                    direction="down",
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.02,
                    yanchor="top"
                ),
            ],
            title=f"Velocity Field - {axis.upper()} slice at {slice_idx}",
            xaxis_title="X",
            yaxis_title="Y"
        )
        
        return fig
    
    def run_shap_analysis(self) -> Tuple[str, go.Figure]:
        """Run SHAP analysis on current sample."""
        if not self.current_sample or not self.shap_analyzer:
            return "No sample loaded or SHAP analyzer not initialized", go.Figure()
        
        try:
            # Prepare data
            input_data = self.current_sample['input'].unsqueeze(0)
            background_data = input_data  # Simplified for demo
            
            # Run SHAP analysis
            results = self.shap_analyzer.analyze_prediction_drivers(
                background_data, input_data, max_evals=100
            )
            
            # Create visualization
            shap_values = results['shap_values'][0]  # First sample
            
            # Create SHAP heatmap for middle slice
            middle_slice = shap_values.shape[0] // 2
            shap_slice = shap_values[middle_slice, :, :]
            
            fig = go.Figure(data=go.Heatmap(
                z=shap_slice,
                colorscale='RdBu',
                zmid=0
            ))
            
            fig.update_layout(
                title="SHAP Values - Feature Importance",
                xaxis_title="X",
                yaxis_title="Y"
            )
            
            return "SHAP analysis completed successfully", fig
            
        except Exception as e:
            return f"Error in SHAP analysis: {str(e)}", go.Figure()
    
    def run_gradcam_analysis(self) -> Tuple[str, go.Figure]:
        """Run GradCAM analysis on current sample."""
        if not self.current_sample or not self.gradcam_analyzer:
            return "No sample loaded or GradCAM analyzer not initialized", go.Figure()
        
        try:
            input_tensor = self.current_sample['input'].unsqueeze(0).to(self.device)
            
            # Run GradCAM analysis
            cam_results = self.gradcam_analyzer.analyze_velocity_importance(input_tensor)
            
            # Get the first CAM result
            if cam_results:
                first_cam = list(cam_results.values())[0]
                
                # Extract middle slice
                middle_slice = first_cam.shape[0] // 2
                cam_slice = first_cam[middle_slice, :, :]
                
                fig = go.Figure(data=go.Heatmap(
                    z=cam_slice,
                    colorscale='Hot'
                ))
                
                fig.update_layout(
                    title="GradCAM - Attention Map",
                    xaxis_title="X",
                    yaxis_title="Y"
                )
                
                return "GradCAM analysis completed successfully", fig
            else:
                return "No GradCAM results generated", go.Figure()
                
        except Exception as e:
            return f"Error in GradCAM analysis: {str(e)}", go.Figure()
    
    def run_q_criterion_analysis(self) -> Tuple[str, go.Figure]:
        """Run Q-criterion analysis on current sample."""
        if not self.current_predictions:
            return "No predictions loaded", go.Figure()
        
        try:
            # Get velocity field (assuming single component for demo)
            velocity_field = self.current_predictions['prediction']
            
            # Create 3-component velocity field for Q-criterion
            # (This is simplified - in practice you'd have all 3 components)
            vel_3d = np.stack([velocity_field, velocity_field * 0.1, velocity_field * 0.1])
            
            # Run Q-criterion analysis
            results = self.q_viz.create_comprehensive_visualization(vel_3d)
            
            if 'error' in results:
                return f"Q-criterion error: {results['error']}", go.Figure()
            
            # Create simple Q-criterion slice plot
            q_field = results['q_field']
            middle_slice = q_field.shape[0] // 2
            q_slice = q_field[middle_slice, :, :]
            
            fig = go.Figure(data=go.Heatmap(
                z=q_slice,
                colorscale='RdBu',
                zmid=0
            ))
            
            fig.update_layout(
                title="Q-criterion Field",
                xaxis_title="X",
                yaxis_title="Y"
            )
            
            stats_text = f"Q-criterion Statistics:\n"
            stats_text += f"Mean: {results['q_statistics']['mean_q']:.4f}\n"
            stats_text += f"Std: {results['q_statistics']['std_q']:.4f}\n"
            stats_text += f"Positive fraction: {results['q_statistics']['positive_q_fraction']:.3f}"
            
            return stats_text, fig
            
        except Exception as e:
            return f"Error in Q-criterion analysis: {str(e)}", go.Figure()

# Initialize demo instance
demo_instance = TurbulenceDemo()

def create_gradio_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks(title="Turbulence Surrogate Modeling Demo", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("# 🌪️ Turbulence Surrogate Modeling Demo")
        gr.Markdown("Interactive demonstration of uncertainty quantification and interpretability methods for turbulence prediction.")
        
        with gr.Tab("🔧 Model Setup"):
            gr.Markdown("## Load Experiment")
            
            experiment_dropdown = gr.Dropdown(
                choices=list(demo_instance.available_experiments.keys()),
                label="Select Experiment",
                value=list(demo_instance.available_experiments.keys())[0] if demo_instance.available_experiments else None
            )
            
            load_btn = gr.Button("Load Experiment", variant="primary")
            load_status = gr.Textbox(label="Status", interactive=False)
            
            load_btn.click(
                demo_instance.load_experiment,
                inputs=[experiment_dropdown],
                outputs=[load_status]
            )
        
        with gr.Tab("📊 Data Exploration"):
            gr.Markdown("## Sample Data")
            
            with gr.Row():
                sample_idx = gr.Number(label="Sample Index", value=0, precision=0)
                load_sample_btn = gr.Button("Load Sample", variant="secondary")
            
            sample_status = gr.Textbox(label="Sample Status", interactive=False)
            sample_stats = gr.JSON(label="Sample Statistics")
            
            load_sample_btn.click(
                demo_instance.get_sample_data,
                inputs=[experiment_dropdown, sample_idx],
                outputs=[sample_status, sample_stats]
            )
            
            gr.Markdown("## Velocity Field Visualization")
            
            with gr.Row():
                axis_choice = gr.Radio(["x", "y", "z"], label="Slice Axis", value="z")
                slice_idx = gr.Slider(0, 31, value=16, step=1, label="Slice Index")
            
            velocity_plot = gr.Plot(label="Velocity Field")
            
            # Update plot when parameters change
            for component in [axis_choice, slice_idx]:
                component.change(
                    demo_instance.create_velocity_slice_plot,
                    inputs=[axis_choice, slice_idx],
                    outputs=[velocity_plot]
                )
        
        with gr.Tab("🔍 Interpretability Analysis"):
            gr.Markdown("## Model Interpretability Methods")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### SHAP Analysis")
                    shap_btn = gr.Button("Run SHAP Analysis", variant="secondary")
                    shap_status = gr.Textbox(label="SHAP Status", interactive=False)
                    shap_plot = gr.Plot(label="SHAP Values")
                    
                    shap_btn.click(
                        demo_instance.run_shap_analysis,
                        outputs=[shap_status, shap_plot]
                    )
                
                with gr.Column():
                    gr.Markdown("### GradCAM Analysis")
                    gradcam_btn = gr.Button("Run GradCAM Analysis", variant="secondary")
                    gradcam_status = gr.Textbox(label="GradCAM Status", interactive=False)
                    gradcam_plot = gr.Plot(label="Attention Map")
                    
                    gradcam_btn.click(
                        demo_instance.run_gradcam_analysis,
                        outputs=[gradcam_status, gradcam_plot]
                    )
        
        with gr.Tab("🌊 Physics Analysis"):
            gr.Markdown("## Q-criterion Analysis")
            gr.Markdown("Analyze vortical structures using Q-criterion isosurfaces.")
            
            q_btn = gr.Button("Run Q-criterion Analysis", variant="secondary")
            q_status = gr.Textbox(label="Q-criterion Status", interactive=False)
            q_plot = gr.Plot(label="Q-criterion Field")
            
            q_btn.click(
                demo_instance.run_q_criterion_analysis,
                outputs=[q_status, q_plot]
            )
        
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## About This Demo
            
            This interactive demo showcases advanced uncertainty quantification and interpretability methods for turbulence surrogate modeling:
            
            ### 🎯 **Features**
            - **Model Loading**: Load pre-trained turbulence prediction models
            - **Data Exploration**: Visualize velocity fields and prediction statistics
            - **Interpretability**: SHAP values and GradCAM attention maps
            - **Physics Analysis**: Q-criterion for vortical structure identification
            
            ### 🔬 **Methods Implemented**
            - **SINDy**: Sparse Identification of Nonlinear Dynamics
            - **Gaussian Process Regression**: Probabilistic predictions with uncertainty
            - **SWAG**: Stochastic Weight Averaging Gaussian for Bayesian deep learning
            - **Sobol Indices**: Global sensitivity analysis
            - **SHAP**: SHapley Additive exPlanations for feature importance
            - **ALE**: Accumulated Local Effects for model interpretability
            - **GradCAM**: Gradient-weighted Class Activation Mapping
            - **Q-criterion**: Vortical structure identification and visualization
            
            ### 📚 **Usage**
            1. **Setup**: Load an experiment configuration
            2. **Explore**: Select and visualize sample data
            3. **Analyze**: Run interpretability and physics analysis methods
            4. **Interpret**: Examine results and model behavior
            
            ### 🏗️ **Technical Details**
            - Built with Gradio for interactive web interface
            - PyTorch models for turbulence prediction
            - Advanced uncertainty quantification methods
            - Physics-informed analysis tools
            """)
    
    return demo

def main():
    """Main function to launch the demo."""
    
    # Create and launch the interface
    demo = create_gradio_interface()
    
    # Launch with custom settings
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True for public sharing
        debug=True,             # Enable debug mode
        show_error=True         # Show detailed errors
    )

if __name__ == "__main__":
    main()
