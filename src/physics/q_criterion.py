import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
from typing import Tuple, Optional, Dict, List
import plotly.graph_objects as go
import plotly.express as px

class QCriterionAnalyzer:
    """Q-criterion analysis for turbulent flow visualization."""
    
    def __init__(self):
        pass
    
    def compute_q_criterion(self, velocity_field: np.ndarray, 
                           spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> np.ndarray:
        """
        Compute Q-criterion from velocity field.
        
        Q = 0.5 * (||Ω||² - ||S||²)
        where Ω is vorticity tensor and S is strain rate tensor
        
        Args:
            velocity_field: Velocity field (3, D, H, W) or (D, H, W, 3)
            spacing: Grid spacing in each direction
            
        Returns:
            Q-criterion field
        """
        # Ensure velocity field is in (3, D, H, W) format
        if velocity_field.shape[-1] == 3:
            velocity_field = velocity_field.transpose(3, 0, 1, 2)
        
        u, v, w = velocity_field[0], velocity_field[1], velocity_field[2]
        dx, dy, dz = spacing
        
        # Compute velocity gradients
        du_dx, du_dy, du_dz = np.gradient(u, dx, dy, dz)
        dv_dx, dv_dy, dv_dz = np.gradient(v, dx, dy, dz)
        dw_dx, dw_dy, dw_dz = np.gradient(w, dx, dy, dz)
        
        # Strain rate tensor components (symmetric part)
        S11 = du_dx
        S22 = dv_dy
        S33 = dw_dz
        S12 = 0.5 * (du_dy + dv_dx)
        S13 = 0.5 * (du_dz + dw_dx)
        S23 = 0.5 * (dv_dz + dw_dy)
        
        # Vorticity tensor components (antisymmetric part)
        Omega12 = 0.5 * (du_dy - dv_dx)
        Omega13 = 0.5 * (du_dz - dw_dx)
        Omega23 = 0.5 * (dv_dz - dw_dy)
        
        # Compute tensor magnitudes squared
        S_magnitude_sq = 2 * (S11**2 + S22**2 + S33**2 + 2 * (S12**2 + S13**2 + S23**2))
        Omega_magnitude_sq = 2 * (Omega12**2 + Omega13**2 + Omega23**2)
        
        # Q-criterion
        Q = 0.5 * (Omega_magnitude_sq - S_magnitude_sq)
        
        return Q
    
    def extract_isosurfaces(self, q_field: np.ndarray, 
                           q_threshold: float = None,
                           n_surfaces: int = 3) -> List[Dict]:
        """
        Extract Q-criterion isosurfaces using marching cubes.
        
        Args:
            q_field: Q-criterion field
            q_threshold: Q value threshold (if None, use percentile-based)
            n_surfaces: Number of isosurfaces to extract
            
        Returns:
            List of isosurface dictionaries with vertices and faces
        """
        if q_threshold is None:
            # Use positive Q values and extract multiple levels
            positive_q = q_field[q_field > 0]
            if len(positive_q) == 0:
                return []
            
            # Use percentiles for multiple surfaces
            percentiles = np.linspace(50, 95, n_surfaces)
            thresholds = [np.percentile(positive_q, p) for p in percentiles]
        else:
            thresholds = [q_threshold]
        
        isosurfaces = []
        
        for i, threshold in enumerate(thresholds):
            try:
                # Extract isosurface using marching cubes
                vertices, faces, normals, _ = measure.marching_cubes(
                    q_field, level=threshold, spacing=(1.0, 1.0, 1.0)
                )
                
                isosurfaces.append({
                    'vertices': vertices,
                    'faces': faces,
                    'normals': normals,
                    'threshold': threshold,
                    'level': i
                })
                
            except (ValueError, RuntimeError) as e:
                print(f"Warning: Could not extract isosurface at level {threshold}: {e}")
                continue
        
        return isosurfaces
    
    def visualize_isosurfaces_matplotlib(self, isosurfaces: List[Dict], 
                                       velocity_field: Optional[np.ndarray] = None,
                                       save_path: Optional[str] = None,
                                       figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        Visualize Q-criterion isosurfaces using matplotlib.
        
        Args:
            isosurfaces: List of isosurface data
            velocity_field: Optional velocity field for background
            save_path: Path to save figure
            figsize: Figure size
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(isosurfaces)))
        
        for i, surface in enumerate(isosurfaces):
            vertices = surface['vertices']
            faces = surface['faces']
            
            # Create triangular mesh
            ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                           triangles=faces, alpha=0.7, color=colors[i],
                           label=f'Q = {surface["threshold"]:.4f}')
        
        # Add velocity field background if provided
        if velocity_field is not None:
            # Sample velocity field for quiver plot
            D, H, W = velocity_field.shape[1:]
            step = max(1, min(D, H, W) // 10)
            
            x, y, z = np.meshgrid(
                np.arange(0, W, step),
                np.arange(0, H, step),
                np.arange(0, D, step),
                indexing='ij'
            )
            
            u = velocity_field[0, ::step, ::step, ::step].flatten()
            v = velocity_field[1, ::step, ::step, ::step].flatten()
            w = velocity_field[2, ::step, ::step, ::step].flatten()
            
            # Normalize for visualization
            magnitude = np.sqrt(u**2 + v**2 + w**2)
            max_mag = np.max(magnitude)
            if max_mag > 0:
                scale = 2.0 / max_mag
                u *= scale
                v *= scale
                w *= scale
            
            ax.quiver(x.flatten(), y.flatten(), z.flatten(),
                     u, v, w, alpha=0.3, length=1.0, arrow_length_ratio=0.1)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Q-criterion Isosurfaces')
        
        if len(isosurfaces) > 0:
            ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def visualize_isosurfaces_plotly(self, isosurfaces: List[Dict],
                                   save_path: Optional[str] = None) -> go.Figure:
        """
        Create interactive Q-criterion isosurface visualization using Plotly.
        
        Args:
            isosurfaces: List of isosurface data
            save_path: Path to save HTML file
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, surface in enumerate(isosurfaces):
            vertices = surface['vertices']
            faces = surface['faces']
            
            # Add mesh3d trace
            fig.add_trace(go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                opacity=0.7,
                color=colors[i % len(colors)],
                name=f'Q = {surface["threshold"]:.4f}'
            ))
        
        fig.update_layout(
            title='Interactive Q-criterion Isosurfaces',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def compute_isosurface_statistics(self, isosurfaces: List[Dict]) -> Dict:
        """Compute statistics for Q-criterion isosurfaces."""
        stats = {}
        
        for i, surface in enumerate(isosurfaces):
            vertices = surface['vertices']
            faces = surface['faces']
            
            surface_stats = {
                'n_vertices': len(vertices),
                'n_faces': len(faces),
                'threshold': surface['threshold'],
                'volume_estimate': self._estimate_volume(vertices, faces),
                'surface_area_estimate': self._estimate_surface_area(vertices, faces),
                'centroid': np.mean(vertices, axis=0).tolist(),
                'bounding_box': {
                    'min': np.min(vertices, axis=0).tolist(),
                    'max': np.max(vertices, axis=0).tolist()
                }
            }
            
            stats[f'surface_{i}'] = surface_stats
        
        return stats
    
    def _estimate_volume(self, vertices: np.ndarray, faces: np.ndarray) -> float:
        """Estimate volume of mesh using divergence theorem."""
        volume = 0.0
        
        for face in faces:
            # Get triangle vertices
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            
            # Compute triangle area and normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            area = 0.5 * np.linalg.norm(normal)
            
            if area > 0:
                # Contribution to volume (simplified)
                centroid = (v0 + v1 + v2) / 3
                volume += np.dot(centroid, normal) * area / 6
        
        return abs(volume)
    
    def _estimate_surface_area(self, vertices: np.ndarray, faces: np.ndarray) -> float:
        """Estimate surface area of mesh."""
        area = 0.0
        
        for face in faces:
            # Get triangle vertices
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            
            # Compute triangle area
            edge1 = v1 - v0
            edge2 = v2 - v0
            area += 0.5 * np.linalg.norm(np.cross(edge1, edge2))
        
        return area
    
    def analyze_q_criterion_evolution(self, velocity_sequence: np.ndarray,
                                    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Dict:
        """
        Analyze Q-criterion evolution over time.
        
        Args:
            velocity_sequence: Velocity fields over time (T, 3, D, H, W)
            spacing: Grid spacing
            
        Returns:
            Evolution analysis results
        """
        T = velocity_sequence.shape[0]
        q_evolution = []
        isosurface_stats = []
        
        for t in range(T):
            # Compute Q-criterion
            q_field = self.compute_q_criterion(velocity_sequence[t], spacing)
            q_evolution.append(q_field)
            
            # Extract isosurfaces
            isosurfaces = self.extract_isosurfaces(q_field, n_surfaces=1)
            
            if isosurfaces:
                stats = self.compute_isosurface_statistics(isosurfaces)
                isosurface_stats.append(stats)
            else:
                isosurface_stats.append({})
        
        # Compute temporal statistics
        q_evolution = np.array(q_evolution)
        
        evolution_stats = {
            'mean_q_over_time': [float(np.mean(q)) for q in q_evolution],
            'max_q_over_time': [float(np.max(q)) for q in q_evolution],
            'positive_q_fraction': [float(np.sum(q > 0) / q.size) for q in q_evolution],
            'isosurface_stats': isosurface_stats,
            'temporal_correlation': float(np.corrcoef(
                [np.mean(q) for q in q_evolution],
                range(T)
            )[0, 1]) if T > 1 else 0.0
        }
        
        return evolution_stats

class TurbulenceVisualization:
    """Comprehensive turbulence visualization including Q-criterion."""
    
    def __init__(self):
        self.q_analyzer = QCriterionAnalyzer()
    
    def create_comprehensive_visualization(self, velocity_field: np.ndarray,
                                         pressure_field: Optional[np.ndarray] = None,
                                         save_dir: Optional[str] = None) -> Dict:
        """
        Create comprehensive turbulence visualization.
        
        Args:
            velocity_field: Velocity field (3, D, H, W)
            pressure_field: Optional pressure field (D, H, W)
            save_dir: Directory to save visualizations
            
        Returns:
            Visualization results and statistics
        """
        results = {}
        
        # Compute Q-criterion
        print("Computing Q-criterion...")
        q_field = self.q_analyzer.compute_q_criterion(velocity_field)
        
        # Extract isosurfaces
        print("Extracting isosurfaces...")
        isosurfaces = self.q_analyzer.extract_isosurfaces(q_field, n_surfaces=3)
        
        if not isosurfaces:
            print("Warning: No isosurfaces found")
            return {'error': 'No isosurfaces found'}
        
        # Create visualizations
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            
            # Matplotlib visualization
            print("Creating matplotlib visualization...")
            self.q_analyzer.visualize_isosurfaces_matplotlib(
                isosurfaces, velocity_field, 
                save_path=os.path.join(save_dir, 'q_criterion_isosurfaces.png')
            )
            
            # Plotly interactive visualization
            print("Creating interactive visualization...")
            fig = self.q_analyzer.visualize_isosurfaces_plotly(
                isosurfaces,
                save_path=os.path.join(save_dir, 'q_criterion_interactive.html')
            )
            results['plotly_figure'] = fig
        
        # Compute statistics
        print("Computing isosurface statistics...")
        stats = self.q_analyzer.compute_isosurface_statistics(isosurfaces)
        
        # Additional Q-criterion statistics
        q_stats = {
            'mean_q': float(np.mean(q_field)),
            'std_q': float(np.std(q_field)),
            'min_q': float(np.min(q_field)),
            'max_q': float(np.max(q_field)),
            'positive_q_fraction': float(np.sum(q_field > 0) / q_field.size),
            'q_percentiles': {
                '25': float(np.percentile(q_field, 25)),
                '50': float(np.percentile(q_field, 50)),
                '75': float(np.percentile(q_field, 75)),
                '90': float(np.percentile(q_field, 90)),
                '95': float(np.percentile(q_field, 95))
            }
        }
        
        results.update({
            'q_field': q_field,
            'isosurfaces': isosurfaces,
            'isosurface_statistics': stats,
            'q_statistics': q_stats,
            'n_isosurfaces': len(isosurfaces)
        })
        
        return results
    
    def compare_q_criterion(self, velocity_fields: Dict[str, np.ndarray],
                           save_dir: Optional[str] = None) -> Dict:
        """
        Compare Q-criterion between different velocity fields (e.g., prediction vs truth).
        
        Args:
            velocity_fields: Dictionary of velocity fields {name: field}
            save_dir: Directory to save comparison plots
            
        Returns:
            Comparison results
        """
        results = {}
        
        for name, vel_field in velocity_fields.items():
            print(f"Analyzing Q-criterion for {name}...")
            q_field = self.q_analyzer.compute_q_criterion(vel_field)
            isosurfaces = self.q_analyzer.extract_isosurfaces(q_field, n_surfaces=2)
            
            results[name] = {
                'q_field': q_field,
                'isosurfaces': isosurfaces,
                'statistics': self.q_analyzer.compute_isosurface_statistics(isosurfaces) if isosurfaces else {}
            }
        
        # Create comparison plots
        if save_dir and len(velocity_fields) > 1:
            import os
            os.makedirs(save_dir, exist_ok=True)
            
            self._create_comparison_plots(results, save_dir)
        
        return results
    
    def _create_comparison_plots(self, results: Dict, save_dir: str):
        """Create comparison plots for multiple Q-criterion analyses."""
        # Extract statistics for comparison
        stats_comparison = {}
        
        for name, result in results.items():
            if 'statistics' in result and result['statistics']:
                # Get first surface statistics
                first_surface = list(result['statistics'].values())[0]
                stats_comparison[name] = {
                    'n_vertices': first_surface['n_vertices'],
                    'volume': first_surface['volume_estimate'],
                    'surface_area': first_surface['surface_area_estimate'],
                    'threshold': first_surface['threshold']
                }
        
        if not stats_comparison:
            return
        
        # Create comparison bar plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        names = list(stats_comparison.keys())
        metrics = ['n_vertices', 'volume', 'surface_area', 'threshold']
        titles = ['Number of Vertices', 'Volume Estimate', 'Surface Area', 'Q Threshold']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i // 2, i % 2]
            values = [stats_comparison[name][metric] for name in names]
            
            ax.bar(names, values)
            ax.set_title(title)
            ax.set_ylabel(metric.replace('_', ' ').title())
            
            # Rotate x-axis labels if needed
            if len(max(names, key=len)) > 8:
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'q_criterion_comparison.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
