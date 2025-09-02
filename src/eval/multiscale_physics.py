import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable
from scipy import signal
from scipy.fft import fftn, fftfreq
import pywt
import matplotlib.pyplot as plt
from pathlib import Path

class MultiScalePhysicsValidator:
    """Multi-scale physics validation for turbulence predictions."""
    
    def __init__(self):
        """Initialize multi-scale physics validator."""
        pass
    
    def validate_energy_cascade(self, velocity_field: np.ndarray,
                               reference_field: np.ndarray = None) -> Dict[str, np.ndarray]:
        """Validate energy cascade across scales using spectral analysis."""
        
        # Ensure 3D velocity field (3, D, H, W)
        if velocity_field.shape[0] != 3:
            raise ValueError("Expected 3-component velocity field")
        
        u, v, w = velocity_field[0], velocity_field[1], velocity_field[2]
        
        # Compute 3D FFT for each component
        u_fft = fftn(u)
        v_fft = fftn(v)
        w_fft = fftn(w)
        
        # Compute energy spectrum
        energy_spectrum = 0.5 * (np.abs(u_fft)**2 + np.abs(v_fft)**2 + np.abs(w_fft)**2)
        
        # Get wavenumber grid
        kx = fftfreq(u.shape[0], d=1.0)
        ky = fftfreq(u.shape[1], d=1.0)
        kz = fftfreq(u.shape[2], d=1.0)
        
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        k_magnitude = np.sqrt(KX**2 + KY**2 + KZ**2)
        
        # Radially average the spectrum
        k_bins = np.linspace(0, np.max(k_magnitude), 50)
        k_centers = (k_bins[:-1] + k_bins[1:]) / 2
        radial_spectrum = np.zeros(len(k_centers))
        
        for i, k_center in enumerate(k_centers):
            mask = (k_magnitude >= k_bins[i]) & (k_magnitude < k_bins[i+1])
            if np.any(mask):
                radial_spectrum[i] = np.mean(energy_spectrum[mask])
        
        results = {
            'wavenumbers': k_centers,
            'energy_spectrum': radial_spectrum,
            'total_energy': np.sum(radial_spectrum)
        }
        
        # Compare with reference if provided
        if reference_field is not None:
            ref_results = self.validate_energy_cascade(reference_field)
            results['reference_spectrum'] = ref_results['energy_spectrum']
            results['spectral_error'] = np.abs(radial_spectrum - ref_results['energy_spectrum'])
            results['energy_ratio'] = np.sum(radial_spectrum) / np.sum(ref_results['energy_spectrum'])
        
        # Fit Kolmogorov -5/3 law in inertial range
        inertial_range = (k_centers > 2) & (k_centers < 20) & (radial_spectrum > 0)
        if np.any(inertial_range):
            log_k = np.log(k_centers[inertial_range])
            log_E = np.log(radial_spectrum[inertial_range])
            
            # Linear fit in log space
            slope, intercept = np.polyfit(log_k, log_E, 1)
            results['inertial_slope'] = slope
            results['kolmogorov_deviation'] = abs(slope + 5/3)
        
        return results
    
    def validate_structure_functions(self, velocity_field: np.ndarray,
                                   orders: List[int] = [2, 3, 4, 6],
                                   max_separation: int = 16) -> Dict[str, Dict]:
        """Validate velocity structure functions across scales."""
        
        if velocity_field.shape[0] != 3:
            raise ValueError("Expected 3-component velocity field")
        
        u, v, w = velocity_field[0], velocity_field[1], velocity_field[2]
        
        results = {}
        
        for order in orders:
            longitudinal_sf = []
            transverse_sf = []
            separations = []
            
            for r in range(1, min(max_separation, min(u.shape) // 2)):
                # Longitudinal structure function (parallel to separation)
                du_long = u[r:, :, :] - u[:-r, :, :]
                dv_long = v[:, r:, :] - v[:, :-r, :]
                dw_long = w[:, :, r:] - w[:, :, :-r]
                
                # Transverse structure function (perpendicular to separation)
                du_trans = u[:, r:, :] - u[:, :-r, :]
                dv_trans = v[r:, :, :] - v[:-r, :, :]
                
                # Compute structure functions
                long_sf = np.mean(np.abs(du_long)**order) + np.mean(np.abs(dv_long)**order) + np.mean(np.abs(dw_long)**order)
                trans_sf = np.mean(np.abs(du_trans)**order) + np.mean(np.abs(dv_trans)**order)
                
                longitudinal_sf.append(long_sf / 3)  # Average over components
                transverse_sf.append(trans_sf / 2)
                separations.append(r)
            
            results[f'order_{order}'] = {
                'separations': np.array(separations),
                'longitudinal': np.array(longitudinal_sf),
                'transverse': np.array(transverse_sf)
            }
            
            # Fit scaling exponents
            if len(separations) > 3:
                log_r = np.log(np.array(separations))
                log_sf_long = np.log(np.array(longitudinal_sf))
                log_sf_trans = np.log(np.array(transverse_sf))
                
                # Fit in inertial range (middle portion)
                mid_start = len(separations) // 4
                mid_end = 3 * len(separations) // 4
                
                if mid_end > mid_start:
                    long_slope = np.polyfit(log_r[mid_start:mid_end], log_sf_long[mid_start:mid_end], 1)[0]
                    trans_slope = np.polyfit(log_r[mid_start:mid_end], log_sf_trans[mid_start:mid_end], 1)[0]
                    
                    results[f'order_{order}']['longitudinal_exponent'] = long_slope
                    results[f'order_{order}']['transverse_exponent'] = trans_slope
                    
                    # Compare with Kolmogorov theory
                    kolmogorov_exponent = order / 3
                    results[f'order_{order}']['kolmogorov_deviation_long'] = abs(long_slope - kolmogorov_exponent)
                    results[f'order_{order}']['kolmogorov_deviation_trans'] = abs(trans_slope - kolmogorov_exponent)
        
        return results
    
    def validate_wavelet_decomposition(self, velocity_field: np.ndarray,
                                     wavelet: str = 'db4',
                                     levels: int = 4) -> Dict[str, Dict]:
        """Validate multi-scale structure using wavelet decomposition."""
        
        if velocity_field.shape[0] != 3:
            raise ValueError("Expected 3-component velocity field")
        
        results = {}
        
        for i, component_name in enumerate(['u', 'v', 'w']):
            component = velocity_field[i]
            
            # 3D wavelet decomposition
            coeffs = pywt.wavedecn(component, wavelet, level=levels)
            
            # Analyze energy at each scale
            scale_energies = []
            scale_names = []
            
            # Approximation coefficients (largest scale)
            approx_energy = np.sum(coeffs[0]**2)
            scale_energies.append(approx_energy)
            scale_names.append('approximation')
            
            # Detail coefficients (smaller scales)
            for level, detail_dict in enumerate(coeffs[1:]):
                level_energy = 0
                for key, detail_coeffs in detail_dict.items():
                    level_energy += np.sum(detail_coeffs**2)
                scale_energies.append(level_energy)
                scale_names.append(f'detail_level_{level+1}')
            
            # Normalize energies
            total_energy = sum(scale_energies)
            scale_energies_norm = [e / total_energy for e in scale_energies]
            
            results[component_name] = {
                'scale_names': scale_names,
                'scale_energies': scale_energies,
                'scale_energies_normalized': scale_energies_norm,
                'total_energy': total_energy
            }
            
            # Compute intermittency measures
            flatness_values = []
            for level, detail_dict in enumerate(coeffs[1:]):
                for key, detail_coeffs in detail_dict.items():
                    if detail_coeffs.size > 0:
                        # Flatness (fourth moment / second moment^2)
                        mean_val = np.mean(detail_coeffs)
                        centered = detail_coeffs - mean_val
                        second_moment = np.mean(centered**2)
                        fourth_moment = np.mean(centered**4)
                        
                        if second_moment > 0:
                            flatness = fourth_moment / (second_moment**2)
                            flatness_values.append(flatness)
            
            results[component_name]['intermittency_flatness'] = np.mean(flatness_values) if flatness_values else 0
        
        return results
    
    def validate_vorticity_dynamics(self, velocity_field: np.ndarray,
                                  reference_field: np.ndarray = None) -> Dict[str, float]:
        """Validate vorticity dynamics across scales."""
        
        if velocity_field.shape[0] != 3:
            raise ValueError("Expected 3-component velocity field")
        
        u, v, w = velocity_field[0], velocity_field[1], velocity_field[2]
        
        # Compute vorticity components
        omega_x = np.gradient(w, axis=1) - np.gradient(v, axis=2)
        omega_y = np.gradient(u, axis=2) - np.gradient(w, axis=0)
        omega_z = np.gradient(v, axis=0) - np.gradient(u, axis=1)
        
        # Vorticity magnitude
        omega_magnitude = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
        
        # Enstrophy (vorticity squared)
        enstrophy = 0.5 * (omega_x**2 + omega_y**2 + omega_z**2)
        
        # Vorticity statistics
        results = {
            'mean_vorticity_magnitude': np.mean(omega_magnitude),
            'std_vorticity_magnitude': np.std(omega_magnitude),
            'max_vorticity_magnitude': np.max(omega_magnitude),
            'total_enstrophy': np.sum(enstrophy),
            'mean_enstrophy': np.mean(enstrophy)
        }
        
        # Vorticity stretching term
        # ω · ∇u · ω (simplified approximation)
        du_dx = np.gradient(u, axis=0)
        dv_dy = np.gradient(v, axis=1)
        dw_dz = np.gradient(w, axis=2)
        
        vorticity_stretching = (omega_x * du_dx * omega_x + 
                               omega_y * dv_dy * omega_y + 
                               omega_z * dw_dz * omega_z)
        
        results['mean_vorticity_stretching'] = np.mean(vorticity_stretching)
        results['vorticity_stretching_intensity'] = np.std(vorticity_stretching)
        
        # Compare with reference if provided
        if reference_field is not None:
            ref_results = self.validate_vorticity_dynamics(reference_field)
            
            results['enstrophy_ratio'] = results['total_enstrophy'] / ref_results['total_enstrophy']
            results['vorticity_magnitude_ratio'] = (results['mean_vorticity_magnitude'] / 
                                                   ref_results['mean_vorticity_magnitude'])
            results['stretching_ratio'] = (results['mean_vorticity_stretching'] / 
                                          ref_results['mean_vorticity_stretching'])
        
        return results
    
    def validate_dissipation_scales(self, velocity_field: np.ndarray,
                                  viscosity: float = 1e-4) -> Dict[str, float]:
        """Validate dissipation scale characteristics."""
        
        if velocity_field.shape[0] != 3:
            raise ValueError("Expected 3-component velocity field")
        
        u, v, w = velocity_field[0], velocity_field[1], velocity_field[2]
        
        # Compute velocity gradients
        du_dx, du_dy, du_dz = np.gradient(u)
        dv_dx, dv_dy, dv_dz = np.gradient(v)
        dw_dx, dw_dy, dw_dz = np.gradient(w)
        
        # Strain rate tensor components
        S11 = du_dx
        S22 = dv_dy
        S33 = dw_dz
        S12 = 0.5 * (du_dy + dv_dx)
        S13 = 0.5 * (du_dz + dw_dx)
        S23 = 0.5 * (dv_dz + dw_dy)
        
        # Strain rate magnitude
        strain_rate_magnitude = np.sqrt(2 * (S11**2 + S22**2 + S33**2 + 
                                            2 * (S12**2 + S13**2 + S23**2)))
        
        # Dissipation rate
        dissipation_rate = 2 * viscosity * strain_rate_magnitude**2
        
        # Kolmogorov scales
        mean_dissipation = np.mean(dissipation_rate)
        kolmogorov_length = (viscosity**3 / mean_dissipation)**(1/4)
        kolmogorov_time = (viscosity / mean_dissipation)**(1/2)
        kolmogorov_velocity = (viscosity * mean_dissipation)**(1/4)
        
        # Taylor microscale
        kinetic_energy = 0.5 * np.mean(u**2 + v**2 + w**2)
        taylor_microscale = np.sqrt(10 * kinetic_energy / mean_dissipation)
        
        # Reynolds number based on Taylor microscale
        rms_velocity = np.sqrt(2 * kinetic_energy / 3)
        reynolds_taylor = rms_velocity * taylor_microscale / viscosity
        
        return {
            'mean_dissipation_rate': mean_dissipation,
            'std_dissipation_rate': np.std(dissipation_rate),
            'kolmogorov_length': kolmogorov_length,
            'kolmogorov_time': kolmogorov_time,
            'kolmogorov_velocity': kolmogorov_velocity,
            'taylor_microscale': taylor_microscale,
            'reynolds_taylor': reynolds_taylor,
            'kinetic_energy': kinetic_energy
        }
    
    def comprehensive_multiscale_validation(self, velocity_field: np.ndarray,
                                          reference_field: np.ndarray = None,
                                          viscosity: float = 1e-4) -> Dict[str, Dict]:
        """Run comprehensive multi-scale physics validation."""
        
        print("Running comprehensive multi-scale physics validation...")
        
        results = {}
        
        # Energy cascade validation
        print("  - Energy cascade analysis...")
        results['energy_cascade'] = self.validate_energy_cascade(velocity_field, reference_field)
        
        # Structure functions validation
        print("  - Structure functions analysis...")
        results['structure_functions'] = self.validate_structure_functions(velocity_field)
        
        # Wavelet decomposition validation
        print("  - Wavelet decomposition analysis...")
        results['wavelet_decomposition'] = self.validate_wavelet_decomposition(velocity_field)
        
        # Vorticity dynamics validation
        print("  - Vorticity dynamics analysis...")
        results['vorticity_dynamics'] = self.validate_vorticity_dynamics(velocity_field, reference_field)
        
        # Dissipation scales validation
        print("  - Dissipation scales analysis...")
        results['dissipation_scales'] = self.validate_dissipation_scales(velocity_field, viscosity)
        
        # Overall multi-scale physics score
        scores = []
        
        # Energy cascade score
        if 'kolmogorov_deviation' in results['energy_cascade']:
            cascade_score = max(0, 1 - results['energy_cascade']['kolmogorov_deviation'])
            scores.append(cascade_score)
        
        # Structure function score
        sf_deviations = []
        for order_key in results['structure_functions']:
            if 'kolmogorov_deviation_long' in results['structure_functions'][order_key]:
                sf_deviations.append(results['structure_functions'][order_key]['kolmogorov_deviation_long'])
        
        if sf_deviations:
            sf_score = max(0, 1 - np.mean(sf_deviations))
            scores.append(sf_score)
        
        # Vorticity score (based on ratios if reference available)
        if reference_field is not None and 'enstrophy_ratio' in results['vorticity_dynamics']:
            enstrophy_ratio = results['vorticity_dynamics']['enstrophy_ratio']
            vorticity_score = max(0, 1 - abs(1 - enstrophy_ratio))
            scores.append(vorticity_score)
        
        results['overall'] = {
            'multiscale_physics_score': np.mean(scores) if scores else 0.0,
            'n_metrics': len(scores)
        }
        
        return results
    
    def plot_multiscale_validation(self, results: Dict, save_path: Optional[str] = None):
        """Plot multi-scale validation results."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Energy spectrum
        if 'energy_cascade' in results:
            cascade_data = results['energy_cascade']
            k = cascade_data['wavenumbers']
            E_k = cascade_data['energy_spectrum']
            
            axes[0, 0].loglog(k[E_k > 0], E_k[E_k > 0], 'b-', label='Prediction')
            
            if 'reference_spectrum' in cascade_data:
                axes[0, 0].loglog(k[E_k > 0], cascade_data['reference_spectrum'][E_k > 0], 
                                 'r--', label='Reference')
            
            # Kolmogorov -5/3 line
            k_ref = k[(k > 2) & (k < 20)]
            if len(k_ref) > 0:
                E_ref = k_ref**(-5/3) * E_k[k > 2][0] * (k_ref[0]**(5/3))
                axes[0, 0].loglog(k_ref, E_ref, 'k:', label='-5/3 slope')
            
            axes[0, 0].set_xlabel('Wavenumber k')
            axes[0, 0].set_ylabel('Energy E(k)')
            axes[0, 0].set_title('Energy Spectrum')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Plot 2: Structure functions
        if 'structure_functions' in results:
            sf_data = results['structure_functions']
            
            for order_key in list(sf_data.keys())[:3]:  # Plot first 3 orders
                order_data = sf_data[order_key]
                r = order_data['separations']
                sf_long = order_data['longitudinal']
                
                axes[0, 1].loglog(r, sf_long, 'o-', label=f'Order {order_key.split("_")[1]}')
            
            axes[0, 1].set_xlabel('Separation r')
            axes[0, 1].set_ylabel('Structure Function')
            axes[0, 1].set_title('Velocity Structure Functions')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Plot 3: Wavelet energy distribution
        if 'wavelet_decomposition' in results:
            wavelet_data = results['wavelet_decomposition']
            
            # Plot for u-component
            if 'u' in wavelet_data:
                u_data = wavelet_data['u']
                scale_names = u_data['scale_names']
                energies = u_data['scale_energies_normalized']
                
                axes[0, 2].bar(range(len(scale_names)), energies)
                axes[0, 2].set_xlabel('Scale')
                axes[0, 2].set_ylabel('Normalized Energy')
                axes[0, 2].set_title('Wavelet Energy Distribution (u-component)')
                axes[0, 2].set_xticks(range(len(scale_names)))
                axes[0, 2].set_xticklabels(scale_names, rotation=45)
        
        # Plot 4: Vorticity statistics
        if 'vorticity_dynamics' in results:
            vort_data = results['vorticity_dynamics']
            
            metrics = ['Mean Magnitude', 'Std Magnitude', 'Total Enstrophy', 'Mean Stretching']
            values = [
                vort_data.get('mean_vorticity_magnitude', 0),
                vort_data.get('std_vorticity_magnitude', 0),
                vort_data.get('total_enstrophy', 0) / 1000,  # Scale for visibility
                vort_data.get('mean_vorticity_stretching', 0)
            ]
            
            axes[1, 0].bar(metrics, values)
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].set_title('Vorticity Dynamics')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 5: Dissipation scales
        if 'dissipation_scales' in results:
            diss_data = results['dissipation_scales']
            
            scales = ['Kolmogorov Length', 'Taylor Microscale', 'Kolmogorov Time']
            scale_values = [
                diss_data.get('kolmogorov_length', 0),
                diss_data.get('taylor_microscale', 0),
                diss_data.get('kolmogorov_time', 0)
            ]
            
            axes[1, 1].bar(scales, scale_values)
            axes[1, 1].set_ylabel('Scale Value')
            axes[1, 1].set_title('Characteristic Scales')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Plot 6: Overall scores
        if 'overall' in results:
            overall_score = results['overall']['multiscale_physics_score']
            
            # Create a gauge-like visualization
            theta = np.linspace(0, np.pi, 100)
            r = np.ones_like(theta)
            
            axes[1, 2].plot(theta, r, 'k-', linewidth=2)
            
            # Add score indicator
            score_angle = overall_score * np.pi
            axes[1, 2].plot([score_angle, score_angle], [0, 1], 'r-', linewidth=3)
            axes[1, 2].fill_between(theta[theta <= score_angle], 0, 1, alpha=0.3, color='green')
            
            axes[1, 2].set_xlim(0, np.pi)
            axes[1, 2].set_ylim(0, 1.2)
            axes[1, 2].set_title(f'Multi-Scale Physics Score: {overall_score:.3f}')
            axes[1, 2].set_xticks([0, np.pi/2, np.pi])
            axes[1, 2].set_xticklabels(['0', '0.5', '1.0'])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def generate_multiscale_report(self, results: Dict) -> str:
        """Generate multi-scale physics validation report."""
        
        report = "# Multi-Scale Physics Validation Report\n\n"
        
        # Overall assessment
        if 'overall' in results:
            overall_score = results['overall']['multiscale_physics_score']
            if overall_score > 0.8:
                assessment = "EXCELLENT"
            elif overall_score > 0.6:
                assessment = "GOOD"
            elif overall_score > 0.4:
                assessment = "MODERATE"
            else:
                assessment = "POOR"
            
            report += f"## Overall Assessment: {assessment}\n"
            report += f"- Multi-Scale Physics Score: {overall_score:.3f}\n\n"
        
        # Energy cascade analysis
        if 'energy_cascade' in results:
            cascade_data = results['energy_cascade']
            report += "## Energy Cascade Analysis\n"
            report += f"- Total Energy: {cascade_data['total_energy']:.6f}\n"
            
            if 'inertial_slope' in cascade_data:
                report += f"- Inertial Range Slope: {cascade_data['inertial_slope']:.3f}\n"
                report += f"- Kolmogorov Deviation: {cascade_data['kolmogorov_deviation']:.3f}\n"
            
            if 'energy_ratio' in cascade_data:
                report += f"- Energy Ratio (pred/ref): {cascade_data['energy_ratio']:.3f}\n"
            
            report += "\n"
        
        # Structure functions analysis
        if 'structure_functions' in results:
            sf_data = results['structure_functions']
            report += "## Structure Functions Analysis\n"
            
            for order_key in sf_data:
                order_data = sf_data[order_key]
                order = order_key.split('_')[1]
                
                if 'longitudinal_exponent' in order_data:
                    report += f"- Order {order} Longitudinal Exponent: {order_data['longitudinal_exponent']:.3f}\n"
                    report += f"- Order {order} Kolmogorov Deviation: {order_data['kolmogorov_deviation_long']:.3f}\n"
            
            report += "\n"
        
        # Vorticity dynamics
        if 'vorticity_dynamics' in results:
            vort_data = results['vorticity_dynamics']
            report += "## Vorticity Dynamics\n"
            report += f"- Mean Vorticity Magnitude: {vort_data['mean_vorticity_magnitude']:.6f}\n"
            report += f"- Total Enstrophy: {vort_data['total_enstrophy']:.6f}\n"
            report += f"- Mean Vorticity Stretching: {vort_data['mean_vorticity_stretching']:.6f}\n"
            
            if 'enstrophy_ratio' in vort_data:
                report += f"- Enstrophy Ratio (pred/ref): {vort_data['enstrophy_ratio']:.3f}\n"
            
            report += "\n"
        
        # Dissipation scales
        if 'dissipation_scales' in results:
            diss_data = results['dissipation_scales']
            report += "## Dissipation Scales\n"
            report += f"- Kolmogorov Length Scale: {diss_data['kolmogorov_length']:.6f}\n"
            report += f"- Taylor Microscale: {diss_data['taylor_microscale']:.6f}\n"
            report += f"- Reynolds Number (Taylor): {diss_data['reynolds_taylor']:.1f}\n"
            report += f"- Mean Dissipation Rate: {diss_data['mean_dissipation_rate']:.6f}\n\n"
        
        return report
