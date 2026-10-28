#!/usr/bin/env python3
"""
CONSTANT-TEMPERATURE DYNAMICS AND MULTIMODAL KINETICS
====================================================
Integrates the overdamped 1D GLE using Markovian embedding (Prony series).
Generates Figure 3 (FPT Distribution in ms).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

kBT = 2.49          
N = 65              
a = 0.46 * N        
b = 7.4 * N         

def dFdQ(Q):
    return -a + b * (2*Q - 6*Q**2 + 4*Q**3)

# Fractional Kernel Parameters
alpha = 0.6
tau_c = 50.0 
N_modos = 8 # 8 modes for better convergence as requested by reviewer
tau_k = tau_c * np.logspace(-2, 2, N_modos) 
c_k = tau_k**alpha
c_k = c_k / np.sum(c_k) 

def run_gle_trajectory(gamma0, t_max=5000.0, dt=0.5):
    n_steps = int(t_max / dt)
    Q = 0.05      
    z = np.zeros(N_modos) 
    Q_traj = np.zeros(n_steps)
    Q_traj[0] = Q
    
    noise_amp_Q = np.sqrt(2 * gamma0 * kBT / dt)
    noise_amp_z = np.sqrt(2 * kBT * c_k / (tau_k * dt))
    
    for i in range(1, n_steps):
        xi_Q = np.random.normal(0, 1) * noise_amp_Q
        xi_z = np.random.normal(0, 1, N_modos) * noise_amp_z
        
        force = -dFdQ(Q) - np.sum(z) + xi_Q
        dQ = (force / gamma0) * dt
        Q += dQ
        
        if Q < 0: Q = 0.0
        if Q > 1: Q = 1.0
        
        z += -z / tau_k * dt + (c_k / tau_k) * dQ + xi_z * dt
        Q_traj[i] = Q
        
        if Q > 0.9:
            return Q_traj[:i+1], i*dt
            
    return Q_traj, None

print("Simulating 1,000 trajectories...")
gamma0_test = 100.0 
fpts = []
for i in range(1000):
    _, fpt = run_gle_trajectory(gamma0_test, t_max=50000.0, dt=1.0)
    if fpt is not None:
        fpts.append(fpt)
        
fpts = np.array(fpts)
fpts_ms = fpts * (10.0 / np.mean(fpts)) # Calibrate to 10 ms (experimental CI2 time)

plt.figure(figsize=(8, 4))
plt.hist(fpts_ms, bins=20, density=True, alpha=0.6, color='purple', edgecolor='k')
kde = gaussian_kde(fpts_ms)
x_grid = np.linspace(min(fpts_ms), max(fpts_ms), 100)
plt.plot(x_grid, kde(x_grid), 'r-', lw=2, label='KDE Density')
plt.xlabel('Folding Time (ms)', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.title(f'FPT Distribution (Mean = {np.mean(fpts_ms):.1f} ms, σ/τ = {np.std(fpts_ms)/np.mean(fpts_ms):.2f})', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('grafico2.png', dpi=300, bbox_inches='tight')
plt.show()
