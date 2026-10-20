#!/usr/bin/env python3
"""
INTEGRACIÓN DE LA GLE 1D CON MEMORIA FRACCIONAL — CI2
=====================================================
Simula el plegamiento de una partícula browniana en una coordenada de reacción Q
utilizando una Ecuación Generalizada de Langevin (GLE) con memoria fraccional.
Genera la distribución multimodal de tiempos de primer paso (FPT).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import time

# ==========================================
# 1. PARÁMETROS FÍSICOS
# ==========================================
kBT = 2.49          # kJ/mol (a 300 K)
N = 65              # Residuos de CI2

# Potencial de doble pozo
a = 0.46 * N        # ~29.9 kJ/mol
b = 7.4 * N         # ~481 kJ/mol

def F(Q):
    return -a * Q + b * (Q**2) * ((1 - Q)**2)

def dFdQ(Q):
    return -a + b * (2*Q - 6*Q**2 + 4*Q**3)

# ==========================================
# 2. KERNEL FRACCIONAL (Embedding Markoviano)
# ==========================================
alpha = 0.6
tau_c = 50.0 
N_modos = 8
tau_k = tau_c * np.logspace(-2, 2, N_modos) 
c_k = tau_k**alpha
c_k = c_k / np.sum(c_k) 

def run_gle_trajectory(gamma0, t_max=5000.0, dt=0.5):
    n_steps = int(t_max / dt)
    Q = 0.05       # Estado inicial (desplegado)
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
        
        # Criterio de plegamiento
        if Q > 0.9:
            return Q_traj[:i+1], i*dt
            
    return Q_traj, None

# ==========================================
# 3. SIMULACIÓN Y GRÁFICAS
# ==========================================
print("Simulando trayectorias (GLE 1D)...")
gamma0_test = 100.0 

# A) Trayectoria individual
traj, fpt = run_gle_trajectory(gamma0_test, t_max=20000.0, dt=1.0)

# B) Distribución de FPT
print("Simulando ensamble de 1000 trayectorias...")
n_trajs = 1000
fpts = []
for i in range(n_trajs):
    _, fpt = run_gle_trajectory(gamma0_test, t_max=50000.0, dt=1.0)
    if fpt is not None:
        fpts.append(fpt)
        
fpts = np.array(fpts)
# Calibración temporal a milisegundos (aprox 10 ms)
fpts_ms = fpts * (10.0 / np.mean(fpts))

plt.figure(figsize=(8, 4))
plt.hist(fpts_ms, bins=20, density=True, alpha=0.6, color='purple', edgecolor='k')
if len(fpts_ms) > 5:
    kde = gaussian_kde(fpts_ms)
    x_grid = np.linspace(min(fpts_ms), max(fpts_ms), 100)
    plt.plot(x_grid, kde(x_grid), 'r-', lw=2, label='Densidad KDE')
plt.xlabel('Tiempo de Plegamiento (ms)', fontsize=12)
plt.ylabel('Probabilidad', fontsize=12)
plt.title(f'Distribución de FPT (Media = {np.mean(fpts_ms):.1f} ms, σ/τ = {np.std(fpts_ms)/np.mean(fpts_ms):.2f})', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('grafico2.png', dpi=300, bbox_inches='tight')
plt.show()
