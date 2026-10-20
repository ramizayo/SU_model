#!/usr/bin/env python3
"""
SOLUCIÓN NUMÉRICA DE GROTE-HYNES Y ESCALAMIENTO ANÓMALO
======================================================
Resuelve la ecuación de Grote-Hynes para un kernel fraccional,
demostrando los tres regímenes cinéticos y el escalamiento anómalo
tau \propto eta^{2.5} en la ventana viscoelástica.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ==========================================
# 1. PARÁMETROS
# ==========================================
alpha = 0.6
tau_c = 50.0      
omega_TS = 100.0  
M = 1.0
gamma0_base = 10.0 

def gamma_hat(s, eta):
    return eta * gamma0_base * (alpha / tau_c) / (1 + s * tau_c)**alpha

def kappa_GH(eta):
    def equation(lam):
        return lam - (omega_TS**2) / (lam + gamma_hat(lam, eta)/M)
    try:
        lam_r = brentq(equation, 1e-10, omega_TS * 2)
        return lam_r / omega_TS
    except:
        return np.nan

# ==========================================
# 2. BARRIDO DE VISCOSIDAD
# ==========================================
print("Resolviendo Grote-Hynes...")
eta_vals = np.logspace(3, 8, 100)
kappa_vals = np.array([kappa_GH(e) for e in eta_vals])
tau_vals = 1.0 / kappa_vals

# ==========================================
# 3. GRÁFICA
# ==========================================
plt.figure(figsize=(8, 5))
plt.loglog(eta_vals, tau_vals, 'b-', lw=2.5, label='Grote-Hynes exact solution')

# Sombrear el régimen fraccional
plt.axvspan(3e4, 3e6, color='orange', alpha=0.2, label=r'Fractional regime ($\tau \propto \eta^{2.5}$)')

# Asíntota
eta_ref = np.array([3e4, 3e6])
tau_ref = tau_vals[np.argmin(np.abs(eta_vals - 3e4))] * (eta_ref / 3e4)**2.5
plt.loglog(eta_ref, tau_ref, 'r--', lw=2, label=r'Asymptote: \(\tau \propto \eta^{2.5}\)')

# Límites
plt.loglog(eta_vals[eta_vals < 1e4], tau_vals[eta_vals < 1e4], 'g--', lw=1, label=r'Kramers limit (\(\tau \propto \eta^0\))')
plt.loglog(eta_vals[eta_vals > 3e6], tau_vals[eta_vals > 3e6], 'k:', lw=1.5, label=r'High-friction Markovian (\(\tau \propto \eta^1\))')

plt.xlabel(r'Viscosity scaling \(\eta / \eta_0\)', fontsize=12)
plt.ylabel(r'\(\tau_{\mathrm{fold}} / \tau_0\)', fontsize=12)
plt.title('Anomalous Viscosity Scaling (Prediction P1)', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig('grafico3.png', dpi=300, bbox_inches='tight')
plt.show()
