#!/usr/bin/env python3
"""
VALIDACIÓN TERMODINÁMICA Y ESTRUCTURAL DEL MODELO S_U — CI2
============================================================
Busca la estructura de mínima energía libre F[q] = U_seq - J·Q^4
mediante recocido simulado. Parte de la estructura nativa, la despliega
térmicamente y luego la vuelve a plegar.
"""

# Asegurar la instalación de OpenMM en el entorno actual
import sys
try:
    import openmm as mm
except ImportError:
    print("Instalando OpenMM de forma automática...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openmm"])
    import openmm as mm

import numpy as np
from openmm import app
import openmm.unit as unit
import os
import urllib.request
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# 1. DESCARGAR Y CARGAR CI2
# ============================================================
pdb_file = '2CI2.pdb'
if not os.path.exists(pdb_file):
    print("Descargando 2CI2.pdb...")
    urllib.request.urlretrieve("https://files.rcsb.org/download/2CI2.pdb", pdb_file)

pdb = app.PDBFile(pdb_file)
forcefield = app.ForceField('amber99sbildn.xml', 'amber99_obc.xml')

# Preparar sistema
modeller = app.Modeller(pdb.topology, pdb.positions)
modeller.deleteWater()
modeller.addHydrogens(forcefield)

# Conservar solo la cadena con más Cα
chains = list(modeller.topology.chains())
best_chain = max(chains, key=lambda c: sum(1 for a in c.atoms() if a.name == 'CA'))
chains_to_del = [c.index for c in chains if c.index != best_chain.index]
if chains_to_del:
    modeller.deleteChains(chains_to_del)

# Extraer índices y coordenadas de los Cα
ca_atoms = [atom for atom in modeller.topology.atoms() if atom.name == 'CA']
ca_indices = [atom.index for atom in ca_atoms]
N = len(ca_indices)
native_coords = np.array(modeller.positions.value_in_unit(unit.nanometers))[ca_indices]
print(f"CI2 cargada: {N} residuos")

# ============================================================
# 2. DEFINIR CONTACTOS NATIVOS Y PARÁMETRO Q
# ============================================================
RC_NM = 0.75       # 7.5 Å
SEP_MIN = 4        # separación secuencial mínima

def compute_distances(coords):
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1))

dmat_native = compute_distances(native_coords)

native_contacts = []
for i in range(N):
    for j in range(i + SEP_MIN, N):
        if dmat_native[i, j] < RC_NM:
            native_contacts.append((i, j))
n_contacts = len(native_contacts)
print(f"Contactos nativos definidos: {n_contacts}")

def compute_Q(coords):
    formed = sum(1 for (i, j) in native_contacts
                 if np.linalg.norm(coords[i] - coords[j]) < RC_NM)
    return formed / n_contacts

# ============================================================
# 3. RECOCODO SIMULADO PARA CADA VALOR DE J
# ============================================================
J_values = [2.0, 5.0, 10.0, 15.0, 20.0]
results = []

best_overall_rmsd = np.inf
best_overall_pos = None
best_overall_J = None

for J_val in J_values:
    print(f"\n{'='*50}")
    print(f"Probando J = {J_val} kJ/mol")
    print('='*50)

    # Crear sistema base
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds
    )

    # Fuerza de contacto nativo
    contact_force = mm.CustomBondForce(f"step({RC_NM} - r)")
    for (i, j) in native_contacts:
        contact_force.addBond(ca_indices[i], ca_indices[j], [])

    Q_cv = mm.CustomCVForce("sum_contacts / N_contacts")
    Q_cv.addCollectiveVariable("sum_contacts", contact_force)
    Q_cv.addGlobalParameter("N_contacts", float(n_contacts))

    coop_force = mm.CustomCVForce(f"-{J_val} * Q_val^4")
    coop_force.addCollectiveVariable("Q_val", Q_cv)
    system.addForce(coop_force)

    # Integrador (paso más pequeño para estabilidad)
    integrator = mm.LangevinIntegrator(
        300 * unit.kelvin,
        1.0 / unit.picosecond,
        0.001 * unit.picoseconds # Reducido para evitar NaN
    )

    simulation = app.Simulation(modeller.topology, system, integrator)

    # COMENZAR DESDE LA ESTRUCTURA NATIVA (Evita el colapso de la cadena extendida)
    simulation.context.setPositions(modeller.positions)

    # Minimización preliminar para relajar choques
    print("  Minimizando energía inicial...")
    simulation.minimizeEnergy()

    # Calentamiento para desplegar (500 K)
    print("  Calentando a 500 K para desplegar...")
    integrator.setTemperature(500 * unit.kelvin)
    simulation.step(10000)

    # Recocido: de 500 K a 1 K
    best_rmsd = np.inf
    best_Q = 0.0
    for T in np.arange(500, 0, -20):
        integrator.setTemperature(T * unit.kelvin)
        simulation.step(5000)
        state = simulation.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)[ca_indices]
        Q = compute_Q(pos)

        # Alinear y calcular RMSD
        pos_c = pos - pos.mean(axis=0)
        nat_c = native_coords - native_coords.mean(axis=0)
        U, s, Vt = np.linalg.svd(pos_c.T @ nat_c)
        R = Vt.T @ U.T
        rmsd = np.sqrt(np.mean(np.sum((pos_c @ R - nat_c)**2, axis=1)))

        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_Q = Q

    # Minimización final
    simulation.minimizeEnergy()
    state = simulation.context.getState(getPositions=True)
    final_pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)[ca_indices]
    final_Q = compute_Q(final_pos)

    pos_c = final_pos - final_pos.mean(axis=0)
    nat_c = native_coords - native_coords.mean(axis=0)
    U, s, Vt = np.linalg.svd(pos_c.T @ nat_c)
    R = Vt.T @ U.T
    final_rmsd = np.sqrt(np.mean(np.sum((pos_c @ R - nat_c)**2, axis=1)))

    results.append((J_val, final_rmsd * 10, best_rmsd * 10, best_Q, final_Q))
    print(f"  RMSD final = {final_rmsd*10:.2f} Å")
    print(f"  Mejor RMSD = {best_rmsd*10:.2f} Å")
    print(f"  Q óptimo = {best_Q:.3f}")
    print(f"  Q final = {final_Q:.3f}")

    if (final_rmsd * 10) < best_overall_rmsd:
        best_overall_rmsd = final_rmsd * 10
        best_overall_pos = final_pos.copy()
        best_overall_J = J_val

# ============================================================
# 4. TABLA RESUMEN
# ============================================================
print("\n" + "="*70)
print("  RESULTADOS DE LA VALIDACIÓN TERMODINÁMICA")
print("="*70)
print(f"  {'J (kJ/mol)':<12} {'RMSD final (Å)':<18} {'Mejor RMSD (Å)':<18} "
      f"{'Q óptimo':<10} {'Q final':<10}")
print("  " + "-"*68)
for J, fr, br, bq, fq in results:
    print(f"  {J:<12.1f} {fr:<18.2f} {br:<18.2f} {bq:<10.3f} {fq:<10.3f}")
print("="*70)
print(f"\n  MEJOR CONFIGURACIÓN: J = {best_overall_J} kJ/mol")
print(f"  RMSD MÍNIMO ALCANZADO: {best_overall_rmsd:.2f} Å")

# ============================================================
# 5. GRÁFICAS
# ============================================================
J_vals = [r[0] for r in results]
rmsd_final = [r[1] for r in results]
rmsd_best = [r[2] for r in results]

plt.figure(figsize=(7, 5))
plt.plot(J_vals, rmsd_best, 'bo-', lw=2, label='Mejor RMSD visitado')
plt.plot(J_vals, rmsd_final, 'rs--', lw=2, label='RMSD después de minimización')
plt.axhline(y=best_overall_rmsd, color='k', ls=':', label=f'Mínimo global ({best_overall_rmsd:.2f} Å)')
plt.xlabel('Coeficiente de cooperatividad J (kJ/mol)', fontsize=12)
plt.ylabel('RMSD (Å)', fontsize=12)
plt.title('Funnel del Paisaje Energético (CI2)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('fig_thermo_rmsd.png', dpi=300, bbox_inches='tight')
plt.show()

if best_overall_pos is not None:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(best_overall_pos[:,0]*10, best_overall_pos[:,1]*10, best_overall_pos[:,2]*10,
            'b-', lw=2, label=f'Simulación OpenMM (J={best_overall_J})')
    ax.plot(native_coords[:,0]*10, native_coords[:,1]*10, native_coords[:,2]*10,
            'r--', lw=2, alpha=0.7, label='Nativa (PDB)')
    ax.set_title(f'Estructura 3D de CI2 (RMSD = {best_overall_rmsd:.2f} Å)', fontsize=14)
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.legend()
    plt.savefig('fig_thermo_3d.png', dpi=300, bbox_inches='tight')
    plt.show()
