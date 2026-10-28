#!/usr/bin/env python3
"""
THERMODYNAMIC VALIDATION OF THE S_U MODEL - CI2
===============================================
This script validates the capillary free energy F(Q) from Eq. (1) of the manuscript.
It performs 10 independent simulated annealing runs using the EXACT F(Q) as a CustomCVForce.
Outputs: figura1.png (RMSD distribution) and figura2.png (3D overlay).
"""

import numpy as np
import openmm as mm
from openmm import app
import openmm.unit as unit
import os
import urllib.request
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. DOWNLOAD AND LOAD CI2
pdb_file = '2CI2.pdb'
if not os.path.exists(pdb_file):
    urllib.request.urlretrieve("https://files.rcsb.org/download/2CI2.pdb", pdb_file)

pdb = app.PDBFile(pdb_file)
forcefield = app.ForceField('amber99sbildn.xml', 'amber99_obc.xml')
modeller = app.Modeller(pdb.topology, pdb.positions)
modeller.deleteWater()
modeller.addHydrogens(forcefield)

chains = list(modeller.topology.chains())
best_chain = max(chains, key=lambda c: sum(1 for a in c.atoms() if a.name == 'CA'))
chains_to_del = [c.index for c in chains if c.index != best_chain.index]
if chains_to_del:
    modeller.deleteChains(chains_to_del)

ca_atoms = [atom for atom in modeller.topology.atoms() if atom.name == 'CA']
ca_indices = [atom.index for atom in ca_atoms]
N = len(ca_indices)
native_coords = np.array(modeller.positions.value_in_unit(unit.nanometers))[ca_indices]

# 2. NATIVE CONTACTS
RC_NM = 0.75
SEP_MIN = 4
def compute_distances(coords):
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1))

dmat_native = compute_distances(native_coords)
native_contacts = [(i, j) for i in range(N) for j in range(i + SEP_MIN, N) if dmat_native[i, j] < RC_NM]
n_contacts = len(native_contacts)

def compute_Q(coords):
    formed = sum(1 for (i, j) in native_contacts if np.linalg.norm(coords[i] - coords[j]) < RC_NM)
    return formed / n_contacts

# 3. SIMULATED ANNEALING WITH EXACT F(Q)
# F(Q) = -a*N*Q + sigma*N^(2/3)*Q^(2/3)*(1-Q)^(2/3)
a_param = 0.46 * N
sigma_param = 7.4 * (N**(2/3))

rmsd_runs = []
best_overall_rmsd = np.inf
best_overall_pos = None

print("Running 10 independent annealing trajectories (GPU recommended)...")
for run in range(10):
    np.random.seed(run)
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    
    # Contact counting for Q
    contact_force = mm.CustomBondForce(f"step({RC_NM} - r)")
    for (i, j) in native_contacts:
        contact_force.addBond(ca_indices[i], ca_indices[j], [])
    Q_cv = mm.CustomCVForce("sum_contacts / N_contacts")
    Q_cv.addCollectiveVariable("sum_contacts", contact_force)
    Q_cv.addGlobalParameter("N_contacts", float(n_contacts))

    # EXACT F(Q) from Eq. (1) (clamped to avoid division by zero)
    capillary_energy = mm.CustomCVForce("-a*N*Q + sigma*N^(2/3)*max(1e-6, Q)^(2/3)*max(1e-6, 1-Q)^(2/3)")
    capillary_energy.addCollectiveVariable("Q", Q_cv)
    capillary_energy.addGlobalParameter("a", a_param)
    capillary_energy.addGlobalParameter("sigma", sigma_param)
    capillary_energy.addGlobalParameter("N", float(N))
    system.addForce(capillary_energy)

    integrator = mm.LangevinIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 0.001 * unit.picoseconds)
    simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy()

    # Annealing
    integrator.setTemperature(500 * unit.kelvin)
    simulation.step(2000) 
    for T in np.arange(500, 0, -50): 
        integrator.setTemperature(T * unit.kelvin)
        simulation.step(1000)

    simulation.minimizeEnergy()
    final_pos = simulation.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometers)[ca_indices]
    
    # RMSD Alignment
    pos_c = final_pos - final_pos.mean(axis=0)
    nat_c = native_coords - native_coords.mean(axis=0)
    U, s, Vt = np.linalg.svd(pos_c.T @ nat_c)
    R = Vt.T @ U.T
    final_rmsd = np.sqrt(np.mean(np.sum((pos_c @ R - nat_c)**2, axis=1))) * 10 # in Å
    rmsd_runs.append(final_rmsd)
    print(f"Run {run+1}/10: RMSD = {final_rmsd:.2f} Å")

    if final_rmsd < best_overall_rmsd:
        best_overall_rmsd = final_rmsd
        best_overall_pos = final_pos.copy()

# 4. PLOTTING (Figure 1: RMSD Distribution)
plt.figure(figsize=(7, 5))
plt.hist(rmsd_runs, bins=5, density=True, alpha=0.6, color='blue', edgecolor='k')
plt.axvline(np.mean(rmsd_runs), color='r', ls='--', label=f'Mean = {np.mean(rmsd_runs):.2f} Å')
plt.xlabel('Final RMSD (Å)', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.title('RMSD Distribution over 10 Annealing Runs', fontsize=14)
plt.legend()
plt.savefig('figura1.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. PLOTTING (Figure 2: 3D Overlay)
if best_overall_pos is not None:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(best_overall_pos[:,0]*10, best_overall_pos[:,1]*10, best_overall_pos[:,2]*10, 'b-', lw=2, label='OpenMM (Best of 10)')
    ax.plot(native_coords[:,0]*10, native_coords[:,1]*10, native_coords[:,2]*10, 'r--', lw=2, alpha=0.7, label='Native (PDB)')
    ax.set_title(f'3D Structure of CI2 (RMSD = {best_overall_rmsd:.2f} Å)', fontsize=14)
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.legend()
    plt.savefig('figura2.png', dpi=300, bbox_inches='tight')
    plt.show()
