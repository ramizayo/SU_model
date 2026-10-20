# Código de simulación para el modelo S_U de plegamiento de proteínas

Este repositorio contiene los scripts utilizados para generar los resultados presentados en el manuscrito *"A Quantitative Framework for Protein Folding: Subextensive Barriers and Anomalous Kinetics in Fractional Langevin Dynamics"*.

## Contenido del repositorio:
1. **`openmm_annealing.py`**: Script de simulación de recocido (annealing) con OpenMM. Valida el colapso estructural de la proteína CI2 usando un término cooperativo \(-JQ^4\).
2. **`gle_1d_integration.py`**: Script para la integración de la Ecuación Generalizada de Langevin (GLE) en 1D con memoria fraccional. Genera la distribución multimodal de tiempos de primer paso.
3. **`grote_hynes_solver.py`**: Script que resuelve numéricamente la ecuación de Grote-Hynes. Genera la gráfica del escalamiento anómalo \(\tau \propto \eta^{2.5}\).

## Requisitos:
- Python 3.8+
- OpenMM (para `openmm_annealing.py`)
- NumPy, SciPy, Matplotlib

## Instalación:
```bash
pip install openmm numpy scipy matplotlib
