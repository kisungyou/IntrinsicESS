# Intrinsic ESS for manifold-valued MCMC output

This repository contains code for the numerical experiments in the paper

> Intrinsic effective sample size for manifold-valued Markov chain Monte Carlo output via kernel discrepancy

The experiments are intentionally sphere-only.  They illustrate the proposed intrinsic kernel effective sample size (ESS) using the Schoenberg-valid kernel on the two-sphere,
$$
k_\rho(x,y) = \{1 - 2\rho x^\top y + \rho^2\}^{-1/2}, \qquad 0 < \rho < 1.
$$

This kernel is positive definite on $\mathbb{S}^2$ because it has the Legendre expansion
$$
k_\rho(x,y) = \sum_{\ell=0}^\infty \rho^\ell P_\ell(x^\top y),
$$
where  $P_\ell$ are Legendre polynomials.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── src.py
├── notebooks/
│   ├── experiment1_rotation_invariance.ipynb
│   └── experiment2_spherical_mixture.ipynb
├── outputs/
│   ├── exp1_quick/          # created by the quick run in the executed notebook
│   └── exp2_quick/          # created by the quick run in the executed notebook
└── figures/
    ├── exp1_quick/
    └── exp2_quick/
```

The file `src.py` contains the reusable computational backbone: simulation on $\mathbb{S}^2$, exact density evaluation for von Mises--Fisher mixtures, Metropolis--Hastings kernels, kernel ESS estimation, empirical MMD calculations, and plotting helpers.

## Installation

Create and activate a virtual environment, then install the required Python packages:

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows PowerShell

pip install -r requirements.txt
```

The code uses only standard scientific Python packages.  No SciPy dependency is required.

## Running the notebooks

Start Jupyter from the repository root:

```bash
jupyter notebook
```

Then open the notebooks in the `notebooks/` directory.

Each notebook begins with the flag

```python
RUN_FULL_EXPERIMENT = False
```

The default value `False` runs a simplified version that finishes quickly.  To reproduce the manuscript-scale results, set

```python
RUN_FULL_EXPERIMENT = True
```

and rerun the notebook.

The notebooks also define

```python
RERUN_SIMULATION = True
```

Set this to `False` if output files already exist and you only want to reload summaries and regenerate figures.

## Experiment 1: rotation invariance

Notebook:

```text
notebooks/experiment1_rotation_invariance.ipynb
```

This experiment runs one Markov chain on $\mathbb{S}^2$, rotates the retained path by independent random rotations, and compares coordinate-wise scalar ESS with intrinsic kernel ESS.  The coordinate-wise ESS changes with the chosen coordinate frame, while the kernel ESS is invariant up to numerical roundoff.

Full manuscript settings:

```text
burn-in:                  1000
retained draws:           3000
number of rotations:      80
kernel scale rho:         0.75
target vMF concentration: 12
proposal concentration:   35
```

Main outputs:

```text
outputs/exp1_full/exp1_rotation_ess.csv
outputs/exp1_full/exp1_config_summary.json
figures/exp1_full/fig1.png
figures/exp1_full/fig1.pdf
```

## Experiment 2: multimodal spherical mixture

Notebook:

```text
notebooks/experiment2_spherical_mixture.ipynb
```

This experiment compares a local von Mises--Fisher random-walk Metropolis chain with a broader independence-mixture Metropolis--Hastings chain on a four-component unequal-weight mixture on $\mathbb{S}^2$.  It reports intrinsic kernel ESS over multiple kernel scales and compares the estimated long-run risk constant with corrected empirical MMD error against an iid reference sample.

Full manuscript settings:

```text
burn-in:                      1000
retained draws per chain:     2500
independent replications:     20
iid reference sample size:    8000
kernel scales rho:            0.35, 0.60, 0.85
local proposal concentration: 90
independence concentration:   12
```

Main outputs:

```text
outputs/exp2_full/exp2_results.csv
outputs/exp2_full/exp2_mode_frequencies.csv
outputs/exp2_full/exp2_reference_mode_probs.csv
outputs/exp2_full/exp2_config_summary.json
figures/exp2_full/fig2.png
figures/exp2_full/fig2.pdf
figures/exp2_full/fig3.png
figures/exp2_full/fig3.pdf
```

## Reproducibility notes

The notebooks are stored with quick-run outputs so that GitHub displays the intermediate tables and figures immediately.  To reproduce the manuscript-scale results, set `RUN_FULL_EXPERIMENT = True` in each notebook and rerun all cells.

The full experiment can take noticeably longer than the quick run because Experiment 2 computes empirical MMD terms against an iid reference sample for several kernel scales and replications.

## Citation

If you use this code, please cite the accompanying manuscript.
