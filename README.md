# scMSGL: Scale-coordinated Multi-view Graph Learning for single-cell RNA-seq clustering

This is the code for the paper "scMSGL: scale-coordinated multi-view graph learning for single-cell RNA-seq clustering".

scMSGL takes a multi-view representation of a single-cell RNA-seq dataset (four views constructed from one expression matrix: PCA of log-expression, PCA of Pearson residuals, spectral embedding, and PCA of the detection pattern), aligns the view scales by Frobenius normalization, and learns a unified cell-to-cell graph with a weighted tensor Schatten p-norm approximation inside an ADMM loop. The fused graph is clustered by spectral clustering and evaluated with ACC / NMI / ARI.

## Requirements

- MATLAB R2024b (the code uses string concatenation and `contains`; R2016b or newer should also work, R2024b is what we tested)
- Python 3 with `numpy`, `pandas`, `scipy`, `scikit-learn`, `h5py` (only needed if you want to regenerate the suite MAT files from the raw data; the processed `dataset/` files are included so you can skip this step)

## Quick start

Open MATLAB in this folder and run one line per dataset:

```matlab
run_dataset_suite('klein')
run_dataset_suite('pollen')
run_dataset_suite('quake_10x_limb_muscle')
run_dataset_suite('adam')
```

For a single quick run of one dataset and one configuration:

```matlab
run_MSGL('pollen')                        % default config: tau03_hvg2000_pc50
run_MSGL('klein','tau03_hvg2000_pc50')
```

`run_dataset_suite` scans every `tau*_hvg*_pc*.mat` configuration of the dataset, applies the fixed mainline parameter set (alpha=300, gamma=40, beta=0.1, p_s=0.9, d=6, Frobenius view alignment), runs 10 repetitions (seeds 101-110) per configuration, and writes results to `results/<dataset>/`:

- `<dataset>_mainline_results.csv` — one row per (config, rep): ACC, NMI, ARI, iteration count and stop reason
- `<dataset>_mainline_diagnostics.csv` — view alignment diagnostics per configuration
- `checkpoint.mat` — completed runs; rerunning the script resumes and skips finished rows

As we fix the random seed in the released code, the results are reproducible on the same platform and might differ slightly across platforms due to floating-point rounding.

## Data

`dataset/` contains four preprocessed scRNA-seq suites, one folder per dataset. Each suite holds several `tauXX_hvgYYYY_pcZZ.mat` files (detection-rate filter tau, number of highly variable genes, number of principal components); each MAT file contains `views` (1xV cell of N x d_v matrices), `X` (the first view) and `truth` (1..K labels).

| Dataset | Suite folder | Source data |
|---|---|---|
| Klein | `dataset/klein_suite` | `preprocessing/raw/Klein/data.h5` |
| Pollen | `dataset/pollen_suite` | `preprocessing/raw/Pollen/` |
| Quake 10x limb muscle | `dataset/quake_10x_limb_muscle_suite` | `preprocessing/raw/Quake_10x_Limb_Muscle/data.h5` |
| Adam | `dataset/adam_suite` | `preprocessing/raw/Adam/data.h5` |

To regenerate the suite MAT files from the raw data instead of using the shipped ones:

```bash
python preprocessing/prepare_h5_suite.py --name Klein --h5 preprocessing/raw/Klein/data.h5 --out-dir dataset/klein_suite
python preprocessing/prepare_matrix_suite.py --name Pollen --matrix preprocessing/raw/Pollen/Pollen_matrix.csv.gz --labels preprocessing/raw/Pollen/Pollen_labels.csv --scale counts --out-dir dataset/pollen_suite
```

(the `--h5` variant covers `data.h5` input; the same script handles the Adam and Quake 10x limb muscle suites.)

## Code layout

```
MSGL.m                    the proposed algorithm (ADMM multi-view graph learning)
run_MSGL.m                single-run entry
run_dataset_suite.m       main experiment runner (all configs x 10 reps, checkpointed)
funs/                     kernels, solvers, spectral clustering, metrics
twist/                    auxiliary solver
ClusteringMeasure/        Accuracy / NMI / ARI measures
preprocessing/            Python suite generators + raw data
```

## Citation

If you use this code, please cite:

```
@misc{scMSGL,
  title={scMSGL: scale-coordinated multi-view graph learning for single-cell RNA-seq clustering},
  author={TODO: fill in after submission},
  note={Preprint / in review},
  year={2026}
}
```
