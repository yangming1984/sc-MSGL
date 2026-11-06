                       
"""Prepare a four-view EUGL suite directly from the local HDF5 datasets.

Supported layouts:
  * 10X-like dense HDF5: datasets X (cells x genes) and Y (labels)
  * scDCL HDF5: exprs/{data,indices,indptr,shape} plus obs/<label-field>

This avoids materialising a large genes x cells CSV.  The generated MAT files
are compatible with run_dataset_suite.m.
"""
from pathlib import Path
import argparse, json
import numpy as np
import h5py
from scipy import sparse
from scipy.io import savemat
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import StandardScaler


def decode(a):
    return np.asarray([x.decode(errors="replace") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in a])


def load_h5(path, label_field=None):
    with h5py.File(path, "r") as f:
        if "X" in f:
            x = np.asarray(f["X"], dtype=float)
            labels = np.asarray(f["Y"] if "Y" in f else f["y"])
        elif "exprs" in f:
            g = f["exprs"]
            shape = tuple(int(v) for v in g["shape"][:])
            data, indices, indptr = g["data"][:], g["indices"][:], g["indptr"][:]
            if len(indptr) == shape[0] + 1:
                x = sparse.csr_matrix((data, indices, indptr), shape=shape, dtype=float)
            elif len(indptr) == shape[1] + 1:
                x = sparse.csc_matrix((data, indices, indptr), shape=shape, dtype=float).tocsr()
            else:
                raise ValueError(f"Cannot infer sparse orientation for {path}: shape={shape}, indptr={len(indptr)}")
            obs = f["obs"]
            candidates = [label_field] if label_field else ["cell_type1", "cell_ontology_class", "cell_type", "celltype"]
            field = next((z for z in candidates if z and z in obs), None)
            if field is None:
                raise ValueError(f"No label field found in {path}; available={list(obs.keys())}")
            labels = obs[field][:]
        else:
            raise ValueError(f"Unsupported HDF5 layout: {path}")
    labels = decode(labels)
    if x.shape[0] != len(labels):
        raise ValueError(f"Matrix has {x.shape[0]} cells but labels have {len(labels)} rows")
    return x, labels


def pca_view(x, n):
    x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(StandardScaler().fit_transform(x), -10, 10)
    solver = "full" if max(x.shape) <= 1500 else "randomized"
    return PCA(n_components=min(n, x.shape[0] - 2, x.shape[1]), svd_solver=solver, random_state=100).fit_transform(x)


def make_views(raw_hvg, log_hvg, n):
    A = pca_view(log_hvg, n)
    mu = raw_hvg.mean(axis=0, keepdims=True)
    lib = raw_hvg.sum(axis=1, keepdims=True)
    expected = lib * (mu / np.maximum(mu.sum(axis=1, keepdims=True), 1e-12))
    B = pca_view((raw_hvg - expected) / np.sqrt(expected + 1.0), n)
    C = SpectralEmbedding(n_components=min(n, raw_hvg.shape[0] - 2),
                          n_neighbors=min(15, raw_hvg.shape[0] - 1),
                          affinity="nearest_neighbors", eigen_solver="arpack", random_state=100, n_jobs=1).fit_transform(A)
    D = pca_view((raw_hvg > 0).astype(float), n)
    return [A, B, C, D]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label-field", default=None)
    ap.add_argument("--taus", default="0,0.01,0.03,0.05,0.10")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    X, labels = load_h5(args.h5, args.label_field)
    if sparse.issparse(X):
        lib = np.asarray(X.sum(axis=1)).ravel()
        ok = lib > 0
        X = X[ok]
    else:
        lib = X.sum(axis=1)
        ok = lib > 0
        X = X[ok]
    labels = labels[ok]
    _, truth = np.unique(labels, return_inverse=True)
    grids = [(float(t), 2000, 50) for t in args.taus.split(",")]
    grids += [(0.03, 1000, 50), (0.03, 3000, 50), (0.03, 2000, 30), (0.03, 2000, 75)]
    grids = list(dict.fromkeys(grids))
    manifest = []
    for tau, n_hvg, n_pcs in grids:
        detection = np.asarray((X > 0).mean(axis=0)).ravel()
        keep = detection > tau
        Xg = X[:, keep]
        libg = np.asarray(Xg.sum(axis=1)).ravel()
        if sparse.issparse(Xg):
            logg = Xg.multiply(1e4 / np.maximum(libg, 1e-12)[:, None]).log1p().toarray()
            rawg = Xg.toarray()
        else:
            logg = np.log1p(Xg / np.maximum(libg, 1e-12)[:, None] * 1e4)
            rawg = np.asarray(Xg, dtype=float)
        var = np.var(logg, axis=0)
        order = np.argsort(var)[::-1][:min(n_hvg, logg.shape[1])]
        raw_hvg, log_hvg = rawg[:, order], logg[:, order]
        views = make_views(raw_hvg, log_hvg, n_pcs)
        key = f"tau{int(round(tau*100)):02d}_hvg{n_hvg}_pc{n_pcs}"
        cells = np.empty((1, 4), dtype=object)
        for i, v in enumerate(views): cells[0, i] = np.asarray(v, dtype=float)
        savemat(out / f"{key}.mat", {
            "X": views[0], "views": cells,
            "truth": (truth + 1).reshape(-1, 1).astype(float),
            "meta": {"dataset": args.name, "n_cells": int(X.shape[0]),
                     "n_genes_raw": int(X.shape[1]), "n_genes_after_filter": int(Xg.shape[1]),
                     "n_hvg": int(len(order)), "tau": tau, "n_pcs": n_pcs}
        }, do_compression=True)
        manifest.append({"config": key, "n_cells": int(X.shape[0]), "n_genes_raw": int(X.shape[1]),
                         "n_genes_after_filter": int(Xg.shape[1]), "n_hvg": int(len(order)),
                         "tau": tau, "n_pcs": n_pcs})
                                                                           
                                                                       
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"saved {key}: cells={X.shape[0]} genes={X.shape[1]} filtered={Xg.shape[1]} hvg={len(order)}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest: {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
