                       
"""Prepare the four-view suite for a genes x cells CSV/CSV.GZ matrix."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from scipy.io import savemat
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import StandardScaler

def pca_view(x, n):
    x=np.nan_to_num(np.asarray(x,dtype=float),nan=0.0,posinf=0.0,neginf=0.0)
    x=np.clip(StandardScaler().fit_transform(x),-10,10)
    solver='full' if max(x.shape)<=1500 else 'randomized'
    return PCA(n_components=min(n,x.shape[0]-2,x.shape[1]),svd_solver=solver,random_state=100).fit_transform(x)

def make_views(raw_hvg, log_hvg, n):
    A=pca_view(log_hvg,n)
    mu=raw_hvg.mean(axis=0,keepdims=True); lib=raw_hvg.sum(axis=1,keepdims=True)
    expected=lib*(mu/np.maximum(mu.sum(axis=1,keepdims=True),1e-12))
    B=pca_view((raw_hvg-expected)/np.sqrt(expected+1.0),n)
    C=SpectralEmbedding(n_components=min(n,raw_hvg.shape[0]-2),n_neighbors=min(15,raw_hvg.shape[0]-1),affinity='nearest_neighbors',eigen_solver='arpack',random_state=100,n_jobs=1).fit_transform(A)
    D=pca_view((raw_hvg>0).astype(float),n)
    return [A,B,C,D]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--name',required=True); ap.add_argument('--matrix',required=True); ap.add_argument('--labels',required=True); ap.add_argument('--scale',choices=['counts','tpm','rpkm'],default='counts'); ap.add_argument('--out-dir',default=None); ap.add_argument('--taus',default='0,0.01,0.03,0.05,0.10'); args=ap.parse_args()
    out=Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent/'data'/f'{args.name.lower()}_suite'; out.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(args.matrix,index_col=0)
    label_series=pd.read_csv(args.labels).iloc[:,0]
    if raw.shape[1]!=len(label_series): raise ValueError(f'matrix has {raw.shape[1]} cells but labels has {len(label_series)} rows')
    label_text=label_series.astype('string')
    labeled=label_text.notna() & label_text.str.strip().ne('') & label_text.str.lower().ne('na')
    if int((~labeled).sum())>0:
        print(f'dropping unlabeled cells: {int((~labeled).sum())}')
        raw=raw.loc[:,labeled.to_numpy()]
    labels=label_text[labeled].astype(str).to_numpy()
    Xc=raw.T.astype(float); ok=Xc.sum(axis=1).to_numpy()>0; Xc=Xc.loc[ok]; labels=labels[ok]; _,truth=np.unique(labels,return_inverse=True)
    grid=[(float(t),2000,50) for t in args.taus.split(',')]+[(0.03,1000,50),(0.03,3000,50),(0.03,2000,30),(0.03,2000,75)]; grid=list(dict.fromkeys(grid))
                                                                    
                                                                        
                                                                        
    manifest=[]
    for tau,n_hvg,n_pcs in grid:
        keep=(Xc.to_numpy()>0).mean(axis=0)>tau; Xg=Xc.loc[:,keep]
        if args.scale=='counts':
            log_x=np.log1p(Xg.div(Xg.sum(axis=1).replace(0,1),axis=0)*1e4); flavor='variance'
        else:
            log_x=np.log1p(Xg); flavor='variance'
        variances=log_x.var(axis=0).to_numpy()
        order=np.argsort(variances)[::-1][:min(n_hvg,log_x.shape[1])]
        hvg=np.zeros(log_x.shape[1],dtype=bool); hvg[order]=True
        views=make_views(Xg.loc[:,hvg].to_numpy(),log_x.loc[:,hvg].to_numpy(),n_pcs); key=f'tau{int(round(tau*100)):02d}_hvg{n_hvg}_pc{n_pcs}'
        cells=np.empty((1,4),dtype=object)
        for i,v in enumerate(views): cells[0,i]=np.asarray(v,dtype=float)
        savemat(out/f'{key}.mat',{'X':views[0],'views':cells,'truth':(truth+1).reshape(-1,1).astype(float),'meta':{'dataset':args.name,'scale':args.scale,'tau':tau,'n_hvg':n_hvg,'n_pcs':n_pcs,'hvg_flavor':flavor,'n_cells':int(Xg.shape[0]),'n_genes_raw':int(raw.shape[0]),'n_genes_after_dr':int(keep.sum())}},do_compression=True)
        manifest.append({'config':key,'n_cells':int(Xg.shape[0]),'tau':tau,'n_hvg':n_hvg,'n_pcs':n_pcs,'n_genes_after_dr':int(keep.sum()),'hvg_flavor':flavor}); print(f'saved {key}: cells={Xg.shape[0]} genes={keep.sum()} hvg={hvg.sum()} pcs={n_pcs}')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8'); print(f'manifest: {out/"manifest.json"}')

if __name__=='__main__': main()
