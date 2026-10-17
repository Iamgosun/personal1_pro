"""Source-backed mechanism figures S1--S4; no invented curves or chart placeholders.

Each figure is scoped by dataset and shot, avoiding mixing unrelated class sets.
Only complete, saved native runs are considered. Charts need matplotlib/numpy.
"""
from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from pathlib import Path


def rows(path):
    with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def json_load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def ecdf(xs):
    import numpy as np
    v=np.sort(np.asarray(xs,dtype=float));return v,np.arange(1,len(v)+1)/len(v)


def make_figures(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    root=Path(root);out=root/'figures';out.mkdir(parents=True,exist_ok=True)
    saved=[]
    # S1: allow 1 seed, sparse epochs, and no val loader. No best-val selection.
    trace=defaultdict(list)
    for path in root.glob('native/**/trace/trajectory.csv'):
        records=rows(path)
        if records:trace[(records[0]['dataset'],records[0]['shot'])].extend(records)
    for (dataset,shot),data in sorted(trace.items()):
        val_present=any(r.get('rep_val_acc') not in (None,'')
                        for r in data if r.get('method') in ('original','branch'))
        if val_present:
            fig,axs=plt.subplots(1,2,figsize=(10,3.6))
            ax_rho,ax_val=axs
        else:
            fig,ax_rho=plt.subplots(figsize=(5.8,3.6))
            ax_val=None
            print(f'[S1] {dataset} shot={shot}: Rep validation missing; saved responsibility-only figure')
        rho_present=False
        for method in ('original','branch'):
            by_epoch=defaultdict(list)
            for r in data:
                if r['method']==method:by_epoch[int(r['epoch'])].append(r)
            if not by_epoch:continue
            xs=sorted(by_epoch)
            if method=='original':
                median=np.asarray([np.mean([float(r['rho_rep_median']) for r in by_epoch[e]]) for e in xs])
                lo=np.asarray([np.mean([float(r['rho_rep_p10']) for r in by_epoch[e]]) for e in xs])
                hi=np.asarray([np.mean([float(r['rho_rep_p90']) for r in by_epoch[e]]) for e in xs])
                ax_rho.plot(xs,median,label='Original median')
                ax_rho.fill_between(xs,lo,hi,alpha=.18,label='image P10-P90 (avg over seeds)')
                ax_rho.axhline(float(by_epoch[xs[0]][0]['branch_weight']),linestyle='--',color='gray',label='1-alpha')
                rho_present=True
            # Plot only epochs that actually have Rep validation data. Never
            # discard an entire curve because one epoch has a missing value.
            if ax_val is not None:
                valid=[(epoch,[float(r['rep_val_acc']) for r in by_epoch[epoch]
                        if r.get('rep_val_acc') not in (None,'')]) for epoch in xs]
                valid=[(epoch,v) for epoch,v in valid if v]
                if valid:
                    epoch_x=[epoch for epoch,_ in valid]
                    center=[np.mean(values) for _,values in valid]
                    low=[np.min(values) for _,values in valid]
                    high=[np.max(values) for _,values in valid]
                    ax_val.plot(epoch_x,center,label=method+' mean')
                    ax_val.fill_between(epoch_x,low,high,alpha=.15,label=method+' seed range')
        if not rho_present:
            ax_rho.text(.5,.5,'Original responsibility unavailable',ha='center',va='center',transform=ax_rho.transAxes)
        ax_rho.set(xlabel='Epoch',ylabel='Rep responsibility',title='Original-MC responsibility')
        if ax_rho.lines:ax_rho.legend(fontsize=7)
        if ax_val is not None:
            ax_val.set(xlabel='Epoch',ylabel='Rep val ACC (%)',title='Rep validation trajectory')
            if ax_val.lines:ax_val.legend(fontsize=7)
            else:ax_val.text(.5,.5,'Rep validation unavailable',ha='center',va='center',transform=ax_val.transAxes)
        fig.tight_layout()
        target=out/f'S1_{dataset}_shot{shot}.png'
        fig.savefig(target,dpi=220)
        saved.append(str(target))
        plt.close(fig)

    # S2: keep samples, dataset, and seeds separate. Prefer highest S for each seed.
    groups=defaultdict(list)
    for p in root.glob('native/**/diagnostics/**/test/moment_vectors.json'):
        m=json_load(p.parent/'metrics.json')
        if m['method']=='davi' and not m.get('partial'):
            key=(m['dataset'],m['shot'],m['seed'],m['checkpoint']['selection'])
            groups[key].append((m['s_ref'],p,m))
    for (dataset,shot,seed,selection),records in sorted(groups.items()):
        _,p,m=max(records,key=lambda z:z[0]);vectors=json_load(p)
        samples=rows(p.parent/'per_image.csv')
        if not vectors:continue
        fig,axs=plt.subplots(1,3,figsize=(12.7,3.5))
        for ax,key,title in ((axs[0],'mu','Mean'),(axs[1],'var','Variance')):
            xx=[];yy=[]
            for v in vectors:
                for branch in ('main','rep'):
                    xx.extend(v[f'{key}_{branch}_mc']);yy.extend(v[f'{key}_{branch}_det'])
            ax.hexbin(xx,yy,gridsize=40,bins='log',mincnt=1)
            low=min(min(xx),min(yy));high=max(max(xx),max(yy))
            ax.plot([low,high],[low,high],linestyle='--',color='gray',linewidth=1)
            ax.set(xlabel='MC '+title,ylabel='DAVI propagated '+title)
        for branch in ('main','rep'):
            x=np.asarray([float(r[f'{branch}_sampled_var']) for r in samples]);y=np.asarray([float(r[f'{branch}_abs_det_gap']) for r in samples])
            bins=np.unique(np.quantile(x,np.linspace(0,1,11)));xs=[];ys=[]
            for i,(low,high) in enumerate(zip(bins[:-1],bins[1:])):
                mask=(x>=low)&((x<=high) if i==len(bins)-2 else (x<high))
                if mask.any():xs.append(float(np.median(x[mask])));ys.append(float(np.mean(y[mask])))
            if xs:axs[2].plot(xs,ys,marker='o',label=branch)
        axs[2].set(xlabel='MC logit variance quantile',ylabel='Branch |DetCE - MC CE|')
        axs[2].legend(fontsize=7)
        fig.suptitle(f'{dataset}, shot {shot}, seed {seed}, S={m["s_ref"]}',fontsize=9)
        fig.tight_layout();target=out/f'S2_{dataset}_shot{shot}_seed{seed}_{selection}.png'
        fig.savefig(target,dpi=220);saved.append(str(target));plt.close(fig)

    # S3: show actual independent repeat variation rather than just the mean.
    groups=defaultdict(list)
    for p in root.glob('native/**/diagnostics/**/gradient.json'):
        m=json_load(p)
        groups[(m['dataset'],m['shot'],m['seed'],m['checkpoint']['selection'])].append(m)
    for (dataset,shot,seed,selection),candidates in sorted(groups.items()):
        m=max(candidates,key=lambda z:z['reference_samples']);by_s=defaultdict(list)
        for r in m['curve']:by_s[int(r['samples'])].append(r)
        if not by_s:continue
        xs=sorted(by_s)
        fig,axs=plt.subplots(1,2,figsize=(9.5,3.5))
        for ax,metric in zip(axs,('cosine','relative_error')):
            center=[np.mean([r[metric] for r in by_s[s]]) for s in xs]
            low=[np.quantile([r[metric] for r in by_s[s]],.1) for s in xs]
            high=[np.quantile([r[metric] for r in by_s[s]],.9) for s in xs]
            ax.plot(xs,center,marker='o',label='Branch-MC repeat mean')
            ax.fill_between(xs,low,high,alpha=.2,label='MC repeat P10-P90')
            ax.axhline(m['davi'][metric],linestyle='--',color='gray',label='DAVI deterministic')
            ax.set_xscale('log');ax.set(xlabel='MC samples',ylabel=metric);ax.legend(fontsize=7)
        fig.suptitle(f'{dataset}, shot {shot}, seed {seed}, reference S={m["reference_samples"]}',fontsize=9)
        fig.tight_layout();target=out/f'S3_{dataset}_shot{shot}_seed{seed}_{selection}.png'
        fig.savefig(target,dpi=220);saved.append(str(target));plt.close(fig)

    # S4: per-dataset comparisons, same fixed input protocol, paired input digest
    groups=defaultdict(list)
    for p in root.glob('native/**/diagnostics/**/test/per_image.csv'):
        m=json_load(p.parent/'metrics.json')
        if not m.get('partial'):
            key=(m['dataset'],m['shot'],m['seed'],m['method'],m['checkpoint']['selection'])
            groups[key].append((m['s_ref'],p,m))
    by_dataset=defaultdict(list)
    for (dataset,shot,seed,method,selection),candidates in groups.items():
        _,p,m=max(candidates,key=lambda z:z[0])
        by_dataset[(dataset,shot,seed,selection)].append((method,p,m))
    for (dataset,shot,seed,selection),entries in sorted(by_dataset.items()):
        if len(entries)<2:continue
        hashes={m.get('input_sha256') for _,_,m in entries}
        if None in hashes or len(hashes)!=1:
            raise RuntimeError(f'E4 nonpaired input tensors: {dataset} shot {shot} seed {seed}')
        fig,axs=plt.subplots(1,3,figsize=(11.8,3.5))
        for method,p,m in sorted(entries):
            sample=rows(p)
            if not sample:continue
            for ax,key in ((axs[0],'visual_clip_cos'),(axs[2],'rep_margin')):
                xx,yy=ecdf([float(x[key]) for x in sample]);ax.plot(xx,yy,label=method)
            if m.get('text_clip_cos_per_class'):
                xx,yy=ecdf(m['text_clip_cos_per_class']);axs[1].plot(xx,yy,label=method)
        axs[2].axvline(0,linestyle='--',color='gray',linewidth=1)
        for ax,name in zip(axs,('Rep vs frozen CLIP image cosine',
                                'Current vs frozen CLIP text cosine',
                                'Rep log posterior predictive margin')):
            ax.set(xlabel=name,ylabel='ECDF');ax.legend(fontsize=7)
        fig.suptitle(f'{dataset}, shot {shot}, seed {seed}',fontsize=9)
        fig.tight_layout();target=out/f'S4_{dataset}_shot{shot}_seed{seed}_{selection}.png'
        fig.savefig(target,dpi=220);saved.append(str(target));plt.close(fig)
    return saved


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='output_supplementary_v7')
    a=p.parse_args();root=Path(a.output)
    if not root.is_absolute():
        from .core import ROOT
        root=ROOT/root
    for item in make_figures(root):print('[figure]',item)

if __name__=='__main__':main()
