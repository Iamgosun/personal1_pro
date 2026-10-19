"""Source-backed mechanism figures S1--S4; no invented curves or chart placeholders.

Each figure is scoped by dataset and shot, avoiding mixing unrelated class sets.
Only complete, saved native runs are considered. Charts need matplotlib/numpy.
"""
from __future__ import annotations
import argparse,csv,json,hashlib
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
            key=(m['dataset'],m['shot'],m['seed'],m['checkpoint']['selection'],
                 m['checkpoint'].get('epoch') if m['checkpoint'].get('epoch') is not None else -1)
            groups[key].append((m['s_ref'],p,m))
    for (dataset,shot,seed,selection,epoch_key),records in sorted(groups.items()):
        _,p,m=max(records,key=lambda z:z[0]);vectors=json_load(p)
        samples=rows(p.parent/'per_image.csv')
        if not vectors:continue
        fig,axs=plt.subplots(1,3,figsize=(13.5,3.7))
        # Do NOT merge Main and Rep into a single hexbin: their uncertainty
        # mechanisms differ (text-only versus visual+text). Subsample the plot
        # deterministically without changing any underlying metric.
        for ax,key,title in ((axs[0],'mu','Mean'),(axs[1],'var','Variance')):
            extrema=[]
            for branch in ('main','rep'):
                xx=np.asarray([v[f'{key}_{branch}_mc'] for v in vectors],dtype=float).ravel()
                yy=np.asarray([v[f'{key}_{branch}_det'] for v in vectors],dtype=float).ravel()
                finite=np.isfinite(xx)&np.isfinite(yy)
                xx,yy=xx[finite],yy[finite]
                if len(xx)==0:continue
                extrema.extend((float(xx.min()),float(xx.max()),float(yy.min()),float(yy.max())))
                sample=np.linspace(0,len(xx)-1,min(len(xx),7000),dtype=int)
                ax.scatter(xx[sample],yy[sample],s=3,alpha=.20,
                           label=f'{branch.capitalize()} (MAE={np.abs(xx-yy).mean():.3g})',
                           rasterized=True)
            if extrema:
                low=min(extrema);high=max(extrema)
                ax.plot([low,high],[low,high],linestyle='--',color='gray',linewidth=1)
            ax.set(xlabel='MC '+title+' (repeat 1)',ylabel='DAVI propagated '+title)
            ax.legend(fontsize=7,markerscale=2)
        for branch in ('main','rep'):
            x=np.asarray([float(r[f'{branch}_sampled_var']) for r in samples]);y=np.asarray([float(r[f'{branch}_abs_det_gap']) for r in samples])
            bins=np.unique(np.quantile(x,np.linspace(0,1,11)));xs=[];ys=[]
            for i,(low,high) in enumerate(zip(bins[:-1],bins[1:])):
                mask=(x>=low)&((x<=high) if i==len(bins)-2 else (x<high))
                if mask.any():xs.append(float(np.median(x[mask])));ys.append(float(np.mean(y[mask])))
            if xs:axs[2].plot(xs,ys,marker='o',label=branch)
        axs[2].set(xlabel='MC logit variance quantile',ylabel='Branch |DetCE - MC CE|')
        axs[2].legend(fontsize=7)
        warning=' [MC repeat warning]' if m.get('mc_precision_warning') else ''
        epoch=m['checkpoint'].get('epoch')
        fig.suptitle(f'{dataset}, shot {shot}, seed {seed}, S={m["s_ref"]}, R={m.get("mc_repeats",1)}'+warning,fontsize=9)
        ep_tag=f'_epoch{epoch}' if epoch is not None else ''
        fig.tight_layout();target=out/f'S2_{dataset}_shot{shot}_seed{seed}_{selection}{ep_tag}.png'
        fig.savefig(target,dpi=220);saved.append(str(target));plt.close(fig)

    e3_grouped=defaultdict(list)
    # Group checkpoint comparisons only for the same native training run,
    # fixed input and exact E3 diagnostic configuration.
    # S3: EVERY E3 configuration is a separate figure. Never pick "largest S"
    # from unrelated batches/checkpoints or silently overwrite pilot results.
    # A pre-R7 diagnostic without an input hash is labeled "legacy" explicitly.
    for p in sorted(root.glob('native/**/diagnostics/**/gradient.json')):
        m=json_load(p)
        by_s=defaultdict(list)
        for r in m['curve']:
            by_s[int(r['samples'])].append(r)
        if not by_s:continue
        xs=sorted(by_s)
        dataset=m['dataset'];shot=m['shot'];seed=m['seed']
        selection=m['checkpoint']['selection']
        epoch=m['checkpoint'].get('epoch')
        ep_tag='best' if epoch is None else f'epoch{epoch}'
        e3=m.get('e3_parameters',{})
        inp=m.get('input',{})
        batch_size=e3.get('batch_size',inp.get('batch_size_requested','unknown'))
        strategy=e3.get('batch_strategy',inp.get('selection_strategy','legacy'))
        input_seed=e3.get('input_seed',inp.get('selection_seed','unknown'))
        samples=m['reference_samples']
        chunk=e3.get('chunk','unknown')
        repeats=e3.get('repeats',len(by_s[xs[0]]))
        budgets='-'.join(str(z) for z in xs)
        digest=str(inp.get('input_sha256') or m['checkpoint'].get('checkpoint_sha256') or 'legacy')[:10]
        partial='_smoke' if m.get('partial') else ''
        # Every declared diagnostic dimension contributes to file identity.
        target=out/(f'S3_{dataset}_shot{shot}_seed{seed}_{selection}_{ep_tag}'
                    f'_refS{samples}_chunk{chunk}_R{repeats}_B{batch_size}'
                    f'_{strategy}_input{input_seed}_sizes{budgets}_hash{digest}{partial}.png')
        fig,axs=plt.subplots(1,2,figsize=(9.5,3.7))
        for ax,metric in zip(axs,('cosine','relative_error')):
            center=[np.mean([r[metric] for r in by_s[n]]) for n in xs]
            low=[np.quantile([r[metric] for r in by_s[n]],.1) for n in xs]
            high=[np.quantile([r[metric] for r in by_s[n]],.9) for n in xs]
            ax.plot(xs,center,marker='o',label='Branch-MC mean over repeats')
            if any(len(by_s[n])>1 for n in xs):
                ax.fill_between(xs,low,high,alpha=.2,label='MC repeat P10-P90')
            ax.axhline(m['davi'][metric],linestyle='--',color='gray',label='DAVI deterministic')
            ax.set_xscale('log');ax.set(xlabel='MC samples',ylabel=metric)
            ax.legend(fontsize=7)
        unstable=not bool(m.get('reference_stable',False))
        legacy=not bool(inp.get('input_sha256'))
        warning=' [MC reference unstable]' if unstable else ''
        if legacy:warning+=' [legacy input not recorded]'
        if m.get('partial'):warning+=' [pilot]'
        title=(f'{dataset}, shot {shot}, seed {seed}, {selection} {ep_tag}; '
               f'Sref={samples}, B={batch_size}, {strategy}; '
               f'classes={inp.get("n_unique_classes","?")}'+warning)
        fig.suptitle(title,fontsize=8)
        fig.tight_layout(rect=(0,0,1,.91))
        fig.savefig(target,dpi=220);saved.append(str(target));plt.close(fig)
        # No grouping across different seeds, batches or input transforms.
        run_prefix=tuple(p.parts[:p.parts.index('diagnostics')])
        group_key=(run_prefix,dataset,shot,seed,
                   str(inp.get('input_sha256') or ''),
                   samples,chunk,repeats,batch_size,strategy,input_seed,
                   tuple(xs), bool(m.get('partial')))
        if inp.get('input_sha256') and epoch is not None and not m.get('partial'):
            e3_grouped[group_key].append(m)

    # E3 checkpoint comparison: a second figure shows the same measurement at
    # early/middle/final epochs, each relative to THAT checkpoint's own S_ref.
    for key, records in sorted(e3_grouped.items(),key=lambda item:str(item[0])):
        epoch_map={}
        for m in records:
            e=int(m['checkpoint']['epoch'])
            if e in epoch_map:
                print(f'[S3 WARNING] Ambiguous duplicate E3 checkpoint epoch={e}; skipped comparison')
                epoch_map={};break
            epoch_map[e]=m
        if len(epoch_map)<2:continue
        epochs=sorted(epoch_map)
        base=epoch_map[epochs[0]]
        dataset=base['dataset'];shot=base['shot'];seed=base['seed']
        budgets=sorted(set(int(r['samples']) for m in epoch_map.values() for r in m['curve']))
        if any({int(r['samples']) for r in m['curve']}!=set(budgets) for m in epoch_map.values()):
            continue
        fig,axs=plt.subplots(1,2,figsize=(10.5,3.9))
        for ax,metric in zip(axs,('cosine','relative_error')):
            ax.plot(epochs,[epoch_map[e]['davi'][metric] for e in epochs],
                    marker='D',linestyle='--',linewidth=2,label='DAVI deterministic')
            for budget in budgets:
                grouped=[[r[metric] for r in epoch_map[e]['curve']
                          if int(r['samples'])==budget] for e in epochs]
                ax.plot(epochs,[float(np.mean(values)) for values in grouped],
                        marker='o',label=f'Branch-MC S={budget}')
                if any(len(values)>1 for values in grouped):
                    low=[np.quantile(values,.1) for values in grouped]
                    high=[np.quantile(values,.9) for values in grouped]
                    ax.fill_between(epochs,low,high,alpha=.12)
            ax.set(xlabel='DAVI checkpoint epoch',ylabel=metric,xticks=epochs)
            ax.legend(fontsize=7)
        unstable=[e for e in epochs if not epoch_map[e].get('reference_stable',False)]
        warn=f' [unstable reference at {unstable}]' if unstable else ''
        fig.suptitle(f'{dataset}, shot {shot}, seed {seed}; E3 across checkpoints'
                     +warn,fontsize=9)
        fig.tight_layout(rect=(0,0,1,.92))
        inp=base['input'];params=base['e3_parameters']
        param_tag=(f'refS{base["reference_samples"]}_chunk{params["chunk"]}_'
                   f'R{params["repeats"]}_B{inp["n_images"]}_'
                   f'{inp["selection_strategy"]}_input{inp["selection_seed"]}_'
                   f'sizes'+'-'.join(map(str,budgets)))
        checkpoint_hashes='-'.join(str(epoch_map[e]['checkpoint'].get('checkpoint_sha256','')) for e in epochs)
        checkpoint_tag=hashlib.sha256(checkpoint_hashes.encode()).hexdigest()[:10]
        tag=(f'S3_epoch_comparison_{dataset}_shot{shot}_seed{seed}_'
             f'{param_tag}_epochs'+'-'.join(map(str,epochs))+'_inputhash'+
             inp['input_sha256'][:10]+'_ck'+checkpoint_tag+'.png')
        target=out/tag;fig.savefig(target,dpi=220);saved.append(str(target));plt.close(fig)

    # S4: exact three-method comparisons. Never choose a "largest S" result
    # from a different checkpoint, MC budget, repeat count or fixed input.
    # The new E4 metric is the margin of MC-mean Rep *logits*, NOT BMA log p.
    from .reference_cache import REFERENCE_SCHEMA
    from .core import sha256
    s4_groups=defaultdict(list)
    for p in sorted(root.glob('native/**/diagnostics/**/test/per_image.csv')):
        mpath=p.parent/'metrics.json'
        if not mpath.is_file():continue
        m=json_load(mpath)
        if m.get('partial') or m.get('reference_schema')!=REFERENCE_SCHEMA:
            continue
        if m.get('rep_margin_type')!='mc_mean_logit' or m.get('split')!='test':
            continue
        ck=m.get('checkpoint',{})
        ck_file=Path(ck.get('path',''))
        # The metrics must still describe the current model, even if a same-seed
        # retrain replaced the checkpoint without removing the diagnostic files.
        if not ck_file.is_file() or ck.get('checkpoint_sha256')!=sha256(ck_file):
            print(f'[S4 WARNING] Stale or missing checkpoint, skip: {mpath}',flush=True)
            continue
        digest=m.get('input_sha256')
        if not digest:continue
        key=(m['dataset'],m['shot'],m['seed'],ck['selection'],ck.get('epoch'),
             m['s_ref'],m.get('mc_repeats',1),digest)
        s4_groups[key].append((m['method'],p,m))
    for (dataset,shot,seed,selection,epoch,sref,repeats,digest),entries in sorted(
            s4_groups.items(), key=lambda kv: tuple(str(v) for v in kv[0])):
        methods=[method for method,_,_ in entries]
        # E4 explicitly compares Original, Branch and DAVI at the SAME
        # checkpoint epoch and with identical data/sampling. Missing methods
        # are fine for pilots, but do not present a partial plot as all-three.
        if set(methods)!={'original','branch','davi'} or len(methods)!=3:
            if len(entries)>1:
                print(f'[S4 WARNING] Incomplete/ambiguous pairing {dataset} '
                      f'seed={seed} epoch={epoch}: {methods}',flush=True)
            continue
        # Check that the per-image CSV really carries the new metric.
        plotted={}
        for method,p,m in entries:
            sample=rows(p)
            if not sample or any('rep_margin' not in x or 'visual_clip_cos' not in x
                                 for x in sample):
                print(f'[S4 WARNING] Missing per-image E4 columns: {p}',flush=True)
                break
            plotted[method]=(sample,m)
        if len(plotted)!=3:continue
        fig,axs=plt.subplots(1,3,figsize=(11.8,3.5))
        for method in ('original','branch','davi'):
            sample,m=plotted[method]
            for ax,k in ((axs[0],'visual_clip_cos'),(axs[2],'rep_margin')):
                xx,yy=ecdf([float(x[k]) for x in sample]);ax.plot(xx,yy,label=method)
            if m.get('text_clip_cos_per_class'):
                xx,yy=ecdf(m['text_clip_cos_per_class']);axs[1].plot(xx,yy,label=method)
        axs[2].axvline(0,linestyle='--',color='gray',linewidth=1)
        for ax,name in zip(axs,('Rep vs frozen CLIP image cosine',
                                'Current vs frozen CLIP text cosine',
                                'Rep MC-mean logit margin')):
            ax.set(xlabel=name,ylabel='ECDF');ax.legend(fontsize=7)
        fig.suptitle(f'{dataset}, shot {shot}, seed {seed}; epoch {epoch}, '
                     f'S={sref}, R={repeats}',fontsize=9)
        # Exact source identity, including three checkpoint hashes, prevents
        # overwriting another valid experiment with the same display settings.
        hashes='|'.join(f'{method}:{plotted[method][1]["checkpoint"]["checkpoint_sha256"]}'
                        for method in ('original','branch','davi'))
        ck_tag=hashlib.sha256(hashes.encode()).hexdigest()[:10]
        ep_tag='unknown' if epoch is None else str(epoch)
        tag=(f'S4_{dataset}_shot{shot}_seed{seed}_{selection}_epoch{ep_tag}'
             f'_S{sref}_R{repeats}_input{digest[:10]}_ck{ck_tag}')
        fig.tight_layout();target=out/(tag+'.png')
        fig.savefig(target,dpi=220);saved.append(str(target));plt.close(fig)
        # Auditable exact source list (not a reconstructed figure label).
        meta={'dataset':dataset,'shot':shot,'seed':seed,'selection':selection,
              'epoch':epoch,'samples':sref,'mc_repeats':repeats,
              'input_sha256':digest,'rep_margin_type':'mc_mean_logit',
              'source_by_method':{method:{'metrics':str(next(p.parent/'metrics.json' for name,p,m in entries if name==method)),
                                               'checkpoint_sha256':plotted[method][1]['checkpoint']['checkpoint_sha256']}
                                  for method in ('original','branch','davi')}}
        (out/(tag+'.json')).write_text(json.dumps(meta,indent=2),encoding='utf-8')
    from .additional_figures import make_additional_figures
    saved.extend(make_additional_figures(root))
    return saved


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='output_supplementary_v7')
    a=p.parse_args();root=Path(a.output)
    if not root.is_absolute():
        from .core import ROOT
        root=ROOT/root
    for item in make_figures(root):print('[figure]',item)

if __name__=='__main__':main()
