"""R13: extra E2/E3/E5 mechanism figures from real, provenance-checked records.

No extrapolation or placeholder points. Only actual checkpoints and measured
metrics are eligible. Produces immutable figure+JSON source manifests.
"""
from __future__ import annotations
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _valid_checkpoint(record):
    from .core import sha256
    checkpoint=record.get('checkpoint', {})
    path=Path(checkpoint.get('path', ''))
    return (path.is_file() and checkpoint.get('checkpoint_sha256')
            and sha256(path)==checkpoint['checkpoint_sha256'])


def _fingerprint(files):
    return hashlib.sha256('|'.join(str(x) for x in files).encode()).hexdigest()[:12]


def _save(fig, root, stem, manifest, plt, saved):
    root.mkdir(parents=True,exist_ok=True)
    figure=root/(stem+'.png')
    fig.tight_layout()
    fig.savefig(figure,dpi=220)
    plt.close(fig)
    (root/(stem+'.json')).write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    saved.append(str(figure))


def plot_gap_evolution(root,out,plt,np,saved):
    """E2 Fig S6: pure Jensen branch gap vs mean sampled logit variance.

    Fixed train inputs and identical S/R are required across checkpoints.
    Do not substitute L_D-L_B for the pure Jensen gap.
    """
    from .reference_cache import REFERENCE_SCHEMA
    grouped=defaultdict(list)
    for path in sorted(root.glob('native/**/diagnostics/**/train_fixed/metrics.json')):
        try:m=_read(path)
        except (OSError,ValueError):continue
        if (m.get('method')!='davi' or m.get('partial') or
            m.get('reference_schema')!=REFERENCE_SCHEMA or not _valid_checkpoint(m)):
            continue
        needed=('main_sampled_var','rep_sampled_var','main_empirical_jensen_gap',
                'rep_empirical_jensen_gap')
        if any(not isinstance(m.get(key),(int,float)) for key in needed):continue
        epoch=m['checkpoint'].get('epoch')
        if epoch is None:continue
        key=(m['dataset'],int(m['shot']),int(m['seed']),m.get('input_sha256'),
             int(m['s_ref']),int(m.get('mc_repeats',1)),m.get('diagnostic_batch_size'))
        grouped[key].append((epoch,path,m))
    for (dataset,shot,seed,digest,samples,repeats,batch_size),entries in grouped.items():
        # Stale duplicates at the SAME epoch/config cannot silently win.
        epochs=defaultdict(list)
        for ep,path,m in entries:epochs[ep].append((path,m))
        if any(len(v)!=1 for v in epochs.values()):continue
        if len(epochs)<2:
            print(f'[E2 gap evolution] {dataset} seed={seed}: only {len(epochs)} checkpoint; need >=2')
            continue
        points=[(ep,*epochs[ep][0]) for ep in sorted(epochs)]
        fig,axs=plt.subplots(1,2,figsize=(10.5,3.7))
        for ax,branch in zip(axs,('main','rep')):
            xs=[m[f'{branch}_sampled_var'] for ep,path,m in points]
            ys=[m[f'{branch}_empirical_jensen_gap'] for ep,path,m in points]
            ax.plot(xs,ys,marker='o',linewidth=1.7)
            for x,y,(ep,path,m) in zip(xs,ys,points):
                ax.annotate(f'e{ep}',(x,y),textcoords='offset points',xytext=(4,4),fontsize=8)
            title=branch.capitalize()+' branch'
            if len(points)>=3 and np.std(xs)>1e-12 and np.std(ys)>1e-12:
                title+=f' (r={np.corrcoef(xs,ys)[0,1]:.3f})'
            ax.set(xlabel='Mean sampled logit variance',ylabel='Pure empirical Jensen gap (nats/image)',
                   title=title)
            ax.grid(alpha=.2)
        fig.suptitle(f'{dataset}, shot {shot}, seed {seed}; E2 fixed train S={samples}, R={repeats}',fontsize=10)
        ck=_fingerprint([m['checkpoint']['checkpoint_sha256'] for ep,path,m in points])
        stem=f'S6_E2_gap_variance_{dataset}_shot{shot}_seed{seed}_S{samples}_R{repeats}_B{batch_size}_epochs'+ '-'.join(str(ep) for ep,_,_ in points)+f'_input{str(digest)[:10]}_ck{ck}'
        manifest={'figure':'S6','dataset':dataset,'shot':shot,'seed':seed,'split':'train_fixed',
                  'input_sha256':digest,'samples':samples,'mc_repeats':repeats,
                  'metric':'pure_empirical_Jensen_gap_not_DAVI_minus_Branch',
                  'checkpoints':[{'epoch':ep,'source':str(path),
                                   'sha256':m['checkpoint']['checkpoint_sha256']} for ep,path,m in points]}
        _save(fig,out,stem,manifest,plt,saved)


def plot_gradient_noise(root,out,plt,np,saved):
    """E3 Fig S7: per-draw gradient variance & pairwise direction agreement."""
    grouped=defaultdict(list)
    for path in sorted(root.glob('native/**/diagnostics/**/gradient_noise.json')):
        try:m=_read(path)
        except (OSError,ValueError):continue
        if (m.get('method')!='branch' or m.get('partial') or
            not _valid_checkpoint(m)):continue
        inp=m.get('input',{});np_=m.get('noise_parameters',{})
        ck=m.get('checkpoint',{});epoch=ck.get('epoch')
        if epoch is None or not m.get('groups'):continue
        key=(m['dataset'],int(m['shot']),int(m['seed']),inp.get('input_sha256'),
             np_.get('draws'),np_.get('noise_seed'),np_.get('batch_size'))
        grouped[key].append((epoch,path,m))
    group_names=('phi','r_mean','r_rho','t_rho')
    for (dataset,shot,seed,digest,draws,noise_seed,bsize),entries in grouped.items():
        epochs=defaultdict(list)
        for ep,path,m in entries:epochs[ep].append((path,m))
        if any(len(v)!=1 for v in epochs.values()):continue
        points=[(ep,*epochs[ep][0]) for ep in sorted(epochs)]
        fig,axs=plt.subplots(1,2,figsize=(10.8,4.0))
        valid_var=[]
        for ep,path,m in points:
            variance=[m['groups'][key]['relative_variance'] for key in group_names]
            cosine=[m['groups'][key]['pairwise_cosine'] for key in group_names]
            if any(v is not None and v>0 for v in variance):valid_var.append(True)
            axs[0].plot(range(4),[float(v) if v is not None and v>0 else np.nan for v in variance],
                        marker='o',label=f'epoch {ep}')
            axs[1].plot(range(4),[float(v) if v is not None else np.nan for v in cosine],
                        marker='o',label=f'epoch {ep}')
        for ax in axs:
            ax.set_xticks(range(4),['Φ','Mᵣ','ρᵣ','ρₜ'])
            ax.grid(alpha=.2)
            ax.legend(fontsize=7,title='Native Branch-MC')
        if valid_var:axs[0].set_yscale('log')
        axs[0].set(ylabel='Relative per-draw gradient variance',title='Sampling gradient fluctuation')
        axs[1].set(ylabel='Mean pairwise gradient cosine',title='Directional consistency',ylim=(-1.05,1.05))
        fig.suptitle(f'{dataset}, shot {shot}, seed {seed}; J={draws}, fixed batch={bsize}',fontsize=10)
        ck=_fingerprint([m['checkpoint']['checkpoint_sha256'] for ep,path,m in points])
        stem=(f'S7_E3_MC_gradient_noise_{dataset}_shot{shot}_seed{seed}_J{draws}_B{bsize}'
              f'_seed{noise_seed}_epochs'+'-'.join(str(ep) for ep,_,_ in points)+
              f'_input{str(digest)[:10]}_ck{ck}')
        manifest={'figure':'S7','dataset':dataset,'shot':shot,'seed':seed,
            'sampling':'independent posterior draws with fixed train images',
            'draws':draws,'input_sha256':digest,'parameter_groups':list(group_names),
            'checkpoints':[{'epoch':ep,'source':str(path),
                  'sha256':m['checkpoint']['checkpoint_sha256']} for ep,path,m in points]}
        _save(fig,out,stem,manifest,plt,saved)


def plot_matched_time_reference(root,out,plt,np,saved):
    """E5 Fig S8: two methods' *same train_fixed MC classification objective* vs wall time."""
    from .reference_cache import REFERENCE_SCHEMA
    from .core import sha256
    runs=defaultdict(dict)
    for path in sorted(root.glob('matched_time/**/training.json')):
        run=path.parent
        mp=run/'manifest.json'
        if not mp.is_file():continue
        try:t=_read(path);meta=_read(mp)
        except (OSError,ValueError):continue
        if meta.get('status')!='complete' or not t.get('budget_reached'):continue
        method=meta.get('method')
        if method not in ('branch','davi'):continue
        checkpoints=t.get('time_curve_checkpoints') or []
        points=[]
        for snap in checkpoints:
            phase=snap['phase'];cp=run/'time_curve'/'refactor_model'/f'model.pth.tar-{phase}'
            if not cp.is_file() or sha256(cp)!=snap.get('checkpoint_sha256'):
                continue
            prefix=run/'diagnostics'/'time_curve'/f'epoch_{phase}'
            for mpath in sorted(prefix.glob('S*_R*/train_fixed/metrics.json')):
                m=_read(mpath)
                if (m.get('partial') or m.get('reference_schema')!=REFERENCE_SCHEMA or
                        m.get('split')!='train_fixed' or not _valid_checkpoint(m) or
                        m['checkpoint']['checkpoint_sha256']!=snap['checkpoint_sha256']):continue
                points.append((snap,m,mpath))
        if not points:continue
        # Require one unambiguous MC budget/split/protocol for every phase.
        by_reference=defaultdict(list)
        for snap,m,mpath in points:
            by_reference[(m['s_ref'],m.get('mc_repeats',1),m['input_sha256'])].append((snap,m,mpath))
        for setting,values in by_reference.items():
            phases=[p['phase'] for p,m,path in values]
            if len(set(phases))!=len(phases):continue
            key=(meta['dataset'],int(meta['shot']),int(meta['seed']),float(t['budget_seconds']),setting)
            runs[key][method]=(values,run,t,meta)
    for (dataset,shot,seed,budget,setting),methods in runs.items():
        if set(methods)!={'branch','davi'}:continue
        expected=(0,25,50,75,100)
        if any(set(snap['phase'] for snap,m,path in methods[method][0])!=set(expected)
               for method in methods):
            continue
        fig,ax=plt.subplots(figsize=(7.0,4.1))
        sources={}
        for method in ('branch','davi'):
            values,run,t,meta=methods[method]
            values=sorted(values,key=lambda row:row[0]['phase'])
            xs=[row[0]['train_wall_seconds'] for row in values]
            ys=[row[1]['branch_nll'] for row in values]
            ax.plot(xs,ys,marker='o',label=('AVI-Branch' if method=='branch' else 'DAVI'))
            for x,y,row in zip(xs,ys,values):
                ax.annotate(str(row[0]['phase'])+'%',(x,y),xytext=(3,4),
                            textcoords='offset points',fontsize=7)
            sources[method]=[{'phase':snap['phase'],'epoch':snap['real_epoch'],
                 'wall_seconds':snap['train_wall_seconds'],'updates':snap['updates'],
                 'metrics_path':str(path),'checkpoint_sha256':m['checkpoint']['checkpoint_sha256']}
                 for snap,m,path in values]
        ax.set(xlabel='Cumulative training wall-clock (s), excludes diagnostic I/O',
               ylabel='Common Branch-MC reference CE (nats/image)',
               title=f'{dataset}, shot {shot}, seed {seed}; matched T={budget:.1f}s')
        ax.grid(alpha=.25);ax.legend()
        samples,repeats,digest=setting
        identity=_fingerprint([x['checkpoint_sha256'] for method in ('branch','davi') for x in sources[method]])
        stem=(f'S8_E5_matched_time_{dataset}_shot{shot}_seed{seed}_T{budget:g}'
              f'_S{samples}_R{repeats}_input{str(digest)[:10]}_ck{identity}')
        manifest={'figure':'S8','dataset':dataset,'shot':shot,'seed':seed,
                'budget_seconds':budget,'diagnostic_split':'train_fixed',
                'diagnostic_target':'common Branch-MC classification CE (without KL/alignment)',
                'diagnostic_sampling':{'samples':samples,'repeats':repeats},
                'input_sha256':digest,'sources':sources}
        _save(fig,out,stem,manifest,plt,saved)


def plot_e4_evolution(root,out,plt,np,saved):
    """S9: strict cross-method time course; no fabricated or stale checkpoints."""
    from .core import sha256
    from .reference_cache import initial_snapshot_hash
    from collections import defaultdict
    grouped=defaultdict(lambda:defaultdict(dict))
    for path in sorted(root.glob('native/**/diagnostics/**/e4_evolution/**/metrics.json')):
        try:m=_read(path)
        except (OSError,ValueError):continue
        ck=m.get('checkpoint',{});inp=m.get('input',{});settings=m.get('settings',{})
        parts=path.parts
        native_root=Path(*parts[:parts.index('diagnostics')])
        if m.get('initial_checkpoint_sha256')!=initial_snapshot_hash(native_root):continue
        if (m.get('schema')!=1 or m.get('partial') or not _valid_checkpoint(m)
            or m.get('rep_margin_type')!='mc_mean_logit'
            or m.get('method') not in ('original','branch','davi') or ck.get('epoch') is None):
            continue
        key=(m['dataset'],int(m['shot']),int(m['seed']),inp.get('input_sha256'),
             settings.get('images'),settings.get('draws'),settings.get('chunk'),
             settings.get('batch'),settings.get('input_seed'))
        epoch=int(ck['epoch']);method=m['method']
        if method in grouped[key][epoch]:
            # Do not silently choose among duplicate versions of same method/epoch.
            grouped[key][epoch][method]=None
        else:grouped[key][epoch][method]=(path,m)
    for key, epochs in grouped.items():
        dataset,shot,seed,input_sha,images,draws,chunk,batch,input_seed=key
        stages=[]
        for epoch,methods in sorted(epochs.items()):
            if set(methods)=={'original','branch','davi'} and all(methods.values()):
                stages.append((epoch,methods))
        if len(stages)<2:
            print(f'[S9] {dataset} seed={seed}: only {len(stages)} paired stages; need >=2')
            continue
        methods=('original','branch','davi')
        fig,axs=plt.subplots(1,3,figsize=(12.3,3.6))
        for method in methods:
            x=[e for e,_ in stages]
            for ax,metric in zip(axs,('r_trace','t_trace','rep_margin_median')):
                ax.plot(x,[d[method][1][metric] for _,d in stages],'-o',label=method,markersize=3.5)
        for ax,title,ylabel in zip(axs,
          ('Visual posterior trace','Text posterior trace','Rep discriminative margin'),
          (r'$\mathrm{tr}\,\Sigma_r$',r'$\mathrm{tr}\,\Sigma_t$','Median MC-mean logit margin')):
            ax.set(xlabel='Completed training epoch',ylabel=ylabel,title=title)
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
        axs[2].axhline(0,ls='--',c='gray',lw=1)
        fig.suptitle(f'{dataset} | {shot}-shot | seed {seed} | fixed class-spread train images (N={images})',fontsize=10)
        sources={method:[{'epoch':ep,'metrics':str(d[method][0]),
                  'checkpoint_sha256':d[method][1]['checkpoint']['checkpoint_sha256']}
                  for ep,d in stages] for method in methods}
        ident=_fingerprint([rec['checkpoint_sha256'] for method in methods for rec in sources[method]])
        stem=(f'S9_E4_posterior_rep_evolution_{dataset}_shot{shot}_seed{seed}'
              f'_N{images}_S{draws}_C{chunk}_B{batch}_seed{input_seed}'
              f'_epochs'+ '-'.join(str(e) for e,_ in stages)+f'_input{str(input_sha)[:10]}_ck{ident}')
        manifest=dict(figure='S9',dataset=dataset,shot=shot,seed=seed,
          input_sha256=input_sha,rep_margin_type='mc_mean_logit',
          diagnostic_split='train_fixed_subset_deterministic',
          diagnostic_settings=dict(images=images,draws=draws,chunk=chunk,batch=batch,input_seed=input_seed),
          stages=[ep for ep,_ in stages],sources=sources)
        _save(fig,out,stem,manifest,plt,saved)


def make_additional_figures(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    root=Path(root);out=root/'figures';out.mkdir(parents=True,exist_ok=True)
    saved=[]
    plot_gap_evolution(root,out,plt,np,saved)
    plot_e4_evolution(root,out,plt,np,saved)
    plot_gradient_noise(root,out,plt,np,saved)
    plot_matched_time_reference(root,out,plt,np,saved)
    return saved
