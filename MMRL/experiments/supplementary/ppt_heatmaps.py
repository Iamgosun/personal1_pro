"""Plots for DAVI-Adapter supplementary study (R15).

No training, GPU, network, or checkpoints required for summary / gradient maps.
For dense feature maps, run `run_ppt_pairs.sh` with SAVE_MATRICES=1 first.

From the MMRL root:
    python -m experiments.supplementary.ppt_heatmaps --output output_supplementary_v7
    python -m experiments.supplementary.ppt_heatmaps --output output_supplementary_v7 --only features

The script never rewrites results/figures produced by run_study.sh.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

DATASET_LABELS = {'caltech101': 'Caltech101', 'eurosat': 'EuroSAT',
                  'oxford_pets': 'OxfordPets', 'fgvc_aircraft': 'FGVCAircraft',
                  'dtd': 'DTD'}
GROUP_LABELS = {'phi': r'$\Phi$', 'r_mean': r'$M_r$',
                'r_rho': r'$\rho_r$', 't_rho': r'$\rho_t$'}
GROUPS = ('phi', 'r_mean', 'r_rho', 't_rho')


def _read_csv(path):
    if not path.is_file():
        return []
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def _num(row, key):
    value = row.get(key, '')
    return float(value) if value not in ('', None) else math.nan


def _one(rows, key):
    if len(rows) != 1:
        raise RuntimeError(f'Expected exactly 1 record for {key}; found {len(rows)}')
    return rows[0]


def _label(dataset):
    return DATASET_LABELS.get(dataset, dataset)


def _save_matrix(out, values, xlabels, ylabels, *, title, bar_label, basename,
                 cmap='viridis', vmin=None, vmax=None, fmt='.3f', note='', text=True):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).any():
        return None
    fig_w = max(6.6, 1.25*len(xlabels) + 2.7)
    fig_h = max(3.6, 0.51*len(ylabels) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    data = np.ma.masked_invalid(values)
    im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation='nearest')
    ax.set_xticks(range(len(xlabels)), labels=xlabels)
    ax.set_yticks(range(len(ylabels)), labels=ylabels)
    ax.tick_params(axis='x', labelrotation=0)
    ax.set_title(title, fontsize=12, pad=13)
    cbar = fig.colorbar(im, ax=ax, fraction=.035, pad=.035)
    cbar.set_label(bar_label)
    if text and values.size <= 120:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if not np.isfinite(values[i,j]):
                    continue
                # White/black annotation based on normalized colormap luminance.
                rgba = im.cmap(im.norm(values[i,j]))
                lum = .2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                ax.text(j,i,format(values[i,j],fmt),ha='center',va='center',
                        fontsize=9, color='black' if lum>.55 else 'white')
    if note:
        fig.text(.02,.015,note,fontsize=8,color='#555555')
    fig.tight_layout(rect=(0,.045 if note else 0,1,1))
    out.mkdir(parents=True,exist_ok=True)
    png = out/(basename+'.png')
    pdf = out/(basename+'.pdf')
    fig.savefig(png,dpi=240,bbox_inches='tight')
    fig.savefig(pdf,bbox_inches='tight')
    plt.close(fig)
    print('[FIG]',png)
    return png


def _paired_reference(root, out, datasets, shot):
    """DAVI vs AVI-Branch on SAME seed, test input, checkpoint, MC settings."""
    rows = [r for r in _read_csv(root/'summary/table_s3_runs.csv')
            if int(r['shot'])==shot and r['method'] in ('davi','branch')
            and r.get('reference_split')=='test']
    if not rows:
        print('[SKIP] no table_s3_runs.csv test reference records')
        return 0
    datasets = [d for d in datasets if any(r['dataset']==d for r in rows)]
    seeds = sorted({int(r['seed']) for r in rows})
    matrices = {key: np.full((len(datasets),len(seeds)),np.nan) for key in
                ('visual_ratio','text_ratio','branch_delta','original_delta')}
    for i, d in enumerate(datasets):
        for j,s in enumerate(seeds):
            pair=[r for r in rows if r['dataset']==d and int(r['seed'])==s]
            if not pair:
                continue
            b=_one([r for r in pair if r['method']=='branch'],(d,s,'branch'))
            v=_one([r for r in pair if r['method']=='davi'],(d,s,'davi'))
            for key in ('checkpoint','reference_input_sha256','s_ref','mc_repeats','reference_split'):
                if b[key]!=v[key]:
                    raise RuntimeError(f'Unpaired reference protocol {d}/seed{s}: {key}')
            if int(b['s_ref'])<2:
                raise RuntimeError('MC reference must have >=2 draws')
            matrices['visual_ratio'][i,j]=_num(v,'r_trace')/_num(b,'r_trace')
            matrices['text_ratio'][i,j]=_num(v,'t_trace')/_num(b,'t_trace')
            matrices['branch_delta'][i,j]=_num(v,'branch_nll')-_num(b,'branch_nll')
            matrices['original_delta'][i,j]=_num(v,'original_nll')-_num(b,'original_nll')
    note = '3 seeds; ratios DAVI/Branch; NLL differences DAVI minus Branch; same test inputs and MC protocol.'
    spec = [
       ('visual_ratio','Posterior dispersion — visual projection', 'trace ratio', 'H1_visual_trace_ratio','viridis',None,None),
       ('text_ratio','Posterior dispersion — text projection','trace ratio','H2_text_trace_ratio','viridis',None,None),
       ('branch_delta','Common Branch-MC classification NLL','DAVI - Branch (nats/image)','H3_branch_mc_nll_delta','coolwarm',None,None),
       ('original_delta','Common Original-MC classification NLL','DAVI - Branch (nats/image)','H4_original_mc_nll_delta','coolwarm',None,None),
    ]
    count=0
    for k,title,unit,name,cmap,low,high in spec:
        m=matrices[k]
        if k.endswith('_delta'):
            mx=np.nanmax(np.abs(m))
            low,high=-mx,mx
        count+=_save_matrix(out,m,[f'Seed {s}' for s in seeds],[_label(d) for d in datasets],
                             title=title,bar_label=unit,basename=name,cmap=cmap,
                             vmin=low,vmax=high,fmt='.3f',note=note) is not None
    return count


def _e3_quality(root,out,datasets,shot):
    rows=[r for r in _read_csv(root/'summary/e3_checkpoint_comparison.csv')
          if int(r['shot'])==shot and r['dataset'] in datasets and int(r['mc_samples'])==1]
    if not rows:
        return 0
    epochs=sorted({int(r['epoch']) for r in rows})
    matrix1=np.full((len(datasets),len(epochs)),np.nan)
    matrix2=np.full_like(matrix1,np.nan)
    for i,d in enumerate(datasets):
        for j,e in enumerate(epochs):
            group=[r for r in rows if r['dataset']==d and int(r['epoch'])==e]
            if not group:continue
            if len(set(int(r['seed']) for r in group))!=len(group):
                raise RuntimeError(f'Duplicate gradient rows {d} epoch {e}')
            matrix1[i,j]=np.mean([_num(r,'davi_relative_error') for r in group])
            matrix2[i,j]=np.mean([_num(r,'mc_relative_error_mean') for r in group])
    note='Relative L2 error against high-S MC reference on fixed train batch; mean across available seeds.'
    count=0
    for m,title,name in [(matrix1,'DAVI deterministic gradient bias','H5_davi_relative_gradient_error'),
                         (matrix2,'AVI-Branch MC (S=1) gradient sampling error','H6_mc1_relative_gradient_error')]:
        count+=_save_matrix(out,m,[f'E{e}' for e in epochs],[_label(d) for d in datasets],
                             title=title,bar_label='relative L2 error',basename=name,
                             note=note) is not None
    return count


def _gradient_noise(root,out,datasets,shot):
    """True group diagnostics from per-checkpoint gradient_noise.json; not fabricated pairwise matrices."""
    files=sorted(p for p in (root/'native').glob('**/gradient_noise/**/gradient_noise.json')
                 if '.ipynb_checkpoints' not in p.parts)
    rows=[]
    for path in files:
        d=json.loads(path.read_text(encoding='utf-8'))
        if d.get('partial') or d.get('method')!='branch' or d.get('dataset') not in datasets \
           or int(d.get('shot',-1))!=shot:continue
        cp=d.get('checkpoint',{});
        if cp.get('epoch') is None:
            continue
        controls=d.get('noise_parameters',{})
        if controls.get('draws',d.get('draws')) is None:
            continue
        rows.append(dict(dataset=d['dataset'],seed=int(d['seed']),epoch=int(cp['epoch']),
              groups=d['groups'],draws=int(d['draws']),
              input_seed=controls.get('input_seed'),noise_seed=controls.get('noise_seed'),
              n_images=d.get('input',{}).get('n_images'),
              input_sha256=d.get('input',{}).get('input_sha256')))
    if not rows:
        print('[SKIP] no gradient_noise.json')
        return 0
    count=0
    for dataset in datasets:
        group=[r for r in rows if r['dataset']==dataset]
        if not group:continue
        # Do not average incompatible diagnostic protocols.
        controls={(r['draws'],r['input_seed'],r['noise_seed'],r['n_images']) for r in group}
        if len(controls)!=1:
            raise RuntimeError(f'Mixed E3 gradient-noise controls for {dataset}: {controls}')
        seed_hash={}
        for r in group:
            key=(r['seed'],r['epoch'])
            if key in seed_hash and seed_hash[key]!=r['input_sha256']:
                raise RuntimeError(f'Incompatible E3 fixed input {dataset} {key}')
            seed_hash[key]=r['input_sha256']
        for s in {r['seed'] for r in group}:
            if len({r['input_sha256'] for r in group if r['seed']==s})!=1:
                raise RuntimeError(f'E3 changed fixed input across epochs {dataset} seed={s}')
        # Files may contain multiple protocol copies with same seed/epoch; reject rather than silently double-count.
        if len({(r['seed'],r['epoch']) for r in group})!=len(group):
            raise RuntimeError(f'Duplicate E3 gradient-noise diagnostics: {dataset}')
        epochs=sorted({r['epoch'] for r in group})
        for stat in ('relative_variance','pairwise_cosine'):
            raw=np.full((len(GROUPS),len(epochs)),np.nan)
            for i,p in enumerate(GROUPS):
                for j,e in enumerate(epochs):
                    numbers=[float(r['groups'][p][stat]) for r in group
                             if r['epoch']==e and r['groups'].get(p,{}).get(stat) is not None]
                    if numbers:raw[i,j]=np.mean(numbers)
            if stat=='relative_variance':
                display=np.log10(np.maximum(raw,1e-12))
                bar='log10(relative per-draw gradient variance)'
                name=f'H7_gradient_variance_{dataset}'
                title=f'{_label(dataset)} — MC gradient relative variance'
                cmap='viridis'
            else:
                display=raw
                bar='pairwise cosine similarity'
                name=f'H8_gradient_direction_{dataset}'
                title=f'{_label(dataset)} — MC gradient directional agreement'
                cmap='coolwarm'
            # Numerical labels display ORIGINAL values even when coloring on log10 scale.
            # Use explicit annotation after creating figure to avoid conflating scale and units.
            note=f'Branch only; J={group[0]["draws"]} posterior draws; mean of matched-seed diagnostics. '
            note+=('Cell color = log10(V); numbers shown in log10 scale.' if stat=='relative_variance' else 'Cell values = cosine.')
            count+=_save_matrix(out,display,[f'E{e}' for e in epochs],[GROUP_LABELS[k] for k in GROUPS],
                                title=title,bar_label=bar,basename=name,cmap=cmap,
                                vmin=(-1 if stat=='pairwise_cosine' else None),
                                vmax=(1 if stat=='pairwise_cosine' else None),
                                note=note,fmt='.2f') is not None
    return count


def _e4_posterior_history(root,out,datasets,shot):
    rows=[r for r in _read_csv(root/'summary/e4_evolution_runs.csv')
          if int(r['shot'])==shot and r['dataset'] in datasets and r['method'] in ('branch','davi')]
    if not rows:return 0
    for r in rows:
        r['seed']=int(r['seed']);r['epoch']=int(r['epoch'])
    epochs=sorted({r['epoch'] for r in rows})
    mats={v:np.full((len(datasets),len(epochs)),np.nan) for v in ('r_trace','t_trace')}
    for i,d in enumerate(datasets):
        for j,e in enumerate(epochs):
            rs=[r for r in rows if r['dataset']==d and r['epoch']==e]
            ratios=defaultdict(list)
            for s in sorted({r['seed'] for r in rs}):
                p=[r for r in rs if r['seed']==s]
                if len(p)!=2 or set(r['method'] for r in p)!={'branch','davi'}:
                    raise RuntimeError(f'Incomplete E4 method pair {d} seed{s} epoch{e}')
                b=_one([r for r in p if r['method']=='branch'],(d,s,e,'branch'))
                v=_one([r for r in p if r['method']=='davi'],(d,s,e,'davi'))
                for ctrl in ('input_sha256','images','draws','chunk','batch','input_seed'):
                    if b[ctrl]!=v[ctrl]:
                        raise RuntimeError(f'E4 mismatch {d} seed{s} epoch{e}: {ctrl}')
                for stat in mats:
                    ratios[stat].append(_num(v,stat)/_num(b,stat))
            for stat in mats:
                if ratios[stat]:mats[stat][i,j]=np.mean(ratios[stat])
    count=0
    for stat,title,name in [('r_trace','Visual posterior trace ratio over training','H9_visual_trace_history'),
                            ('t_trace','Text posterior trace ratio over training','H10_text_trace_history')]:
        count+=_save_matrix(out,mats[stat],[f'E{e}' for e in epochs],[_label(d) for d in datasets],
                   title=title,bar_label='DAVI / Branch trace ratio',basename=name,
                   note='Fixed train images; per-seed DAVI/Branch ratio then mean across seeds. Epoch 0 may reflect native initialization.',
                   fmt='.3f') is not None
    return count


def _feature_maps(root,out,datasets,shot,matrix_size):
    files=sorted((root/'summary/ppt_feature_matrices').glob('*.npz'))
    if not files:
        print('[SKIP] no paired feature matrices; run SAVE_MATRICES=1 bash experiments/supplementary/run_ppt_pairs.sh')
        return 0
    count=0
    for path in files:
        with np.load(path,allow_pickle=False) as x:
            d=str(x['dataset'].item());s=int(x['seed']);k=int(x['shot']);
            if d not in datasets or k!=shot:continue
            n=min(matrix_size,int(x['n_images']))
            for key in ('main','rep','text'):
                z=np.asarray(x[key],dtype=float)
                n1=min(n,z.shape[0],z.shape[1]);mat=z[:n1,:n1]
                name=f'H11_{key}_crossmodel_{d}_seed{s}'
                title=f'{_label(d)} seed{s} — DAVI vs Branch {key} cosine'
                count+=_save_matrix(out,mat,[str(i+1) for i in range(n1)],
                            [str(i+1) for i in range(n1)],title=title,
                            bar_label='cosine similarity',basename=name,cmap='coolwarm',
                            vmin=-1,vmax=1,text=False,
                            note='Rows DAVI, columns Branch; same ordered inputs/classes on both axes. Diagonal = matched instance.') is not None
    return count


def _draw_cosine_maps(root,out,datasets,shot):
    files=sorted((root/'summary/ppt_gradient_matrices').glob('*.json'))
    if not files:
        print('[SKIP] no JxJ gradient matrices; run bash experiments/supplementary/run_ppt_gradient_matrices.sh')
        return 0
    count=0
    for metadata in files:
        m=json.loads(metadata.read_text(encoding='utf-8'))
        if int(m['shot'])!=shot or m['dataset'] not in datasets:continue
        if m['method']!='branch' or m['gradient_target']!='branch_mc_classification_only':
            raise RuntimeError('Unsupported gradient matrix target: '+str(metadata))
        npz=metadata.with_suffix('.npz')
        if not npz.is_file():raise FileNotFoundError(npz)
        with np.load(npz,allow_pickle=False) as tensors:
            for key in GROUPS:
                z=np.array(tensors[key],dtype=float)
                if z.ndim!=2 or z.shape[0]!=z.shape[1] or len(z)!=int(m['draws']):
                    raise RuntimeError('Invalid JxJ cosine matrix '+str(npz))
                n=len(z)
                count+=_save_matrix(out,z,[str(i+1) for i in range(n)],
                      [str(i+1) for i in range(n)],
                      title=(f'{_label(m["dataset"])} seed{m["seed"]} E{m["epoch"]} — '
                        f'{GROUP_LABELS[key]} posterior-draw gradient cosine'),
                      bar_label='cosine similarity',
                      basename=f'H12_draw_cosine_{m["dataset"]}_seed{m["seed"]}_E{m["epoch"]}_{key}',
                      cmap='coolwarm',vmin=-1,vmax=1,text=n<=12,fmt='.2f',
                      note='Branch classification gradients; shared checkpoint and input; different posterior draws.') is not None
    return count


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='output_supplementary_v7')
    p.add_argument('--datasets',nargs='+',default=['caltech101','eurosat','oxford_pets'])
    p.add_argument('--shot',type=int,default=16)
    p.add_argument('--only',choices=('all','summary','gradients','features'),default='all')
    p.add_argument('--matrix-size',type=int,default=32,
                   help='Maximum rows and columns in saved cross-model feature maps')
    a=p.parse_args(argv)
    if a.matrix_size<2:p.error('--matrix-size must be >=2')
    root=Path(a.output)
    if not root.is_absolute():root=ROOT/root
    root=root.resolve()
    if not root.is_dir():p.error('Output directory does not exist: '+str(root))
    out=root/'figures'/'ppt_heatmaps'
    count=0
    if a.only in ('all','summary'):
        count+=_paired_reference(root,out,a.datasets,a.shot)
        count+=_e4_posterior_history(root,out,a.datasets,a.shot)
    if a.only in ('all','gradients'):
        count+=_e3_quality(root,out,a.datasets,a.shot)
        count+=_gradient_noise(root,out,a.datasets,a.shot)
        count+=_draw_cosine_maps(root,out,a.datasets,a.shot)
    if a.only in ('all','features'):
        count+=_feature_maps(root,out,a.datasets,a.shot,a.matrix_size)
    print(f'[DONE] Generated {count} new figures (PNG+PDF) in {out}')
    if not count:
        p.error('No valid input diagnostics found. Run study/report or paired checkpoint diagnostics first.')

if __name__=='__main__':
    main()
