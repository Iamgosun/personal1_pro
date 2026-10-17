"""E1--E5 tables from validated run-level evidence, never renaming generic CSVs.

Pure Python. Invalid matched-time pairs are excluded from Table S4, with reasons.
Nothing in this module launches training, consumes GPU, or fabricates observations.
"""
from __future__ import annotations
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def csv_write(path, rows, columns=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if columns is None:columns=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    return path


def aggregate_seeds(rows,group_keys,measure_keys):
    groups=defaultdict(list)
    for row in rows:groups[tuple(row[k] for k in group_keys)].append(row)
    summary=[]
    for key,group in sorted(groups.items()):
        seeds=[g['seed'] for g in group]
        if len(seeds)!=len(set(seeds)):
            raise ValueError('Duplicate seed in table group: '+repr(key))
        record=dict(zip(group_keys,key));record['n_seeds']=len(group)
        for metric in measure_keys:
            vals=[]
            for g in group:
                v=g.get(metric)
                if v is None or v=='':break
                try:value=float(v)
                except (TypeError,ValueError):break
                if not math.isfinite(value):break
                vals.append(value)
            if len(vals)==len(group):
                record[metric+'_mean']=statistics.mean(vals)
                record[metric+'_std']=statistics.stdev(vals) if len(vals)>1 else ''
        summary.append(record)
    return summary


def checkpoint_key(obj):
    c=obj['checkpoint'];sel=c['selection']
    return sel if c.get('epoch') is None else f'{sel}_epoch_{c["epoch"]}'


def identity(row):
    return (row['study'],row['dataset'],row['method'],row['shot'],row['seed'],row['checkpoint'])


def paired_e5(rows):
    """Only pairs satisfying the *same* experiment protocol enter Table S4.

    Raw individual runs must NOT automatically become a matched-time comparison.
    """
    groups=defaultdict(list)
    for row in rows:
        groups[(row['dataset'],row['shot'],row['seed'])].append(row)
    accepted=[];rejected=[]
    for (dataset,shot,seed),records in sorted(groups.items()):
        reasons=[]
        if len(records)!=2 or {r['method'] for r in records}!={'branch','davi'}:
            reasons.append('requires exactly one Branch and one DAVI run')
        if len({r['budget_seconds'] for r in records})!=1:
            reasons.append('different declared budgets')
        if len({r['native_max_epoch'] for r in records})!=1:
            reasons.append('different native epoch/LR scheduler horizon')
        if len({r['batch_size'] for r in records})!=1:
            reasons.append('different training batch sizes')
        if len({r['mc_repeats'] for r in records})!=1:
            reasons.append('different reference independent MC repeat counts')
        if len({r['s_ref'] for r in records})!=1:
            reasons.append('different reference MC sample counts')
        if len({r['n_mc_test'] for r in records})!=1:
            reasons.append('different formal inference sample counts')
        if any(not r.get('reference_input_sha256') for r in records):
            reasons.append('missing reference input digest')
        if len({r.get('reference_split') for r in records})!=1 or \
           len({r.get('reference_input_sha256') for r in records})!=1:
            reasons.append('reference split/input digests differ')
        if len({r.get('formal_split') for r in records})!=1:
            reasons.append('formal evaluation splits differ')
        for r in records:
            if r.get('ambiguous_reference'):reasons.append(r['method']+': multiple reference sample counts')
            if not r.get('budget_reached',False):reasons.append(r['method']+': budget not reached')
            if r['train_seconds'] < r['budget_seconds']:
                reasons.append(r['method']+': actual training time below budget')
            # One optimizer step may exceed the budget by its own duration.
            step_cap=r.get('last_step_seconds')
            if step_cap is not None and r['train_seconds']-r['budget_seconds'] > step_cap+0.25:
                reasons.append(r['method']+': overshoot exceeds final training step')
        if reasons:
            rejected.append(dict(dataset=dataset,shot=shot,seed=seed,
                                 methods=','.join(sorted(r['method'] for r in records)),
                                 reasons='; '.join(dict.fromkeys(reasons))))
        else:
            accepted.extend(records)
    return accepted,rejected


def make_tables(root):
    root=Path(root);out=root/'summary';out.mkdir(parents=True,exist_ok=True)
    formal=[];reference=[];traces=[]
    # ** allows trajectory epoch_N and last-update epoch_N paths.
    for p in sorted(root.glob('**/formal/**/complete/test.json')):
        v=read_json(p)
        if v.get('partial'):continue
        formal.append(dict(study=p.relative_to(root).parts[0],dataset=v['dataset'],method=v['method'],
            shot=v['shot'],seed=v['seed'],checkpoint=checkpoint_key(v),
            n_mc_test=v['n_mc_test'],formal_split='test',
            fusion_acc=v['fusion_accuracy'],fusion_ece=v['metrics']['ece'],
            fusion_aece=v['metrics']['aece'],main_acc=v['main_accuracy'],
            rep_acc=v['rep_accuracy'],rep_nll=v['rep_predictive_nll']))
    for p in sorted(root.glob('**/diagnostics/**/S*/test/metrics.json')):
        v=read_json(p)
        if v.get('partial'):continue
        reference.append(dict(study=p.relative_to(root).parts[0],dataset=v['dataset'],method=v['method'],
            shot=v['shot'],seed=v['seed'],checkpoint=checkpoint_key(v),
            s_ref=v['s_ref'],mc_repeats=v.get('mc_repeats',1),
            mc_reference_stable=v.get('mc_reference_stable'),
            reference_split='test',objective_type=v['objective_type'],
            reference_input_sha256=v.get('input_sha256'),
            original_nll=v['original_nll'],branch_nll=v['branch_nll'],gap1=v['gap1'],
            det_nll=v.get('det_nll'),gap2=v.get('gap2'),abs_gap2=v.get('abs_gap2'),
            mean_mae=v.get('mean_mae'),var_mae=v.get('var_mae'),
            empirical_jensen_gap=v.get('empirical_jensen_gap'),
            gaussian_mgf_error=v.get('gaussian_mgf_error'),
            propagation_and_native_variance_error=v.get('propagation_and_native_variance_error'),
            r_mean_fro=v['r_mean_fro'],r_trace=v['r_trace'],t_trace=v['t_trace'],
            visual_prompt_init_cos=v.get('visual_prompt_init_cos'),
            text_prompt_init_cos=v.get('text_prompt_init_cos'),
            visual_clip_cos=v.get('visual_clip_cos'),text_clip_cos_mean=v.get('text_clip_cos_mean'),
            rep_margin_median=v.get('rep_margin_median'),rep_margin_type=v.get('rep_margin_type')))
    for p in sorted(root.glob('native/**/trace/trajectory.csv')):
        with p.open(encoding='utf-8',newline='') as f:
            traces.extend(csv.DictReader(f))
    csv_write(out/'formal_runs.csv',formal)
    csv_write(out/'reference_runs.csv',reference)
    if traces:csv_write(out/'e1_trajectory_runs.csv',traces)

    # Keep the *actual* checkpoint selection in every paper table. Native
    # TrainerX is not required to write model-best.pth.tar; E1 defaults to its
    # final-epoch checkpoint, never a checkpoint falsely relabeled 'best'.
    native_formal=[x for x in formal if x['study']=='native' and (
        x['checkpoint']=='best' or x['checkpoint'].startswith('latest_epoch_'))]
    native_ref=[x for x in reference if x['study']=='native' and (
        x['checkpoint']=='best' or x['checkpoint'].startswith('latest_epoch_'))]
    s1_metrics=['fusion_acc','main_acc','rep_acc','rep_nll','fusion_ece','fusion_aece']
    csv_write(out/'table_s1_runs.csv',native_formal)
    csv_write(out/'table_s1_summary.csv',aggregate_seeds(native_formal,
        ['dataset','method','shot','checkpoint','n_mc_test'],s1_metrics))

    # S2 is the native DAVI surrogate; optional counterfactuals are not a training objective.
    s2=[r for r in native_ref if r['method']=='davi' and r['objective_type']=='native_davi']
    s2_metrics=['original_nll','branch_nll','det_nll','gap1','gap2','abs_gap2',
         'mean_mae','var_mae','empirical_jensen_gap','gaussian_mgf_error',
         'propagation_and_native_variance_error']
    csv_write(out/'table_s2_runs.csv',s2)
    csv_write(out/'table_s2_summary.csv',aggregate_seeds(s2,
        ['dataset','method','shot','checkpoint','s_ref','mc_repeats'],s2_metrics))

    s3_metrics=['r_mean_fro','r_trace','t_trace','visual_prompt_init_cos',
        'text_prompt_init_cos','visual_clip_cos','text_clip_cos_mean','rep_margin_median']
    csv_write(out/'table_s3_runs.csv',native_ref)
    csv_write(out/'table_s3_summary.csv',aggregate_seeds(native_ref,
        ['dataset','method','shot','checkpoint','s_ref','mc_repeats','rep_margin_type'],s3_metrics))

    budget_runs=[]
    formal_index={identity(r):r for r in formal if r['study']=='matched_time'}
    # Retain sample count as an explicit dimension; reject ambiguous duplicates.
    ref_index=defaultdict(list)
    for r in reference:
        if r['study']=='matched_time':ref_index[identity(r)].append(r)
    for p in sorted(root.glob('matched_time/**/training.json')):
        info=read_json(p);meta=read_json(p.parent/'manifest.json')
        key=('matched_time',meta['dataset'],meta['method'],meta['shot'],meta['seed'],
             f'latest_epoch_{info["last_checkpoint_epoch"]}')
        f=formal_index.get(key);rs=ref_index.get(key,[])
        if not f or not rs:continue
        if len(rs)!=1:
            # Multiple S_ref reference evaluations for one run cannot be selected silently.
            for rr in rs:
                budget_runs.append(dict(dataset=meta['dataset'],method=meta['method'],shot=meta['shot'],
                    seed=meta['seed'],budget_seconds=info['budget_seconds'],
                    train_seconds=info['elapsed_training_seconds'],updates=info['updates'],
                    budget_reached=info['budget_reached'],last_step_seconds=info.get('last_step_seconds'),
                    native_max_epoch=meta.get('native_max_epoch'),batch_size=meta.get('batch_size'),
                    s_ref=rr['s_ref'],n_mc_test=f['n_mc_test'],
                    mc_repeats=rr['mc_repeats'],
                    reference_split=rr['reference_split'],reference_input_sha256=rr['reference_input_sha256'],
                    formal_split=f['formal_split'],branch_nll_ref=rr['branch_nll'],
                    original_nll_ref=rr['original_nll'],fusion_acc=f['fusion_acc'],
                    fusion_ece=f['fusion_ece'],ambiguous_reference=True))
            continue
        rr=rs[0]
        budget_runs.append(dict(dataset=meta['dataset'],method=meta['method'],shot=meta['shot'],
            seed=meta['seed'],budget_seconds=info['budget_seconds'],
            train_seconds=info['elapsed_training_seconds'],updates=info['updates'],
            budget_reached=info['budget_reached'],last_step_seconds=info.get('last_step_seconds'),
            native_max_epoch=meta.get('native_max_epoch'),batch_size=meta.get('batch_size'),
            s_ref=rr['s_ref'],n_mc_test=f['n_mc_test'],
            mc_repeats=rr['mc_repeats'],
            reference_split=rr['reference_split'],reference_input_sha256=rr['reference_input_sha256'],
            formal_split=f['formal_split'],branch_nll_ref=rr['branch_nll'],
            original_nll_ref=rr['original_nll'],fusion_acc=f['fusion_acc'],
            fusion_ece=f['fusion_ece'],ambiguous_reference=False))
    csv_write(out/'table_s4_candidate_runs.csv',budget_runs)
    matched,rejected=paired_e5(budget_runs)
    csv_write(out/'table_s4_rejected_pairs.csv',rejected)
    csv_write(out/'table_s4_matched_runs.csv',matched)
    # Paired within-seed differences are more interpretable than differences
    # between unpaired cross-seed method averages.
    by_pair=defaultdict(dict)
    for r in matched:
        by_pair[(r['dataset'],r['shot'],r['seed'],r['budget_seconds'],
                 r['s_ref'],r['mc_repeats'],r['n_mc_test'])][r['method']]=r
    pair_deltas=[]
    for key,pair in sorted(by_pair.items()):
        if set(pair)!={'branch','davi'}:
            raise RuntimeError('Matched E5 group unexpectedly lost a model')
        branch,davi=pair['branch'],pair['davi']
        pair_deltas.append(dict(zip(('dataset','shot','seed','budget_seconds',
             's_ref','mc_repeats','n_mc_test'),key))|
             {k+'_davi_minus_branch':davi[k]-branch[k] for k in
                ('train_seconds','updates','branch_nll_ref','original_nll_ref',
                 'fusion_acc','fusion_ece')})
    csv_write(out/'table_s4_paired_deltas.csv',pair_deltas)
    csv_write(out/'table_s4_paired_deltas_summary.csv',aggregate_seeds(
        pair_deltas,['dataset','shot','budget_seconds','s_ref','mc_repeats','n_mc_test'],
        [k+'_davi_minus_branch' for k in ('train_seconds','updates','branch_nll_ref',
          'original_nll_ref','fusion_acc','fusion_ece')]))
    s4_metrics=['train_seconds','updates','branch_nll_ref','original_nll_ref','fusion_acc','fusion_ece']
    csv_write(out/'table_s4_summary.csv',aggregate_seeds(matched,
        ['dataset','method','shot','budget_seconds','s_ref','mc_repeats','n_mc_test'],s4_metrics))
    return {'formal_runs':len(formal),'reference_runs':len(reference),
            's1_runs':len(native_formal),'s2_runs':len(s2),'s3_runs':len(native_ref),
            'e5_candidates':len(budget_runs),'e5_accepted_runs':len(matched),
            'e5_rejected_pairs':len(rejected),'summary_directory':str(out)}
