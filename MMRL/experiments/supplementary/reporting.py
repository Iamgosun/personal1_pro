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
    """Accept E5 pairs only with common elapsed-time and source protocols."""
    groups=defaultdict(list)
    for row in rows:
        groups[(row['dataset'],row['shot'],row['seed'])].append(row)
    accepted=[];rejected=[]
    for (dataset,shot,seed),records in sorted(groups.items()):
        reasons=[]
        if len(records)!=2 or {r['method'] for r in records}!={'branch','davi'}:
            reasons.append('requires exactly one Branch and one DAVI run')
        checks=(
            ('budget_seconds','different declared budgets'),
            ('native_max_epoch','different native epoch/LR scheduler horizon'),
            ('batch_size','different training batch sizes'),
            ('train_optimizer_name','different optimizer names'),
            ('train_lr','different optimizer learning rates'),
            ('train_lr_scheduler','different LR schedulers'),
            ('train_warmup_epoch','different warmup settings'),
            ('train_precision','different training precision'),
            ('cuda_device','different training hardware'),
            ('mc_repeats','different reference independent MC repeat counts'),
            ('s_ref','different reference MC sample counts'),
            ('n_mc_test','different formal inference sample counts'),
            ('reference_split','different primary reference splits'),
            ('reference_input_sha256','reference split/input digests differ'),
            ('test_reference_input_sha256','test reference split/input digests differ'),
            ('formal_split','formal evaluation splits differ'),
        )
        for key,reason in checks:
            if len({r.get(key) for r in records})!=1:
                reasons.append(reason)
        for r in records:
            if r.get('source_issues'):
                reasons.append(r['method']+': '+str(r['source_issues']))
            if r.get('ambiguous_reference'):
                reasons.append(r['method']+': multiple reference sample counts')
            if not r.get('budget_reached',False):
                reasons.append(r['method']+': budget not reached')
            if r.get('train_seconds',-1)<r.get('budget_seconds',float('inf')):
                reasons.append(r['method']+': actual training time below budget')
            step_cap=r.get('last_step_seconds')
            if step_cap is not None and r.get('train_seconds',0)-r.get('budget_seconds',0)>step_cap+0.25:
                reasons.append(r['method']+': overshoot exceeds final training step')
        if reasons:
            rejected.append(dict(dataset=dataset,shot=shot,seed=seed,
                methods=','.join(sorted(r['method'] for r in records)),
                reasons='; '.join(dict.fromkeys(reasons))))
        else:
            accepted.extend(records)
    return accepted,rejected


from .reference_cache import REFERENCE_SCHEMA

def e5_candidate_from_training(path):
    """Construct E5 table row only from exact checkpoint-backed artifacts.

    An invalid candidate is RETAINED with source_issues and rejected later;
    report alone never silently chooses a stale diagnostic by epoch/seed.
    """
    from .core import sha256
    path=Path(path)
    run=path.parent
    info=read_json(path)
    meta=read_json(run/'manifest.json')
    epoch=info.get('last_checkpoint_epoch')
    row=dict(dataset=meta['dataset'],method=meta['method'],shot=meta['shot'],
        seed=meta['seed'],budget_seconds=info['budget_seconds'],
        train_seconds=info['elapsed_training_seconds'],updates=info['updates'],
        budget_reached=info['budget_reached'],last_step_seconds=info.get('last_step_seconds'),
        native_max_epoch=meta.get('native_max_epoch'),batch_size=meta.get('batch_size'),
        train_optimizer_name=meta.get('train_optimizer_name'),train_lr=meta.get('train_lr'),
        train_lr_scheduler=meta.get('train_lr_scheduler'),
        train_warmup_epoch=meta.get('train_warmup_epoch'),
        train_precision=meta.get('train_precision'),cuda_device=meta.get('cuda_device'),
        last_checkpoint_epoch=epoch,completed_epochs=info.get('completed_epochs'),
        partial_epoch=info.get('partial_epoch'),
        last_update_batch_in_epoch=info.get('last_update_batch_in_epoch'),
        checkpoint_sha256=info.get('last_checkpoint_sha256'),
        s_ref=None,mc_repeats=None,n_mc_test=None,
        reference_split='train_fixed',reference_input_sha256=None,
        test_reference_input_sha256=None,formal_split='test',
        train_branch_nll_ref=None,train_original_nll_ref=None,
        test_branch_nll_ref=None,test_original_nll_ref=None,
        branch_nll_ref=None,original_nll_ref=None,
        fusion_acc=None,fusion_ece=None,ambiguous_reference=False)
    errors=[]
    if meta.get('study')!='matched_time' or meta.get('status')!='complete':
        errors.append('incomplete or non-E5 manifest')
    if epoch is None:
        errors.append('missing budget checkpoint epoch')
    model=run/'refactor_model'/f'model.pth.tar-{epoch}'
    actual_sha=sha256(model) if model.is_file() else None
    if not actual_sha or not row['checkpoint_sha256'] or actual_sha!=row['checkpoint_sha256']:
        errors.append('missing or mismatched training checkpoint SHA256')
    if actual_sha!=meta.get('last_checkpoint_sha256'):
        errors.append('manifest/checkpoint SHA256 mismatch')
    formal=run/'formal'/'latest'/f'epoch_{epoch}'/'complete'/'test.json'
    if not formal.is_file():
        errors.append('missing formal test result')
    else:
        f=read_json(formal)
        if f.get('partial') or f.get('checkpoint',{}).get('checkpoint_sha256')!=actual_sha or \
            f.get('checkpoint',{}).get('epoch')!=epoch or f.get('checkpoint',{}).get('selection')!='latest':
            errors.append('stale/mismatched formal checkpoint')
        if (f.get('dataset'),f.get('method'),f.get('shot'),f.get('seed')) != \
            (row['dataset'],row['method'],row['shot'],row['seed']):
            errors.append('formal run identity mismatch')
        if not errors or ('stale/mismatched formal checkpoint' not in errors):
            row.update(n_mc_test=f.get('n_mc_test'),
                fusion_acc=f.get('fusion_accuracy'),
                fusion_ece=f.get('metrics',{}).get('ece'))
    candidates=sorted((run/'diagnostics'/'latest'/f'epoch_{epoch}').glob('S*_R*/'))
    if not candidates:
        errors.append('missing common MC reference for train_fixed and test')
    if len(candidates)!=1:
        errors.append('multiple reference sample settings or missing unique setting')
        row['ambiguous_reference']=True
    if len(candidates)==1:
        folder=candidates[0]
        references={}
        for split in ('train_fixed','test'):
            rp=folder/split/'metrics.json'
            if not rp.is_file():
                errors.append('missing '+split+' reference')
                continue
            v=read_json(rp)
            if (v.get('partial') or v.get('split')!=split or
                v.get('checkpoint',{}).get('checkpoint_sha256')!=actual_sha or
                v.get('checkpoint',{}).get('epoch')!=epoch or
                v.get('checkpoint',{}).get('selection')!='latest' or
                v.get('reference_schema')!=REFERENCE_SCHEMA or
                v.get('diagnostic_limit_batches') is not None):
                errors.append('stale or partial '+split+' reference')
            if (v.get('dataset'),v.get('method'),v.get('shot'),v.get('seed')) != \
                (row['dataset'],row['method'],row['shot'],row['seed']):
                errors.append(split+' reference identity mismatch')
            if not v.get('input_sha256') or not v.get('n_images'):
                errors.append('missing '+split+' full input identity/count')
            if v.get('s_ref') is None or v.get('mc_repeats') is None:
                errors.append('missing '+split+' MC sampling identity')
            elif folder.name != f'S{v["s_ref"]}_R{v["mc_repeats"]}':
                errors.append(split+' sampling count differs from its folder')
            for metric in ('original_nll','branch_nll'):
                value=v.get(metric)
                if not isinstance(value,(int,float)) or not math.isfinite(value):
                    errors.append('invalid '+split+' '+metric)
            references[split]=v
        if len(references)==2:
            tr,te=references['train_fixed'],references['test']
            if (tr['s_ref'],tr['mc_repeats'])!=(te['s_ref'],te['mc_repeats']):
                errors.append('train/test reference MC settings disagree')
            row.update(s_ref=tr['s_ref'],mc_repeats=tr['mc_repeats'],
                reference_input_sha256=tr['input_sha256'],
                test_reference_input_sha256=te['input_sha256'],
                train_branch_nll_ref=tr.get('branch_nll'),
                train_original_nll_ref=tr.get('original_nll'),
                test_branch_nll_ref=te.get('branch_nll'),
                test_original_nll_ref=te.get('original_nll'),
                branch_nll_ref=tr.get('branch_nll'),
                original_nll_ref=tr.get('original_nll'))
    if row['n_mc_test'] is None or row['n_mc_test']<1:
        errors.append('invalid formal MC test sample count')
    row['source_issues']='; '.join(dict.fromkeys(errors))
    return row

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
            mc_precision_warning=v.get('mc_precision_warning',False),
            mc_repeat_sd_gap1=v.get('mc_repeat_sd',{}).get('gap1'),
            mc_repeat_sd_gap2=v.get('mc_repeat_sd',{}).get('gap2'),
            mc_repeat_spread_gap1=v.get('mc_repeat_spread',{}).get('gap1'),
            mc_repeat_spread_gap2=v.get('mc_repeat_spread',{}).get('gap2'),
            mc_mean_repeat_abs_delta_main=v.get('mc_moment_repeat_abs_delta',{}).get('mu_main_mc_repeat_abs_delta'),
            mc_var_repeat_abs_delta_main=v.get('mc_moment_repeat_abs_delta',{}).get('var_main_mc_repeat_abs_delta'),
            mc_mean_repeat_abs_delta_rep=v.get('mc_moment_repeat_abs_delta',{}).get('mu_rep_mc_repeat_abs_delta'),
            mc_var_repeat_abs_delta_rep=v.get('mc_moment_repeat_abs_delta',{}).get('var_rep_mc_repeat_abs_delta'),
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
            rep_margin_median=v.get('rep_margin_median'),rep_margin_type=v.get('rep_margin_type'),
            rep_logp_margin_median=v.get('rep_logp_margin_median'),
            rep_logp_margin_type=v.get('rep_logp_margin_type'),
            reference_schema=v.get('reference_schema')))
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
         'propagation_and_native_variance_error',
         'mc_repeat_sd_gap1','mc_repeat_sd_gap2','mc_repeat_spread_gap1','mc_repeat_spread_gap2',
         'mc_mean_repeat_abs_delta_main','mc_var_repeat_abs_delta_main',
         'mc_mean_repeat_abs_delta_rep','mc_var_repeat_abs_delta_rep']
    csv_write(out/'table_s2_runs.csv',s2)
    csv_write(out/'table_s2_summary.csv',aggregate_seeds(s2,
        ['dataset','method','shot','checkpoint','s_ref','mc_repeats'],s2_metrics))

    # Table S3 is the V7 MC-mean-logit-margin experiment, not legacy BMA margin.
    # E4 rows must still correspond to the actual checkpoint files. A same
    # seed/path retrain can replace weights; reporting must not trust old JSON.
    from .core import sha256
    from .reference_cache import initial_snapshot_hash
    s3=[]
    for r in native_ref:
        from .reference_cache import REFERENCE_SCHEMA
        if r.get('reference_schema')!=REFERENCE_SCHEMA or r.get('rep_margin_type')!='mc_mean_logit':
            continue
        run_dir=root/r['study']/r['dataset']/r['method']/f'shots_{r["shot"]}'/f'seed_{r["seed"]}'
        # Locate only the exact method/selection/epoch/sampling result; no
        # fallback to another seed or a different saved epoch.
        for mpath in sorted((run_dir/'diagnostics').glob('**/test/metrics.json')):
            obj=read_json(mpath)
            if (checkpoint_key(obj)!=r['checkpoint'] or obj.get('s_ref')!=r['s_ref']
                or obj.get('mc_repeats',1)!=r['mc_repeats']
                or obj.get('rep_margin_type')!='mc_mean_logit'):
                continue
            ck=obj.get('checkpoint',{})
            path=Path(ck.get('path',''))
            if (path.is_file() and ck.get('checkpoint_sha256')==sha256(path)
                and obj.get('initial_checkpoint_sha256')==initial_snapshot_hash(run_dir)):
                s3.append(r)
            else:
                print(f'[S3 WARNING] Stale or missing source checkpoint: {mpath}',flush=True)
            break

    s3_metrics=['r_mean_fro','r_trace','t_trace','visual_prompt_init_cos',
        'text_prompt_init_cos','visual_clip_cos','text_clip_cos_mean','rep_margin_median']
    csv_write(out/'table_s3_runs.csv',s3)
    csv_write(out/'table_s3_summary.csv',aggregate_seeds(s3,
        ['dataset','method','shot','checkpoint','s_ref','mc_repeats','rep_margin_type'],s3_metrics))

    # R15 E4 time-course: light posterior stats + stratified fixed-input
    # Rep MC-mean logit margins, not a replacement for final Table S3.
    e4_evolution_rows=[]
    for p in sorted(root.glob('native/**/diagnostics/**/e4_evolution/**/metrics.json')):
        try:v=read_json(p)
        except (OSError,ValueError):continue
        if v.get('schema')!=1 or v.get('partial') or v.get('rep_margin_type')!='mc_mean_logit':continue
        ck=v.get('checkpoint',{});cp=Path(ck.get('path',''))
        from .reference_cache import initial_snapshot_hash
        parts=p.parts;run_dir=Path(*parts[:parts.index('diagnostics')])
        if v.get('initial_checkpoint_sha256')!=initial_snapshot_hash(run_dir):continue
        if not cp.is_file() or ck.get('checkpoint_sha256')!=sha256(cp):continue
        settings=v['settings']
        e4_evolution_rows.append(dict(dataset=v['dataset'],method=v['method'],
             shot=v['shot'],seed=v['seed'],epoch=ck['epoch'],
             checkpoint_sha256=ck['checkpoint_sha256'],input_sha256=v['input']['input_sha256'],
             images=settings['images'],draws=settings['draws'],
             chunk=settings['chunk'],batch=settings['batch'],input_seed=settings['input_seed'],
             r_mean_fro=v['r_mean_fro'],r_trace=v['r_trace'],t_trace=v['t_trace'],
             rep_margin_median=v['rep_margin_median'],rep_margin_mean=v['rep_margin_mean'],
             rep_margin_positive_fraction=v['rep_margin_positive_fraction']))
    csv_write(out/'e4_evolution_runs.csv',e4_evolution_rows)
    e4_metrics=['r_mean_fro','r_trace','t_trace','rep_margin_median',
                'rep_margin_mean','rep_margin_positive_fraction']
    csv_write(out/'e4_evolution_summary.csv',aggregate_seeds(e4_evolution_rows,
        ['dataset','method','shot','epoch','images','draws','chunk','batch','input_seed'], e4_metrics))

    # E3: one record per real DAVI checkpoint and per MC budget, including the
    # deterministic gradient. One seed or two observed epochs are valid pilots.
    # This is not an optimization-performance table or a test-set ACC metric.
    e3_rows=[]
    for p in sorted(root.glob('native/**/diagnostics/**/gradient.json')):
        v=read_json(p)
        if v.get('partial') or v.get('method')!='davi':continue
        inp=v.get('input',{}); c=v['checkpoint']
        if c.get('epoch') is None:continue
        group=defaultdict(list)
        for r in v.get('curve',[]):group[int(r['samples'])].append(r)
        for budget,rs in sorted(group.items()):
            record=dict(dataset=v['dataset'],shot=v['shot'],seed=v['seed'],
                epoch=int(c['epoch']),checkpoint_selection=c['selection'],
                checkpoint_sha256=c.get('checkpoint_sha256'),
                input_sha256=inp.get('input_sha256'),
                n_images=inp.get('n_images'),n_classes=inp.get('n_unique_classes'),
                input_strategy=inp.get('selection_strategy'),
                input_seed=inp.get('selection_seed'),
                reference_samples=v['reference_samples'],
                reference_stable=v.get('reference_stable'),
                reference_repeat_cosine=v.get('reference_repeat_quality',{}).get('cosine'),
                reference_repeat_relative_error=v.get('reference_repeat_quality',{}).get('relative_error'),
                chunk=v.get('e3_parameters',{}).get('chunk'),
                mc_repeats=len(rs),mc_samples=budget,
                davi_cosine=v['davi']['cosine'],
                davi_relative_error=v['davi']['relative_error'],
                mc_cosine_mean=statistics.mean(r['cosine'] for r in rs),
                mc_cosine_std=statistics.stdev(r['cosine'] for r in rs) if len(rs)>1 else '',
                mc_relative_error_mean=statistics.mean(r['relative_error'] for r in rs),
                mc_relative_error_std=statistics.stdev(r['relative_error'] for r in rs) if len(rs)>1 else '')
            e3_rows.append(record)
    csv_write(out/'e3_checkpoint_comparison.csv',e3_rows)

    # E5 reads sources from the exact run folder, never from generic
    # epoch-only indices (which can select a different checkpoint/split).
    budget_runs=[]
    for path in sorted(root.glob('matched_time/**/training.json')):
        try:
            budget_runs.append(e5_candidate_from_training(path))
        except (KeyError,ValueError,TypeError,FileNotFoundError,OSError) as exc:
            print(f'[E5 REPORT WARNING] Cannot parse {path}: {exc}',flush=True)
    csv_write(out/'table_s4_candidate_runs.csv',budget_runs)
    matched,rejected=paired_e5(budget_runs)
    csv_write(out/'table_s4_rejected_pairs.csv',rejected)
    csv_write(out/'table_s4_matched_runs.csv',matched)
    by_pair=defaultdict(dict)
    for r in matched:
        by_pair[(r['dataset'],r['shot'],r['seed'],r['budget_seconds'],
                 r['s_ref'],r['mc_repeats'],r['n_mc_test'])][r['method']]=r
    differences=('train_seconds','updates','train_branch_nll_ref','train_original_nll_ref',
                 'test_branch_nll_ref','test_original_nll_ref','fusion_acc','fusion_ece')
    pair_deltas=[]
    for key,pair in sorted(by_pair.items()):
        if set(pair)!={'branch','davi'}:
            raise RuntimeError('Matched E5 group unexpectedly lost a model')
        branch,davi=pair['branch'],pair['davi']
        pair_deltas.append(dict(zip(('dataset','shot','seed','budget_seconds',
             's_ref','mc_repeats','n_mc_test'),key)) |
             {k+'_davi_minus_branch':davi[k]-branch[k] for k in differences})
    csv_write(out/'table_s4_paired_deltas.csv',pair_deltas)
    csv_write(out/'table_s4_paired_deltas_summary.csv',aggregate_seeds(
        pair_deltas,['dataset','shot','budget_seconds','s_ref','mc_repeats','n_mc_test'],
        [k+'_davi_minus_branch' for k in differences]))
    s4_metrics=list(differences)
    csv_write(out/'table_s4_summary.csv',aggregate_seeds(matched,
        ['dataset','method','shot','budget_seconds','s_ref','mc_repeats','n_mc_test'],s4_metrics))
    return {'formal_runs':len(formal),'reference_runs':len(reference),
            's1_runs':len(native_formal),'s2_runs':len(s2),'s3_runs':len(s3),
            'e3_checkpoint_rows':len(e3_rows),
            'e5_candidates':len(budget_runs),'e5_accepted_runs':len(matched),
            'e5_rejected_pairs':len(rejected),'summary_directory':str(out)}
