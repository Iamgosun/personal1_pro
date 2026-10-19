"""Only adapters to the *production* DAVI/AVI implementations live here.

Training:
 DAVI = DetBayesRTMMRL + det_bayesrt_mmrl_clamp.yaml (run_plan alias
         DetBayesRTMMRL_CLAMP).
 Branch = BayesRTMMRL + bayesrt_mmrl.yaml.
 Original = BayesRTMMRL with only the classification NLL replaced.
No CLIP, KL, alignment or fusion implementations are duplicated.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODES = {'davi': 'DetBayesRTMMRL', 'branch': 'BayesRTMMRL',
         'original': 'SupplementOriginalMC'}
EXPERIMENT_NAMES = {'davi': 'DetBayesRTMMRL_CLAMP',
                    'branch': 'BayesRTMMRL', 'original': 'AVI-Original'}
CONFIGS = {'davi': 'det_bayesrt_mmrl_clamp.yaml',
           'branch': 'bayesrt_mmrl.yaml', 'original': 'bayesrt_mmrl.yaml'}
SCHEMA = 2


def location(output_root, study, dataset, method, shot, seed):
    out = Path(output_root)
    return (out if out.is_absolute() else ROOT / out) / study / dataset / method / f'shots_{shot}' / f'seed_{seed}'


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for buf in iter(lambda: f.read(1024 * 1024), b''):
            h.update(buf)
    return h.hexdigest()


def original_nll(c, r, y, alpha):
    """MC expected -log(alpha*p_main(y)+(1-alpha)*p_rep(y)); shared draws."""
    import torch
    import torch.nn.functional as F
    if not (0 <= alpha <= 1):
        raise ValueError('alpha must be in [0,1]')
    if c.shape != r.shape or c.ndim != 3 or y.shape != c.shape[1:2]:
        raise ValueError('Expected matching logits [S,B,C] and labels [B]')
    index = y[None, :, None].expand(c.shape[0], -1, 1)
    a = F.log_softmax(c.float(), -1).gather(2, index).squeeze(-1)
    b = F.log_softmax(r.float(), -1).gather(2, index).squeeze(-1)
    if alpha == 0:
        return -b.mean()
    if alpha == 1:
        return -a.mean()
    return -torch.logaddexp(a + math.log(alpha), b + math.log1p(-alpha)).mean()


def register_original():
    from core.registry import METHOD_REGISTRY
    from methods.bayesrt_mmrl.model import BayesRTMMRLMethod
    if 'SupplementOriginalMC' in METHOD_REGISTRY.keys():
        return

    class SupplementOriginalMC(BayesRTMMRLMethod):
        method_name = 'SupplementOriginalMC'

        def _build_train_outputs(self, label, img_ref, out):
            outputs = super()._build_train_outputs(label, img_ref, out)
            alpha = float(self.cfg.BAYESRT_MMRL.ALPHA)
            original = original_nll(out['logits_main_stack'][..., :self.num_classes],
                                    out['logits_rep_stack'][..., :self.num_classes], label, alpha)
            branch = alpha * outputs.losses['loss_main'] + (1-alpha) * outputs.losses['loss_rep']
            correction = original - branch
            # KL and cosine are inherited *without* recomputing or rescaling them.
            outputs.losses['data_term'] = outputs.losses['data_term'] + correction
            outputs.losses['total'] = outputs.losses['total'] + correction
            outputs.losses['original_mc_nll'] = original.detach()
            return outputs

    METHOD_REGISTRY.register('SupplementOriginalMC')(SupplementOriginalMC)


def make_args(dataset, method, shot, seed, output_root, study, data_root,
              epochs=None, mc_train=None, extra=(), trajectory=False, init_snapshot=False,
              e3_snapshot_epochs=()):
    root = Path(data_root)
    if not root.is_absolute():
        root = ROOT / root
    opts = ['DATASET.NUM_SHOTS', str(shot), 'DATASET.SUBSAMPLE_CLASSES', 'all',
            'HPO.ENABLED', 'False']
    # None means KEEP THE NATIVE YAML, including DAVI's unused N_MC_TRAIN=3.
    if epochs is not None:
        opts += ['OPTIM.MAX_EPOCH', str(epochs)]
    if mc_train is not None:
        opts += ['BAYESRT_MMRL.N_MC_TRAIN', str(mc_train)]
    if e3_snapshot_epochs and (trajectory or init_snapshot):
        raise ValueError('E3 selected epoch snapshots must not be mixed with --trajectory/--init-snapshot')
    trainer_name=('SupplementSelectedRunner' if e3_snapshot_epochs else
                  'SupplementTrajectoryRunner' if trajectory else
                  'SupplementInitRunner' if init_snapshot else 'RefactorRunner')
    if trainer_name!='RefactorRunner':
        opts += ['TRAINER.NAME', trainer_name]
    opts += list(extra)
    return Namespace(root=str(root), output_dir=str(location(output_root, study, dataset, method, shot, seed)),
        dataset_config_file=str(ROOT/'configs/datasets'/f'{dataset}.yaml'),
        method_config_file=str(ROOT/'configs/methods'/CONFIGS[method]),
        protocol_config_file=str(ROOT/'configs/protocols/fs.yaml'),
        runtime_config_file=str(ROOT/'configs/runtime/mmrl_family.yaml'),
        exp_config='', method=MODES[method], protocol='FS', exec_mode='online', seed=seed,
        trainer=trainer_name,
        eval_only=False, model_dir='', load_epoch=None, no_train=False, opts=opts)


def setup(ns):
    """Match run.py's YAML sanitization and native trainer construction."""
    import run
    from core.config import setup_cfg
    from dassl.engine import build_trainer
    from dassl.utils import set_random_seed
    from datasets.fair_val_protocol import install_kplusval_datasetbase_patch
    run._import_runtime_modules()
    register_original()
    if any(x in ns.opts for x in ('SupplementInitRunner','SupplementTrajectoryRunner',
                                    'SupplementSelectedRunner')):
        from .runner import register_trajectory
        register_trajectory()
    original = ns.method_config_file
    tmp = run._prepare_method_config_for_kplusval(ns)
    try:
        cfg = setup_cfg(ns)
    finally:
        ns.method_config_file = original
        if tmp:
            Path(tmp).unlink(missing_ok=True)
    if str(cfg.METHOD.NAME) != ns.method:
        raise RuntimeError('Resolved model name differs from experiment model')
    if str(cfg.BAYESRT_MMRL.EVAL_FUSION_VARIANT) != 'static_logit':
        raise RuntimeError('FS experiment expects native EVAL_FUSION_VARIANT=static_logit')
    install_kplusval_datasetbase_patch(cfg)
    set_random_seed(ns.seed)
    trainer = build_trainer(cfg)
    return trainer, cfg


def available_trajectory_epochs(run_dir):
    """Existing E1 snapshots, sorted numerically. Missing epochs are allowed."""
    folder = Path(run_dir) / 'trajectory' / 'refactor_model'
    result = set()
    for path in folder.glob('model.pth.tar-*'):
        suffix = path.name.removeprefix('model.pth.tar-')
        if suffix.isdecimal():
            result.add(int(suffix))
    return sorted(result)


def e1_trace_is_current(run_dir, samples, batches, batch_size):
    """Reuse only an E1 trace computed with the current diagnostic settings/snapshots.

    A mismatch means recompute the *diagnostic*, never retrain the model.
    No fixed epoch count or fixed number of seeds is required.
    """
    folder = Path(run_dir) / 'trace'
    if not (folder / 'trajectory.csv').is_file():
        return False
    try:
        data = json.loads((folder / 'trajectory.json').read_text(encoding='utf-8'))
        return (int(data.get('e1_samples', -1)) == int(samples)
                and int(data.get('e1_batches', -1)) == int(batches)
                and int(data.get('batch_size', -1)) == int(batch_size)
                and data.get('epochs_evaluated') == available_trajectory_epochs(run_dir))
    except (FileNotFoundError, ValueError, TypeError, json.JSONDecodeError):
        return False


def checkpoint_path(run_dir, selection='best', epoch=None):
    run_dir = Path(run_dir)
    if selection == 'trajectory':
        if epoch is None or epoch < 0:
            raise ValueError('trajectory requires --epoch >=0')
        root = run_dir/'trajectory'
        path = root/'refactor_model'/f'model.pth.tar-{epoch}'
        load_epoch = epoch
    elif selection == 'time_curve':
        if epoch not in (0,25,50,75,100):
            raise ValueError('E5 time_curve phase must be one of 0,25,50,75,100')
        root = run_dir/'time_curve'
        path = root/'refactor_model'/f'model.pth.tar-{epoch}'
        load_epoch = epoch
    elif selection == 'best':
        root = run_dir
        path = root/'refactor_model'/'model-best.pth.tar'
        load_epoch = None
    elif selection == 'latest':
        root = run_dir
        matches = list((root/'refactor_model').glob('model.pth.tar-*'))
        if not matches:
            raise FileNotFoundError('No epoch checkpoint under '+str(root))
        path = max(matches, key=lambda p: int(p.name.rsplit('-', 1)[-1]))
        load_epoch = int(path.name.rsplit('-', 1)[-1])
    else:
        raise ValueError(f'Unknown checkpoint selection {selection}')
    if not path.exists():
        if selection == 'best':
            raise FileNotFoundError(str(path) +
                '; the native trainer may only save model.pth.tar-N. '
                'Use --checkpoint latest to explicitly evaluate the last saved epoch, '
                'or produce a genuine validation-best checkpoint.')
        raise FileNotFoundError(path)
    return root, path, load_epoch


def e3_checkpoint_request(run_dir, request):
    """Resolve an E3 epoch to a REAL checkpoint; never infer missing weights.

    Returns (selection, epoch). `latest` remains the native final checkpoint;
    a numeric final epoch also resolves to that file. Intermediate checkpoints
    use snapshots from a prior DAVI run with --e3-snapshot-epochs or --trajectory.
    """
    run_dir=Path(run_dir)
    if request=='latest':
        checkpoint_path(run_dir,'latest')
        return 'latest',None
    if not str(request).isdecimal():
        raise ValueError('E3_EPOCHS accepts only positive epoch numbers or latest')
    epoch=int(request)
    if epoch<1:
        raise ValueError('E3_EPOCHS epoch numbers must be positive')
    try:
        _, _, last=checkpoint_path(run_dir,'latest')
    except FileNotFoundError:
        last=None
    if epoch==last:
        return 'latest',None
    snapshot=run_dir/'trajectory'/'refactor_model'/f'model.pth.tar-{epoch}'
    if snapshot.is_file():
        return 'trajectory',epoch
    raise FileNotFoundError(
        f'E3 requested epoch {epoch} but checkpoint does not exist: {snapshot}. '
        'The saved final epoch cannot reconstruct earlier weights. '
        'Use a NEW OUT (e.g. output_supplementary_v7_e3epochs) with '
        'E3_EPOCHS="5 25 latest" to train DAVI and save selected snapshots; '
        'do not delete or overwrite the existing DAVI run.')


def verify_checkpoint(model, checkpoint_file):
    """Verify every *trainable* parameter from native lightweight checkpoint."""
    import torch
    state = torch.load(checkpoint_file, map_location='cpu', weights_only=False)['state_dict']
    actual = model.state_dict()
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    missing = trainable - set(state)
    if missing:
        raise RuntimeError('Checkpoint lacks trainable tensors: '+repr(sorted(missing)[:8]))
    # Frozen posterior anchors matter for interpreting KL and moment propagation,
    # even though they are not part of the trainable parameter set.
    posterior_buffers=(
        'image_encoder.bayes_proj_rep.prior_mean',
        'image_encoder.bayes_proj_rep.prior_std',
        'text_posterior.posterior_mean',
        'text_posterior.prior_mean',
        'text_posterior.prior_std',
    )
    for key in posterior_buffers:
        if key not in state or key not in actual:
            raise RuntimeError('Lightweight checkpoint missing posterior anchor: '+key)
    for name in trainable.union(posterior_buffers):
        if not torch.equal(state[name].cpu(), actual[name].detach().cpu()):
            raise RuntimeError('Restored trainable parameter mismatch: '+name)
    return {'trainable_tensors_verified': len(trainable), 'checkpoint_sha256': sha256(checkpoint_file)}


def restore(run_dir, gpu='0', selection='best', epoch=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.chdir(ROOT)
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir/'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f'No supplementary manifest at {manifest_path}; arbitrary historical checkpoint import is not assumed')
    meta = json.loads(manifest_path.read_text(encoding='utf-8'))
    ns = make_args(meta['dataset'], meta['method'], meta['shot'], meta['seed'],
        meta['output_root'], meta['study'], meta['data_root'],
        meta['epochs_override'], meta['mc_train_override'], meta['extra'],
        meta['trajectory'], (meta.get('init_snapshot',False) and
                             not meta.get('e3_snapshot_epochs')),
        meta.get('e3_snapshot_epochs',()))
    if Path(ns.output_dir).resolve() != run_dir:
        raise RuntimeError('Manifest output path does not match selected run')
    trainer, cfg = setup(ns)
    if hashlib.sha256(cfg.dump().encode()).hexdigest() != meta['config_sha256']:
        raise RuntimeError('Configuration hash differs from training manifest')
    root, checkpoint, load_epoch = checkpoint_path(run_dir, selection, epoch)
    trainer.load_model(str(root), epoch=load_epoch)
    result = verify_checkpoint(trainer.method.model, checkpoint)
    trainer.method.model.eval()
    return trainer, meta, {**result, 'path': str(checkpoint), 'selection': selection, 'epoch': load_epoch}


def reference_loader(trainer, split, batch_size=32):
    """Fixed-train and held-out test have the same deterministic TEST transform."""
    from torch.utils.data import DataLoader, SequentialSampler
    test = trainer.dm.test_loader.dataset
    if split == 'train_fixed':
        source = trainer.dm.train_loader_x.dataset
    elif split == 'test':
        source = test
    else:
        raise ValueError('Only train_fixed/test reference splits are supported')
    if not hasattr(source, 'data_source') or not hasattr(test, 'transform'):
        raise RuntimeError('Unsupported Dassl DatasetWrapper')
    transform = test.transform
    if any(s in repr(transform).lower() for s in (
            'randomcrop', 'randomresizedcrop', 'randomhorizontalflip', 'randaugment',
            'colorjitter', 'randomrotation', 'randomaffine')):
        raise RuntimeError('Reference transform is stochastic')
    copy_data = copy.copy(source)
    copy_data.transform = transform
    copy_data.is_train = False
    if hasattr(copy_data, 'k_tfm'):
        copy_data.k_tfm = 1
    return DataLoader(copy_data, batch_size=batch_size, sampler=SequentialSampler(copy_data),
                      num_workers=0, drop_last=False)


def cached_encode(model, images, classes):
    """Only deterministic frozen/deep-prompt encoders are cached; posterior draws stay native."""
    import torch.nn.functional as F
    text, visual = model.representation_learner()
    eot = model.text_encoder.forward_hidden(model.prompt_embeddings, model.tokenized_prompts, text)
    cls, rep = model.image_encoder.encode_hidden(images.type(model.dtype), visual)
    main = F.normalize(cls @ model.image_encoder.visual.proj, dim=-1)
    return dict(model=model, eot=eot, rep=rep, main=main, classes=classes)


def cached_draw(cache, n_samples):
    import torch
    import torch.nn.functional as F
    model = cache['model']
    text, _ = model.text_sample_features(cache['eot'], n_samples, use_mean=False)
    rep = F.normalize(model.image_encoder.rep_samples(cache['rep'], n_samples, use_mean=False), dim=-1)
    main = cache['main']
    text = text.type(main.dtype)
    c = cache['classes']
    return (100.0 * torch.einsum('bd,scd->sbc', main, text))[..., :c], \
           (100.0 * torch.einsum('sbd,scd->sbc', rep, text))[..., :c]


def native_moments(model, images, method_cfg):
    """Call production DAVI moments (not a reimplementation)."""
    if hasattr(model, 'forward_joint_moments'):
        return model.forward_joint_moments(images)
    from methods.det_bayesrt_mmrl.modules import DetBayesRTMMRLModel
    # Only for explicitly labeled cross-method counterfactual diagnostics.
    model.det_moment_eps = float(getattr(method_cfg, 'DET_MOMENT_EPS', 1.e-6))
    model.logit_scale = float(getattr(method_cfg, 'LOGIT_SCALE', 100.0))
    return DetBayesRTMMRLModel.forward_joint_moments(model, images)


def validate_cache_parity(model, images, classes, cache, n=3, atol=3e-3, rtol=3e-3):
    """Fail closed if the cached logits fail same-RNG native-model parity."""
    import torch
    devices = [images.device.index if images.device.index is not None else torch.cuda.current_device()] if images.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(993879)
        native = model.forward_joint(images, n, use_posterior_mean=False)
        m_ref = native['logits_main_stack'][..., :classes]
        r_ref = native['logits_rep_stack'][..., :classes]
        torch.manual_seed(993879)
        m, r = cached_draw(cache, n)
    diff = max(float((m.float()-m_ref.float()).abs().max()),
               float((r.float()-r_ref.float()).abs().max()))
    if not torch.allclose(m.float(), m_ref.float(), atol=atol, rtol=rtol) or \
       not torch.allclose(r.float(), r_ref.float(), atol=atol, rtol=rtol):
        raise RuntimeError('Cached posterior logits disagree with native forward_joint; max_abs='+str(diff))
    return diff
