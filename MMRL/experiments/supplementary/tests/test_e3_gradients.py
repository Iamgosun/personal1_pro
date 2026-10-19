"""E3 input selection, fixed-posterior draw gradient parity, and provenance."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from experiments.supplementary.cli import e3_output_tag, select_e3_fixed_batch
from experiments.supplementary.diagnostics import (
    weighted_mc_gradient, check_mc_chunk_gradient_parity)


class _Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data_source = [SimpleNamespace(label=i // 6) for i in range(60)]
        self.transform = lambda x: x
        self.is_train = True
        self.k_tfm = 1

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, i):
        return {'img': torch.tensor([float(i), float(i*i)], dtype=torch.float32),
                'label': self.data_source[i].label}


class _Posterior(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer('mean', torch.eye(dim))

    def sample_many(self, n):
        return self.mean.unsqueeze(0) + .03 * torch.randn(n, *self.mean.shape)


class _RepLearner(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.compound_rep_tokens = nn.Parameter(torch.randn(dim)*.07)

    def forward(self):
        return [self.compound_rep_tokens], [self.compound_rep_tokens]


class _Text(nn.Module):
    def forward_hidden(self, prompts, tokens, compound):
        return prompts + compound[0]


class _Visual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer('proj', torch.eye(dim))


class _Image(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.visual = _Visual(dim)
        self.bayes_proj_rep = _Posterior(dim)

    def encode_hidden(self, image, compound):
        return image + compound[0], image + 2 * compound[0]


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dtype = torch.float32
        self.representation_learner = _RepLearner(3)
        self.text_encoder = _Text()
        self.image_encoder = _Image(3)
        self.text_posterior = _Posterior(3)
        self.register_buffer('prompt_embeddings', torch.tensor([[.9,.1,.1], [.1,.8,.2], [.2,.1,.7]]))
        self.register_buffer('tokenized_prompts', torch.zeros(3,dtype=torch.long))


class E3GradientTests(unittest.TestCase):
    def test_weighted_microchunk_is_equal_even_for_partial_last_chunk(self):
        torch.manual_seed(5)
        values = torch.randn(7, 3)
        param = nn.Parameter(torch.tensor([.2, -.3, .8]))

        def one(start, count):
            loss = (values[start:start+count] * param).square().sum(dim=1).mean()
            return torch.autograd.grad(loss, [param])[0].detach()
        full = one(0,7)
        self.assertTrue(torch.allclose(weighted_mc_gradient(7,3,one),full,atol=1e-6))
        self.assertTrue(torch.allclose(weighted_mc_gradient(7,8,one),full,atol=1e-6))
        with self.assertRaises(ValueError): weighted_mc_gradient(0,2,one)

    def test_true_model_graph_parity_fixed_weights(self):
        torch.manual_seed(41)
        model = _ToyModel()
        trainer = SimpleNamespace(method=SimpleNamespace(model=model,device=torch.device('cpu'),
                 num_classes=3,cfg=SimpleNamespace(BAYESRT_MMRL=SimpleNamespace(ALPHA=.7))))
        batch={'img':torch.tensor([[.5,-.1,.4],[.4,.5,.2],[.2,.3,.6]]),
               'label':torch.tensor([0,1,2])}
        params=list(model.representation_learner.parameters())
        for count in (1,2,3):
            result=check_mc_chunk_gradient_parity(trainer,batch,params,n_samples=5,chunk=count,
                                                   atol=1.e-4,rtol=1.e-4)
            self.assertTrue(result['checked'])
            self.assertLess(result['relative_error'],1.e-5)

    def test_stratified_reproducible_and_class_diverse(self):
        dataset=_Dataset()
        trainer=SimpleNamespace(dm=SimpleNamespace(
            test_loader=SimpleNamespace(dataset=dataset),
            train_loader_x=SimpleNamespace(dataset=dataset)))
        a, am=select_e3_fixed_batch(trainer,10,strategy='stratified',selection_seed=23)
        b, bm=select_e3_fixed_batch(trainer,10,strategy='stratified',selection_seed=23)
        _, cm=select_e3_fixed_batch(trainer,10,strategy='stratified',selection_seed=24)
        self.assertEqual(am['n_unique_classes'],10)
        self.assertEqual(am['input_sha256'],bm['input_sha256'])
        self.assertNotEqual(am['selected_indices_sha256'],cm['selected_indices_sha256'])
        self.assertTrue(torch.equal(a['label'],b['label']))
        _, old=select_e3_fixed_batch(trainer,10,strategy='first',selection_seed=23)
        self.assertEqual(old['n_unique_classes'],2)
        self.assertEqual(old['n_images'],10)

    def test_different_diagnostic_controls_have_distinct_paths(self):
        def tag(**kw):
            cfg=dict(ref=512,chunk=8,repeats=5,batch_size=32,
                     strategy='stratified',seed=2027,sizes=[1,5,10,50]);cfg.update(kw)
            return e3_output_tag(**cfg)
        base=tag()
        for changed in (dict(batch_size=64),dict(strategy='first'),
                        dict(seed=2028),dict(sizes=[1,5,10]),dict(ref=256),
                        dict(chunk=16),dict(repeats=3)):
            self.assertNotEqual(base,tag(**changed))

if __name__=='__main__': unittest.main()
