"""Stage-1 regression tests that do NOT require Dassl or CLIP checkpoint."""
import sys,unittest,math,torch
from pathlib import Path
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stage1_gap import GapAccumulator,decompose_batch,weighted_aggregate
from stage1_cached import encode_once,cached_joint_logits
from stage1_protocol import get_grid,mean_se

class Q:
    def __init__(self,mean,sigma):
        self.posterior_mean=mean
        self.sigma=sigma
    def sample_many(self,n,use_mean=False):
        w=self.posterior_mean
        return w[None].expand(n,*w.shape) if use_mean else w[None]+self.sigma*torch.randn((n,*w.shape))
class V:
    def __init__(self,w,p):self.proj=w;self.proj_rep=p
class I:
    def __init__(self,q,v):self.bayes_proj_rep=q;self.visual=v
    def encode_hidden(self,images,tokens):
        return images[:,:3],images[:,3:]
    def forward_mean(self,images,tokens):
        c,r=self.encode_hidden(images,tokens)
        return c@self.visual.proj,r@self.bayes_proj_rep.posterior_mean,c,r
    def rep_samples(self,rep_hidden,n,use_mean=False):
        q=self.bayes_proj_rep.sample_many(n,use_mean)
        return torch.einsum('bd,sdk->sbk',rep_hidden,q)
class T:
    def __init__(self,emb):self.emb=emb
    def forward_hidden(self,embedding,tokenized,rep_tokens):return self.emb
class Model:
    def __init__(self):
        torch.manual_seed(123)
        self.dtype=torch.float32
        self.prompt_embeddings=None;self.tokenized_prompts=None
        self.text_encoder=T(torch.randn(5,4))
        self.text_posterior=Q(torch.randn(4,4),.1)
        self.image_encoder=I(Q(torch.randn(3,4),.2),V(torch.randn(3,4),torch.randn(3,4)))
    def representation_learner(self):return None,None
    def text_sample_features(self,eot_hidden,num_samples,use_mean=False):
        samples=self.text_posterior.sample_many(num_samples,use_mean)
        z=F.normalize(torch.einsum('cd,sdk->sck',eot_hidden.float(),samples),dim=-1)
        return z,F.normalize(eot_hidden@self.text_posterior.posterior_mean,dim=-1)
    def forward_joint(self,images,n,use_posterior_mean=False):
        tt,tv=self.representation_learner()
        eot=self.text_encoder.forward_hidden(self.prompt_embeddings,self.tokenized_prompts,tt)
        txt,_=self.text_sample_features(eot,n,use_posterior_mean)
        c,r,_,h=self.image_encoder.forward_mean(images,tv)
        rep=F.normalize(self.image_encoder.rep_samples(h,n,use_posterior_mean),dim=-1)
        main=F.normalize(c,dim=-1);txt=txt.type(main.dtype)
        return {'logits_main_stack':100.*torch.einsum('bd,scd->sbc',main,txt),
                'logits_rep_stack':100.*torch.einsum('sbd,scd->sbc',rep,txt)}

class TestExtended(unittest.TestCase):
    def test_cached_posterior_draws_match_original_math(self):
        m=Model();x=torch.randn(7,6)
        cache=encode_once(m,x,5)
        torch.manual_seed(355)
        ref=m.forward_joint(x,13)
        torch.manual_seed(355)
        got=cached_joint_logits(cache,13)
        self.assertTrue(torch.allclose(ref['logits_main_stack'],got[0],atol=1e-5))
        self.assertTrue(torch.allclose(ref['logits_rep_stack'],got[1],atol=1e-5))
    def test_per_image_abs_does_not_cancel(self):
        s=19;z=torch.zeros((s,2,2));labels=torch.tensor([0,0])
        mu=torch.tensor([[3.,0.],[-3.,0.]])
        variance=torch.zeros_like(mu)
        result=decompose_batch([(z,z)],labels,mu,variance,mu,variance)
        agg=weighted_aggregate([(2,result)])
        self.assertLess(agg['abs_total_gap'],agg['mae_total_gap'])
        self.assertAlmostEqual(agg['mae_total_gap'],sum(result['_abs_vec'])/2)
        self.assertEqual(agg['underestimate_fraction'],.5)
        self.assertAlmostEqual(agg['p90_abs_total_gap'],float(torch.quantile(torch.tensor(result['_abs_vec']),.9)),places=5)
    def test_streaming_prefix_same_as_direct(self):
        torch.manual_seed(44);m=torch.randn(30,3,4);r=torch.randn(30,3,4)
        y=torch.tensor([0,1,2]);pm=torch.zeros(3,4);pv=torch.ones(3,4)
        acc=GapAccumulator(y,pm,pv,pm,pv)
        acc.add(m[:9],r[:9]);snap9=acc.snapshot()
        acc.add(m[9:],r[9:]);snap30=acc.snapshot()
        direct9=decompose_batch([(m[:9],r[:9])],y,pm,pv,pm,pv)
        direct30=decompose_batch([(m,r)],y,pm,pv,pm,pv)
        for key in ('total_gap','mix_gap','softmax_jensen_gap'):
            self.assertAlmostEqual(snap9[key],direct9[key],places=10)
            self.assertAlmostEqual(snap30[key],direct30[key],places=10)
    def test_grid_requires_last_sample(self):
        self.assertEqual(get_grid('auto',128),[64,128])
        self.assertRaises(ValueError,get_grid,'16,32',128)
        self.assertEqual(get_grid('16,64,128',128),[16,64,128])
    def test_mc_summary(self):
        a=mean_se([{'x':1.},{'x':2.},{'x':3.}])['x']
        self.assertAlmostEqual(a['mean'],2.)
        self.assertAlmostEqual(a['mc_se'],1/math.sqrt(3))

if __name__=='__main__':unittest.main()
