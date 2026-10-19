"""E3 must render every actual run separately, including epoch and input provenance."""
import json
import tempfile
import unittest
from pathlib import Path
from experiments.supplementary.figures import make_figures

class E3Figures(unittest.TestCase):
    def test_two_batch_configurations_are_not_merged_or_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for bs, epoch, sha, stable in ((32,50,'a'*64,True),(64,40,'b'*64,False)):
                folder=(root/'native'/'caltech101'/'davi'/'shots_16'/'seed_1'/
                        'diagnostics'/'latest'/f'epoch_{epoch}'/'gradients'/
                        f'refS512_chunk8_R5_B{bs}_selstratified_seed2027_sizes1-5-10-50')
                folder.mkdir(parents=True)
                m=dict(dataset='caltech101',shot=16,seed=1,
                       checkpoint=dict(selection='latest',epoch=epoch,checkpoint_sha256=sha),
                       reference_samples=512,reference_stable=stable,
                       davi=dict(cosine=.92,relative_error=.12),
                       curve=[dict(samples=s,repeat=r,cosine=.7+.001*s+.01*r,
                                   relative_error=.25-.001*s+.004*r)
                              for s in (1,5,10,50) for r in range(5)],
                       input=dict(batch_size_requested=bs,n_unique_classes=min(bs,100),
                                  selection_seed=2027,selection_strategy='stratified',
                                  input_sha256=sha),
                       e3_parameters=dict(batch_size=bs,reference_samples=512,
                                          chunk=8,repeats=5,mc_sizes=[1,5,10,50],
                                          batch_strategy='stratified',input_seed=2027))
                (folder/'gradient.json').write_text(json.dumps(m),encoding='utf8')
            out=[Path(p) for p in make_figures(root) if Path(p).name.startswith('S3_')]
            self.assertEqual(len(out),2)
            self.assertTrue(all(p.exists() for p in out))
            self.assertNotEqual(out[0].name,out[1].name)
            self.assertTrue(any('epoch50' in p.name and 'B32' in p.name for p in out))
            self.assertTrue(any('epoch40' in p.name and 'B64' in p.name for p in out))

if __name__=='__main__':unittest.main()
