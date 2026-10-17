import tempfile
import unittest
from pathlib import Path
from experiments.supplementary.core import MODES,CONFIGS,make_args,location,save_json


class NativeConfiguration(unittest.TestCase):
    def test_correct_model_and_config(self):
        self.assertEqual((MODES['davi'],CONFIGS['davi']),('DetBayesRTMMRL','det_bayesrt_mmrl_clamp.yaml'))
        self.assertEqual((MODES['branch'],CONFIGS['branch']),('BayesRTMMRL','bayesrt_mmrl.yaml'))
        self.assertEqual((MODES['original'],CONFIGS['original']),('SupplementOriginalMC','bayesrt_mmrl.yaml'))

    def test_no_implicit_mc_training_override(self):
        a=make_args('caltech101','davi',16,1,'results','native','DATASETS')
        self.assertNotIn('BAYESRT_MMRL.N_MC_TRAIN',a.opts)
        self.assertEqual(a.method,'DetBayesRTMMRL')
        b=make_args('caltech101','branch',16,1,'results','native','DATASETS')
        self.assertNotIn('BAYESRT_MMRL.DET_VAR_CLAMP',b.opts)
        self.assertIn('bayesrt_mmrl.yaml',b.method_config_file)

    def test_trajectory_saves_under_distinct_study(self):
        a=make_args('caltech101','davi',16,2,'output','native','DATASETS',trajectory=True)
        self.assertEqual(a.trainer,'SupplementTrajectoryRunner')
        self.assertIn('TRAINER.NAME',a.opts)
        self.assertNotEqual(location('output','native','caltech101','davi',16,2),
                            location('output','matched_time','caltech101','davi',16,2))

    def test_atomic_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'results.json'
            save_json(file,{'checked':True})
            self.assertEqual(file.read_text(),'{\n  "checked": true\n}')

if __name__=='__main__':unittest.main()
