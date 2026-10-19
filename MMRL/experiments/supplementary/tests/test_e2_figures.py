"""E2 Figure S2 smoke with paired Main/Rep synthetic reference vectors."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from experiments.supplementary.figures import make_figures


class E2Figures(unittest.TestCase):
    def test_e2_scatter_handles_main_and_rep_separately(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            sub=root/'native/caltech101/davi/shots_16/seed_1/diagnostics/latest/epoch_5/S32_R2/test'
            sub.mkdir(parents=True)
            meta={'dataset':'caltech101','method':'davi','shot':16,'seed':1,
                  'checkpoint':{'selection':'latest','epoch':5},'partial':False,
                  's_ref':32,'mc_repeats':2,'mc_precision_warning':True}
            (sub/'metrics.json').write_text(json.dumps(meta),encoding='utf-8')
            vectors=[{'mu_main_mc':[1.0,2.0],'var_main_mc':[.2,.5],
                      'mu_main_det':[1.1,2.1],'var_main_det':[.3,.4],
                      'mu_rep_mc':[2.1,3.1],'var_rep_mc':[1.2,2.2],
                      'mu_rep_det':[2.0,3.0],'var_rep_det':[1.3,2.1]}]
            (sub/'moment_vectors.json').write_text(json.dumps(vectors),encoding='utf-8')
            with (sub/'per_image.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['main_sampled_var','main_abs_det_gap',
                                               'rep_sampled_var','rep_abs_det_gap'])
                w.writeheader()
                for i in range(5):
                    w.writerow({'main_sampled_var':.1+i*.1,
                                'main_abs_det_gap':.02+i*.01,
                                'rep_sampled_var':.8+i*.2,
                                'rep_abs_det_gap':.05+i*.01})
            files=make_figures(root)
            matches=[Path(x) for x in files if '/S2_' in x]
            self.assertEqual(len(matches),1)
            self.assertIn('epoch5',matches[0].name)
            self.assertTrue(matches[0].is_file())
            self.assertGreater(matches[0].stat().st_size,1000)
