"""E1 must generate a responsibility-only plot when no validation loader existed."""
import csv
import tempfile
import unittest
from pathlib import Path


class E1FigureTest(unittest.TestCase):
    def test_no_validation_single_seed_sparse_epoch(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest('matplotlib unavailable')
        from experiments.supplementary.figures import make_figures
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            folder=root/'native'/'caltech101'/'original'/'shots_16'/'seed_1'/'trace'
            folder.mkdir(parents=True)
            with (folder/'trajectory.csv').open('w',encoding='utf-8',newline='') as f:
                fields=['dataset','shot','seed','method','epoch','rho_rep_median',
                        'rho_rep_p10','rho_rep_p90','branch_weight','rep_val_acc']
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
                for epoch in (0,2,5):
                    w.writerow(dict(dataset='caltech101',shot=16,seed=1,method='original',
                                    epoch=epoch,rho_rep_median=.22,rho_rep_p10=.1,
                                    rho_rep_p90=.4,branch_weight=.3,rep_val_acc=''))
            paths=make_figures(root)
            self.assertEqual(len(paths),1)
            self.assertTrue((root/'figures'/'S1_caltech101_shot16.png').is_file())

    def test_partial_val_does_not_drop_all_epochs(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest('matplotlib unavailable')
        from experiments.supplementary.figures import make_figures
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for method in ('original','branch'):
                folder=root/'native'/'eurosat'/method/'shots_16'/'seed_1'/'trace'
                folder.mkdir(parents=True)
                with (folder/'trajectory.csv').open('w',encoding='utf-8',newline='') as f:
                    fields=['dataset','shot','seed','method','epoch','rho_rep_median',
                            'rho_rep_p10','rho_rep_p90','branch_weight','rep_val_acc']
                    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
                    for epoch in (0,2,5):
                        w.writerow(dict(dataset='eurosat',shot=16,seed=1,method=method,
                            epoch=epoch,rho_rep_median=.2,rho_rep_p10=.1,rho_rep_p90=.3,
                            branch_weight=.3,rep_val_acc=80. if epoch!=2 else ''))
            paths=make_figures(root)
            self.assertEqual(len(paths),1)
            self.assertTrue((root/'figures'/'S1_eurosat_shot16.png').is_file())
