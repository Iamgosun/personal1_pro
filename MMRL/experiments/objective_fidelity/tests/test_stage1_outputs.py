"""Output-location and plot-font regression tests. Values are synthetic fixtures."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))
from stage1_make_table import NEEDED, run as write_table


class Stage1OutputLayout(unittest.TestCase):
    def _create_fixture(self, outroot: Path, method: str, s_levels=(10,)):
        base = {
            'original_nll': 0.2, 'branch_nll': 0.3, 'davi_nll': 0.4,
            'mix_gap': 0.1, 'det_gap': 0.1, 'total_gap': 0.2,
            'mae_det_gap': 0.11, 'mae_total_gap': 0.21,
            'softmax_jensen_gap': 0.02, 'gaussian_error': 0.03,
            'propagation_error': 0.05, 'scale_error': 0., 'clamp_error': 0.,
            'p90_abs_det_gap': .15, 'empirical_logit_variance': .5,
            'propagated_logit_variance': .6,
            'moment_mu_mae': .01, 'moment_var_mae': .02,
            'mgf_ess_ratio_min': .9, 'clip_fraction': 0.,
        }
        self.assertEqual(set(base), NEEDED)
        dest = (outroot / 'native' / method / 'caltech101' / 'shot_16' / 'seed1'
                / 'phase1' / 'stage1_S10_R1.json')
        dest.parent.mkdir(parents=True, exist_ok=True)
        def summary(values):
            return {k: {'mean': v, 'mc_se': None} for k, v in values.items()}
        data = {
            'diagnostic_schema': 'stage1-native-3', 'mode': method,
            'profile': 'native', 'dataset': 'caltech101', 'shot': 16, 'seed': 1,
            'mc_samples': 10, 'mc_repeats': 1, 'diagnostic_partial': False,
            'cache_validated': True, 'checkpoint_sha256': method+'-sha',
            'effective_config_sha256': method+'-cfg',
            'alpha': .7, 'eta': 1., 'clamp': 50.,
            'splits': {'train_fixed': {
                'cache_parity': {'ok': True}, 'input_sha256': 'identical-inputs',
                'num_examples': 4, 'visual_pre_snr': 5.,
                'by_sample_count': {str(s): {
                    'summary': summary(base), 'repeats': [base]
                } for s in s_levels},
            }}
        }
        dest.write_text(json.dumps(data), encoding='utf-8')
        return dest

    def _run_table(self, root):
        from argparse import Namespace
        write_table(Namespace(output_root=str(root), datasets=['caltech101'],
                              seeds=[1], modes=['branch_mc', 'davi'],
                              shot=16, samples=10, mc_repeats=1,
                              split='train_fixed', out=None))
        return root / 'phase1_summary' / 'train_fixed'

    def test_output_root_contains_summary_and_original_json_is_untouched(self):
        with tempfile.TemporaryDirectory(prefix='phase 1 paths ') as tmp:
            root = Path(tmp) / 'out with spaces'
            existing = [self._create_fixture(root, mode) for mode in ('branch_mc', 'davi')]
            original_data = [path.read_bytes() for path in existing]
            summary = self._run_table(root)
            self.assertTrue((summary / 'table_a_native_train_fixed.csv').is_file())
            self.assertTrue((summary / 'table_a_native_train_fixed.md').is_file())
            self.assertTrue((summary / 'table_a_native_train_fixed_raw.csv').is_file())
            self.assertTrue((summary / 'table_a_native_train_fixed_convergence.csv').is_file())
            self.assertFalse((root.parent / 'table_a_native_train_fixed.csv').exists())
            self.assertEqual(original_data, [path.read_bytes() for path in existing])

    def test_plot_fonts_png_and_one_sample_does_not_fake_convergence(self):
        try:
            import matplotlib  # noqa
        except ImportError:
            self.skipTest('matplotlib not installed')
        from stage1_plot import run as plot
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'data'
            for mode in ('branch_mc', 'davi'):
                self._create_fixture(root, mode)
            dest = self._run_table(root)
            plot(dest/'table_a_native_train_fixed_raw.csv',
                 dest/'table_a_native_train_fixed_convergence.csv')
            figure = dest / 'figures'
            self.assertTrue((figure / 'figure_A_det_gap_vs_variance.pdf').is_file())
            self.assertTrue((figure / 'figure_A_det_gap_vs_variance.png').is_file())
            self.assertFalse((figure / 'figure_A_mc_convergence_caltech101.pdf').exists())

    def test_two_sample_sizes_create_real_convergence_chart(self):
        try:
            import matplotlib  # noqa
        except ImportError:
            self.skipTest('matplotlib not installed')
        from stage1_plot import run as plot
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'data'
            for mode in ('branch_mc', 'davi'):
                self._create_fixture(root, mode, (5, 10))
            dest = self._run_table(root)
            plot(dest/'table_a_native_train_fixed_raw.csv',
                 dest/'table_a_native_train_fixed_convergence.csv')
            self.assertTrue((dest/'figures'/'figure_A_mc_convergence_caltech101.pdf').is_file())
            self.assertTrue((dest/'figures'/'figure_A_mc_convergence_caltech101.png').is_file())


if __name__ == '__main__':
    unittest.main()
