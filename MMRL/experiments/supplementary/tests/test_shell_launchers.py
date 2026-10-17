"""Shell argument assembly smoke tests; no Dassl/GPU needed."""
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]

class ShellLaunchersTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name) / 'MMRL'
        (root / 'core').mkdir(parents=True)
        (root / 'core' / 'registry.py').write_text('# test marker\n')
        dest = root / 'experiments' / 'supplementary'
        dest.mkdir(parents=True)
        (dest / 'cli.py').write_text('# test CLI marker\n')
        for source in HERE.glob('*.sh'):
            shutil.copy2(source, dest / source.name)
        self.script = dest / 'run_study.sh'
        self.cwd = root

    def tearDown(self):
        self.temp.cleanup()

    def execute(self, stage, **kwargs):
        env = dict(os.environ, DRY_RUN='1', DATASETS='caltech101', SEEDS='1', **kwargs)
        return subprocess.run(['bash', str(self.script), stage], cwd=self.cwd,
                              env=env, text=True, capture_output=True)

    def test_native_training_preserves_method_config_mc(self):
        p = self.execute('train')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.count('cli train'), 3)
        self.assertIn('--method davi', p.stdout)
        self.assertIn('--method branch', p.stdout)
        self.assertIn('--method original', p.stdout)
        self.assertIn('--init-snapshot', p.stdout)
        self.assertIn('--trajectory', p.stdout)
        self.assertNotIn('--mc-train', p.stdout)

    def test_e1_and_e4_use_explicit_latest_native_checkpoint(self):
        e1 = self.execute('e1')
        self.assertEqual(e1.returncode, 0, e1.stderr)
        self.assertEqual(e1.stdout.count('--checkpoint latest'), 3)
        self.assertNotIn('--checkpoint best', e1.stdout)
        e4 = self.execute('e4')
        self.assertEqual(e4.returncode, 0, e4.stderr)
        self.assertNotIn('--checkpoint best', e4.stdout)
        self.assertIn('--checkpoint latest', e4.stdout)

    def test_e1_ignores_best_val_even_if_other_stages_opt_in(self):
        e1 = self.execute('e1', NATIVE_CHECKPOINT='best')
        self.assertEqual(e1.returncode, 0, e1.stderr)
        self.assertEqual(e1.stdout.count('--checkpoint latest'), 3)
        self.assertNotIn('--checkpoint best', e1.stdout)
        e2 = self.execute('e2', NATIVE_CHECKPOINT='best')
        self.assertEqual(e2.returncode, 0, e2.stderr)
        self.assertIn('--checkpoint best', e2.stdout)

    def test_e1_all_batches_allowed_for_single_seed_and_pilot_epochs(self):
        pilot = self.execute('e1', E1_BATCHES='0', EPOCHS='5')
        self.assertEqual(pilot.returncode, 0, pilot.stderr)
        self.assertIn('--e1-batches 0', pilot.stdout)
        self.assertIn('--epochs 5', pilot.stdout)
        self.assertEqual(pilot.stdout.count('--checkpoint latest'), 3)
        invalid = self.execute('e1', E1_BATCHES='-1')
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn('E1_BATCHES', invalid.stderr)

    def test_e2_e3_share_reference_path(self):
        a = self.execute('e2'); b = self.execute('e3')
        self.assertEqual(a.returncode, 0, a.stderr)
        self.assertEqual(b.returncode, 0, b.stderr)
        self.assertIn('--vectors --gradient', a.stdout)
        self.assertEqual(a.stdout, b.stdout)

    def test_e5_requires_explicit_budget(self):
        p = self.execute('e5')
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('BUDGET_SECONDS', p.stderr)
        p = self.execute('e5', BUDGET_SECONDS='300')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.count('cli budget'), 2)
        self.assertIn('--checkpoint latest', p.stdout)

    def test_help_and_wrappers(self):
        self.assertEqual(self.execute('help').returncode, 0)
        for s in ('e1','e2','e3','e4','e5'):
            script = self.script.parent / f'run_{s}.sh'
            env = dict(os.environ, DRY_RUN='1', BUDGET_SECONDS='100')
            p = subprocess.run(['bash', str(script)], env=env, cwd=self.cwd,
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, (s,p.stderr))
