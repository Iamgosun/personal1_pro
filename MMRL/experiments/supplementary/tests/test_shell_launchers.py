"""Shell argument assembly smoke tests; no Dassl/GPU needed."""
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]

class ShellLaunchersTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name) / 'MMRL'
        (root / 'core').mkdir(parents=True)
        (root / 'core' / 'registry.py').write_text('# test marker\n')
        cfg = root / 'configs' / 'datasets'
        cfg.mkdir(parents=True)
        for dataset in ('caltech101', 'eurosat', 'fgvc_aircraft', 'oxford_pets'):
            (cfg / f'{dataset}.yaml').write_text('DATASET:\n  NAME: TestSet\n')
        dest = root / 'experiments' / 'supplementary'
        dest.mkdir(parents=True)
        (dest / 'cli.py').write_text('# test CLI marker\n')
        for source in HERE.glob('*.sh'):
            shutil.copy2(source, dest / source.name)
        self.script = dest / 'run_study.sh'
        self.cwd = root

    def tearDown(self):
        self.temp.cleanup()

    # The real caller is allowed to export DATASETS=stanford_cars (or any
    # other native dataset).  Unit-test launchers use a deliberately minimal
    # temporary MMRL tree and must not inherit those experiment settings.
    _LAUNCHER_ENV_KEYS = (
        'GPU', 'DATASETS', 'SEEDS', 'SHOTS', 'OUT', 'DATA_ROOT',
        'EPOCHS', 'MC_TRAIN', 'SAMPLES', 'CHUNK', 'MC_REPEATS',
        'MC_TOLERANCE', 'E2_SPLITS', 'GRAD_REF', 'GRAD_CHUNK',
        'GRAD_REPEATS', 'GRAD_SIZES', 'GRAD_BATCH_STRATEGY',
        'GRAD_INPUT_SEED', 'GRAD_PARITY', 'E3_EPOCHS',
        'E2_GAP_EPOCHS','E3_NOISE_EPOCHS','E3_NOISE_DRAWS','E3_NOISE_SEED',
        'E4_EVOLUTION','E4_EVOLUTION_EPOCHS','E4_EVOLUTION_IMAGES',
        'E4_EVOLUTION_DRAWS','E4_EVOLUTION_CHUNK','E4_EVOLUTION_BATCH',
        'E4_EVOLUTION_INPUT_SEED',
        'E1_SAMPLES', 'E1_BATCHES', 'NATIVE_CHECKPOINT',
        'BATCH_SIZE', 'BUDGET_SECONDS', 'SKIP_EXISTING',
        'FORCE_EVAL', 'DRY_RUN',
    )

    def shell_env(self, **kwargs):
        env = dict(os.environ)
        for key in self._LAUNCHER_ENV_KEYS:
            env.pop(key, None)
        env.update(DRY_RUN='1', DATASETS='caltech101', SEEDS='1')
        env.update(kwargs)
        return env

    def execute(self, stage, **kwargs):
        return subprocess.run(['bash', str(self.script), stage], cwd=self.cwd,
                              env=self.shell_env(**kwargs), text=True,
                              capture_output=True)

    def test_native_configured_dataset_without_hardcoded_whitelist(self):
        p = self.execute('all', DATASETS='oxford_pets', BUDGET_SECONDS='300')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('--dataset oxford_pets', p.stdout)
        self.assertNotIn('Unsupported dataset', p.stderr)
        self.assertIn('--mode gradient', p.stdout)
        self.assertIn('cli budget', p.stdout)

    def test_dataset_must_have_native_config(self):
        p = self.execute('train', DATASETS='made_up_dataset')
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('Missing native dataset config', p.stderr)
        p = self.execute('train', DATASETS='../oxford_pets')
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('Invalid dataset config basename', p.stderr)

    def test_native_training_preserves_method_config_mc(self):
        p = self.execute('train')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.count('cli train'), 3)
        self.assertIn('--method davi', p.stdout)
        self.assertIn('--method branch', p.stdout)
        self.assertIn('--method original', p.stdout)
        self.assertEqual(p.stdout.count('--trajectory'),3)
        self.assertNotIn('--init-snapshot', p.stdout)
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

    def test_e2_e3_run_independently(self):
        e2 = self.execute('e2'); e3 = self.execute('e3')
        self.assertEqual(e2.returncode, 0, e2.stderr)
        self.assertEqual(e3.returncode, 0, e3.stderr)
        self.assertIn('--mode reference',e2.stdout)
        self.assertIn('--vectors',e2.stdout)
        self.assertNotIn('--mode gradient',e2.stdout)
        self.assertNotIn('--grad-ref',e2.stdout)
        self.assertIn('--mode gradient',e3.stdout)
        self.assertIn('--grad-ref 512',e3.stdout)
        self.assertNotIn('--mode reference',e3.stdout)
        self.assertNotIn('--samples 1024',e3.stdout)
        self.assertNotEqual(e2.stdout,e3.stdout)

    def test_e3_batch_and_input_controls_are_forwarded_independently(self):
        base = self.execute('e3')
        changed = self.execute('e3', BATCH_SIZE='64', GRAD_INPUT_SEED='99',
                               GRAD_BATCH_STRATEGY='first', GRAD_SIZES='1 3 9',
                               GRAD_PARITY='0')
        self.assertEqual(changed.returncode,0,changed.stderr)
        self.assertIn('--batch-size 64',changed.stdout)
        self.assertIn('--grad-input-seed 99',changed.stdout)
        self.assertIn('--grad-batch-strategy first',changed.stdout)
        self.assertIn('--grad-sizes 1 3 9',changed.stdout)
        self.assertIn('--skip-grad-parity',changed.stdout)
        self.assertNotEqual(base.stdout,changed.stdout)
        self.assertNotIn('--mode reference',changed.stdout)

    def test_e3_multiple_epochs_from_full_trajectory(self):
        result=self.execute('e3',E3_EPOCHS='5 25 latest')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertNotIn('--e3-snapshot-epochs',result.stdout)
        self.assertIn('--trajectory',next(line for line in result.stdout.splitlines() if 'cli train' in line and '--method davi' in line))
        self.assertEqual(result.stdout.count('--mode gradient --'),3)
        self.assertIn('--checkpoint trajectory',result.stdout)
        self.assertIn('--epoch 5',result.stdout)
        self.assertIn('--epoch 25',result.stdout)
        self.assertIn('--checkpoint latest',result.stdout)
        self.assertNotIn('--mode reference',result.stdout)
        p=self.execute('e3',E3_EPOCHS='5 5 latest')
        self.assertNotEqual(p.returncode,0)
        self.assertIn('duplicate',p.stderr)
        p=self.execute('e3',E3_EPOCHS='0 latest')
        self.assertNotEqual(p.returncode,0)
        self.assertIn('E3_EPOCHS',p.stderr)

    def test_all_stages_train_all_three_methods_with_full_trajectory(self):
        for stage,opts in (('train',{}),('e1',{}),('e2',{}),('e3',{'E3_EPOCHS':'5 25 latest'}),
                           ('all_no_e5',{}),('all',{'BUDGET_SECONDS':'300'})):
            result=self.execute(stage,**opts)
            self.assertEqual(result.returncode,0,(stage,result.stderr))
            # An `all` dry-run intentionally prints E4-only fallback training
            # as well, because DRY_RUN does not create preceding E1 directories.
            # The actual first native training pass must be a full trajectory.
            training_lines=[line for line in result.stdout.splitlines() if 'cli train' in line]
            expected=1 if stage=='e2' else 2 if stage=='e3' else 3
            self.assertGreaterEqual(len(training_lines),expected,(stage,training_lines))
            for line in training_lines[:expected]:
                self.assertIn('--trajectory',line,(stage,line))
                self.assertNotIn('--init-snapshot',line,(stage,line))
                self.assertNotIn('--e3-snapshot-epochs',line,(stage,line))

    def test_r13_three_figure_dry_run_arguments(self):
        e2=self.execute('e2',E2_GAP_EPOCHS='5 25 latest')
        self.assertEqual(e2.returncode,0,e2.stderr)
        self.assertIn('--checkpoint trajectory',e2.stdout)
        self.assertIn('--epoch 5',e2.stdout)
        self.assertIn('--epoch 25',e2.stdout)
        self.assertIn('--splits train_fixed',e2.stdout)
        e3=self.execute('e3',E3_NOISE_EPOCHS='5 25 latest')
        self.assertEqual(e3.returncode,0,e3.stderr)
        self.assertEqual(e3.stdout.count('--mode gradient_noise'),3)
        self.assertIn('--noise-draws 10',e3.stdout)
        e5=self.execute('e5',BUDGET_SECONDS='300',E5_CURVE='1')
        self.assertEqual(e5.returncode,0,e5.stderr)
        self.assertEqual(e5.stdout.count('--time-curve'),2)
        self.assertEqual(e5.stdout.count('--checkpoint time_curve'),10)
        self.assertEqual(e5.stdout.count('--epoch 0'),2)
        self.assertEqual(e5.stdout.count('--epoch 100'),2)

    def test_e4_new_train_trajectory_when_evolution_enabled(self):
        result=self.execute('e4')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout.count('cli train'),3)
        self.assertEqual(result.stdout.count('--trajectory'),3)
        self.assertEqual(result.stdout.count('--mode reference'),3)
        self.assertEqual(result.stdout.count('--mode formal'),3)
        self.assertEqual(result.stdout.count('--mode e4_evolution'),18)
        self.assertIn('--epoch 0',result.stdout)
        self.assertIn('--epoch 38',result.stdout)
        self.assertIn('--e4-images 128 --e4-draws 64',result.stdout)

    def test_e4_can_disable_evolution_and_keep_legacy_minimal_snapshots(self):
        result=self.execute('e4',E4_EVOLUTION='0')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout.count('--init-snapshot'),3)
        self.assertNotIn('--trajectory',result.stdout)
        self.assertNotIn('--mode e4_evolution',result.stdout)

    def test_r15_e2_e3_default_epochs_are_dense_but_configurable(self):
        e2=self.execute('e2')
        self.assertEqual(e2.returncode,0,e2.stderr)
        self.assertEqual(e2.stdout.count('--splits train_fixed'),11)
        e3=self.execute('e3')
        self.assertEqual(e3.returncode,0,e3.stderr)
        self.assertEqual(e3.stdout.count('--mode gradient_noise'),5)

    def test_e4_does_not_implicitly_run_e3(self):
        result=self.execute('e4')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--mode reference',result.stdout)
        self.assertNotIn('--mode gradient',result.stdout)

    def test_e5_requires_explicit_budget(self):
        p = self.execute('e5')
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('BUDGET_SECONDS', p.stderr)
        p = self.execute('e5', BUDGET_SECONDS='300')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.count('cli budget'), 2)
        self.assertIn('--checkpoint latest', p.stdout)
        # Both reference splits must be evaluated on identical budget endpoints.
        self.assertEqual(p.stdout.count('--splits train_fixed test'), 2)
        self.assertEqual(p.stdout.count('--mode formal'), 2)

    def test_help_and_wrappers(self):
        # Regression: caller's real DATASETS may include native datasets that
        # are intentionally absent from this miniature fixture tree.
        with patch.dict(os.environ, {
            'DATASETS': 'stanford_cars', 'SEEDS': '2 3',
            'E3_EPOCHS': '5 25 latest', 'BATCH_SIZE': '64',
        }):
            self.assertEqual(self.execute('help').returncode, 0)
            for s in ('e1','e2','e3','e4','e5'):
                script = self.script.parent / f'run_{s}.sh'
                p = subprocess.run(['bash', str(script)],
                                   env=self.shell_env(BUDGET_SECONDS='100'),
                                   cwd=self.cwd, capture_output=True, text=True)
                self.assertEqual(p.returncode, 0, (s,p.stderr))
                self.assertIn('--dataset caltech101', p.stdout)
                self.assertNotIn('--dataset stanford_cars', p.stdout)
