"""Confirm CLI accepts native-configured datasets beyond the original V7 shortlist."""
import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from experiments.supplementary import cli, core

class NativeDatasetTest(unittest.TestCase):
    def test_dataset_validation_uses_configuration_file_not_fixed_choices(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = root / 'configs' / 'datasets'
            config.mkdir(parents=True)
            (config / 'oxford_pets.yaml').write_text('DATASET:\n  NAME: OxfordPets\n')
            with patch.object(core, 'ROOT', root):
                self.assertEqual(cli.native_dataset('oxford_pets'), 'oxford_pets')
                with self.assertRaisesRegex(argparse.ArgumentTypeError, 'not found'):
                    cli.native_dataset('fake_dataset')
                with self.assertRaisesRegex(argparse.ArgumentTypeError, 'Invalid'):
                    cli.native_dataset('../oxford_pets')

if __name__ == '__main__': unittest.main()
