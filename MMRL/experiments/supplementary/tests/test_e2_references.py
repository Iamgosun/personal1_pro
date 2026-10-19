"""E2 MC repeat pooling, uncertainty stability and independent stages."""
import unittest
from experiments.supplementary.cli import pool_mc_references, paired_mc_moment_differences


def reference(original=2.0, branch=3.0):
    return dict(original_nll=original,branch_nll=branch,gap1=branch-original,
                det_nll=4.0,gap2=4.0-branch,mean_mae=.1,var_mae=.25,
                n_images=100,s_ref=512,input_sha256='same',r_trace=.03,
                gap_identity_abs=0.0)


class E2ReferenceTests(unittest.TestCase):
    def test_means_both_references_and_recomputes_identity(self):
        result=pool_mc_references([reference(2,3),reference(2.4,3.2)],.5)
        self.assertAlmostEqual(result['original_nll'],2.2)
        self.assertAlmostEqual(result['branch_nll'],3.1)
        self.assertAlmostEqual(result['gap1'],.9)
        self.assertAlmostEqual(result['gap2'],.9)
        self.assertAlmostEqual(result['gap_identity_abs'],0)
        self.assertEqual(result['mc_repeats'],2)
        self.assertAlmostEqual(result['mc_repeat_spread']['original_nll'],.4)
        self.assertTrue(result['mc_reference_stable'])

    def test_instability_warns_without_dropping_results(self):
        result=pool_mc_references([reference(2,3),reference(3,4)],.03)
        self.assertFalse(result['mc_reference_stable'])
        self.assertTrue(result['mc_precision_warning'])
        self.assertAlmostEqual(result['original_nll'],2.5)

    def test_single_mc_repeat_is_not_assumed_stable(self):
        result=pool_mc_references([reference()],.03)
        self.assertIsNone(result['mc_reference_stable'])
        self.assertFalse(result['mc_precision_warning'])

    def test_repeated_inputs_must_match(self):
        left=reference();right=reference();right['input_sha256']='different'
        with self.assertRaises(RuntimeError):pool_mc_references([left,right],.03)

    def test_moment_repeat_differences_are_paired_per_branch(self):
        a=[dict(mu_main_mc=[1,2],var_main_mc=[3,4],mu_rep_mc=[5,6],var_rep_mc=[7,8])]
        b=[dict(mu_main_mc=[2,4],var_main_mc=[4,5],mu_rep_mc=[8,7],var_rep_mc=[11,8])]
        d=paired_mc_moment_differences(a,b)
        self.assertAlmostEqual(d['mu_main_mc_repeat_abs_delta'],1.5)
        self.assertAlmostEqual(d['var_main_mc_repeat_abs_delta'],1.)
        self.assertAlmostEqual(d['mu_rep_mc_repeat_abs_delta'],2.)
        self.assertAlmostEqual(d['var_rep_mc_repeat_abs_delta'],2.)
        with self.assertRaises(RuntimeError):paired_mc_moment_differences(a,a+a)
