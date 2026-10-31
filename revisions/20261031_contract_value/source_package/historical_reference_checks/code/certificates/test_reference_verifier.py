"""Small synthetic regression tests, NOT annual experimental validations."""
import unittest
from fractions import Fraction as F
from reference_verifier import (Q, floor_to, ceil_to, fixed, finite_box_bound,
    check_primal, verify_facets, exact_piecewise_minimum, aggregate_certificates)


class ExactCertificateTests(unittest.TestCase):
    def test_equality_multiplier_and_nonzero_lower_bounds(self):
        # min 2*x + y, x+y=3, 1<=x<=2, 0<=y<=3 has optimum 4.
        m = {"objective":["2","1"], "lower":["1","0"], "upper":["2","3"],
             "equalities":[{"coefficients":[[0,"1"],[1,"1"]],"rhs":"3","multiplier":"1"}]}
        self.assertEqual(finite_box_bound(m)["corrected_bound"], 4)
        self.assertTrue(check_primal(m, ["1","2"])["exactly_feasible"])

    def test_inequality_orientation(self):
        # min x s.t. -x<=-1, 0<=x<=2: mu=1 gives exact LB 1.
        m = {"objective":["1"], "lower":["0"], "upper":["2"],
             "inequalities_le":[{"coefficients":[[0,"-1"]],"rhs":"-1","multiplier":"1"}]}
        self.assertEqual(finite_box_bound(m)["corrected_bound"], 1)

    def test_sign_clipping_recomputes_entire_bound(self):
        # Invalid negative mu could give 2 although the optimum is 0.
        m = {"objective":["2"], "lower":["0"], "upper":["1"],
             "inequalities_le":[{"coefficients":[[0,"1"]],"rhs":"1","multiplier":"-2"}]}
        r = finite_box_bound(m)
        self.assertEqual(r["clipped_inequality_indices"], [0])
        self.assertEqual(r["corrected_bound"], 0)
        with self.assertRaises(ValueError): finite_box_bound(m, False)

    def test_nonstationary_multiplier_needs_box_correction(self):
        # lambda=2 gives uncorrected constant 2 > optimum 1; correction is -2.
        m = {"objective":["1"], "lower":["0"], "upper":["2"],
             "equalities":[{"coefficients":[[0,"1"]],"rhs":"1","multiplier":"2"}]}
        r = finite_box_bound(m)
        self.assertEqual(r["box_correction"], -2)
        self.assertEqual(r["corrected_bound"], 0)

    def test_negative_price_facet(self):
        # -10*(x-1)+ over x in [0,2] is nonconvex; its chord -5*x is a minorant.
        pts = [["0","0"],["1","0"],["2","-10"]]
        self.assertTrue(verify_facets(pts,[{"slope":"-5","intercept":"0"}])["valid_minorants"])
        self.assertFalse(verify_facets(pts,[{"slope":"0","intercept":"0"}])["valid_minorants"])
        self.assertEqual(exact_piecewise_minimum(pts), (F(-10), F(2)))

    def test_every_breakpoint_is_needed(self):
        # Facet 0 passes endpoints but misses an interior dip.
        pts = [["0","0"],["1","-1"],["2","0"]]
        r = verify_facets(pts,[{"slope":"0","intercept":"0"}])
        self.assertFalse(r["valid_minorants"])
        self.assertEqual(r["violations"][0]["breakpoint"], 1)

    def test_exact_primal_check_does_not_treat_tolerance_as_repair(self):
        m = {"objective":["1"], "lower":["0"], "upper":["1"],
             "equalities":[{"coefficients":[[0,"1"]],"rhs":"1","multiplier":"0"}]}
        r = check_primal(m,["0.999999999999"])
        self.assertFalse(r["exactly_feasible"])
        self.assertEqual(r["max_equality_residual"], F(1,10**12))

    def test_exact_binary64_and_decimal_are_distinct(self):
        self.assertEqual(Q({"binary64_hex": (0.1).hex()}), F.from_float(0.1))
        self.assertNotEqual(Q({"binary64_hex": (0.1).hex()}), Q("0.1"))

    def test_signed_outward_rounding(self):
        self.assertEqual(fixed(floor_to("1.239")), "1.23")
        self.assertEqual(fixed(ceil_to("1.231")), "1.24")
        self.assertEqual(fixed(floor_to("-1.231")), "-1.24")
        self.assertEqual(fixed(ceil_to("-1.239")), "-1.23")

    def test_published_scalar_arithmetic_and_rounding(self):
        # These are supplied manuscript scalars, not reconstructed certificates.
        r = aggregate_certificates("555168.2050556970", ["454956"],
            common_cost="2409.7842143056", witness_saving="117000.2387599464")
        self.assertEqual(r["savings_ceiling"], Q("97802.4208413914"))
        self.assertEqual(r["savings_ceiling_cents"], "97802.43")
        self.assertEqual(r["witness_saving_lower_cents"], "117000.23")
        self.assertEqual(r["premium_lower"], Q("19197.8179185550"))
        self.assertEqual(r["premium_lower_cents"], "19197.81")

    def test_witness_loss_reduces_guaranteed_premium(self):
        r = aggregate_certificates("555168.2050556970", ["454956"],
            common_cost="2409.7842143056", witness_saving="117000.2387599464",
            witness_loss_upper="0.008")
        self.assertEqual(r["premium_lower_cents"], "19197.80")

    def test_allowance_must_have_correct_sign(self):
        m={"objective":["1"],"lower":["0"],"upper":["1"],"certified_allowance":"-1"}
        with self.assertRaises(ValueError): finite_box_bound(m)


if __name__ == "__main__": unittest.main(verbosity=2)
