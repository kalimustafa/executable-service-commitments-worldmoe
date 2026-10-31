"""Synthetic tiny-contract tests; no annual experiment inputs are implied."""
import unittest
from fractions import Fraction as F
from annual_dual_revaluation import revalue_zone, original_slot_breakpoints


class OriginalCostDualTests(unittest.TestCase):
    def test_two_slot_contract_has_known_optimum(self):
        # One unit can shift from price 10 to 1; two deviation fees total 0.5.
        result=revalue_zone(arrivals=[1,0],rt=[10,1],pue=[1,1],pv=[0,0],
            idle=0,power=1,reference_grid=[1,0],grid_limit=1,pue_max=1,
            terminal_rt=1,multipliers=[2,2],window_slots=2)
        self.assertEqual(result["cost_lower_bound"], F(3,4))
        self.assertEqual(result["arithmetic_loss_strict_upper"], F(6,10**12))

    def test_one_slot_contract_forces_immediate_execution(self):
        result=revalue_zone(arrivals=[1,0],rt=[10,1],pue=[1,1],pv=[0,0],
            idle=0,power=1,reference_grid=[1,0],grid_limit=1,pue_max=1,
            terminal_rt=1,multipliers=[3,0],window_slots=1)
        self.assertEqual(result["cost_lower_bound"], F(5,2))

    def test_negative_price_original_cost_minimum(self):
        # The cost is nonconvex at the PV/import breakpoint. No sign shortcut.
        result=revalue_zone(arrivals=[2],rt=[-10],pue=[1],pv=[1],
            idle=0,power=1,reference_grid=[0],grid_limit=2,pue_max=1,
            terminal_rt=1,multipliers=[F(-5,4)],window_slots=1,compute_limit=2)
        self.assertEqual(result["cost_lower_bound"], F(-9,4))

    def test_import_and_reference_breakpoints_are_both_retained(self):
        pts=original_slot_breakpoints(2,1,1,0,1,F(1,2),3,compute_limit=3)
        self.assertEqual([x for x,_ in pts],[F(0),F(1),F(3,2),F(3)])

    def test_infeasible_idle_load_does_not_hide_behind_zero_cap(self):
        with self.assertRaises(ValueError):
            original_slot_breakpoints(1,1,0,2,1,2,1)


if __name__ == "__main__": unittest.main(verbosity=2)
