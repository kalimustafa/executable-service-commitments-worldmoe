"""Small exact regressions for new formulae; these are not annual experiments."""
import itertools
import unittest
from fractions import Fraction as F
import numpy as np

from envelope_certificate import (reference_envelopes,original_points,exact_dual,
                                  exact_clamp,ceil_cents,floor_cents)
from epigraph_certificate import exact_facets,exact_box_revalue


class EnvelopeRegression(unittest.TestCase):
    def setUp(self):
        self.ref=[F(1),F(1),F(1,2),F(0)]
        _,self.R,self.nextclear,self.E,self.Amin,self.Amax,self.runs=reference_envelopes(self.ref)

    def test_every_compatible_small_arrival_has_admission_envelope(self):
        count=0
        for a in itertools.product([F(i,4) for i in range(11)],repeat=4):
            queue=F(0);book=[];A=F(0)
            for v in a:
                queue+=v;b=min(queue,F(1));queue-=b;book.append(b)
            if book!=self.ref or queue:continue
            count+=1
            for t,v in enumerate(a,1):
                A+=v
                self.assertLessEqual(self.Amin[t],A)
                self.assertLessEqual(A,self.Amax[t])
        self.assertGreater(count,1)

    def test_explicit_reference_nonidentification(self):
        a=[F(1),F(1),F(1,2),F(0)]
        b=[F(3,2),F(1,2),F(1,2),F(0)]
        def execute(arr):
            q=F(0);out=[]
            for v in arr:q+=v;s=min(F(1),q);q-=s;out.append(s)
            return out
        self.assertNotEqual(a,b)
        self.assertEqual(execute(a),execute(b))

    def test_exact_robust_witness_clamp_with_terminal_endpoint(self):
        D=3
        L=[F(0)]+[self.R[self.nextclear[max(0,t-D+1)]] for t in range(1,5)]
        U=list(self.R);L[-1]=U[-1]
        x,report=exact_clamp([.4,.9,1.01,.2],L,U,[F(1)]*4)
        self.assertTrue(report['encoded_corridor_exactly_feasible'])
        self.assertEqual(sum(x),sum(self.ref))
        self.assertEqual(sum(x[:3]),F(5,2))

    def test_negative_price_cost_and_exact_facets(self):
        pts,_,_,_=original_points(-7,1,F(1,2),0,1,F(1,4),2)
        facets=exact_facets(pts)
        self.assertTrue(any(cost<0 for _,cost in pts))
        self.assertTrue(all(alpha*x+beta<=cost for alpha,beta in facets for x,cost in pts))
        self.assertIn(F(1,2),[x for x,_ in pts])

    def test_both_duals_below_enumerated_original_primal(self):
        prices=[-7,2,5,-1]
        pts=[original_points(p,1,F(1,2),0,1,F(1,4),2)[0] for p in prices]
        costs=[original_points(p,1,F(1,2),0,1,F(1,4),2)[3] for p in prices]
        facets=[exact_facets(p) for p in pts]
        D=3;L=[F(0)]+[self.Amin[max(0,t-D+1)] for t in range(1,5)];U=self.Amax
        values=[]
        for xs in itertools.product([F(i,4) for i in range(5)],repeat=4):
            S=F(0);ok=True
            for t,x in enumerate(xs,1):
                S+=x
                if not L[t]<=S<=U[t]:ok=False;break
            if ok:values.append(sum(fn(x) for fn,x in zip(costs,xs)))
        self.assertTrue(values);minimum=min(values)
        for lam in itertools.product([-2,0,3],repeat=4):
            direct,_=exact_dual(pts,L,U,lam)
            # Includes deliberately invalid raw inequality signs; exact checker
            # clips and recomputes the entire finite-box expression.
            mu=[F((-1)**j,j+1) for j in range(sum(map(len,facets)))]
            box,checks=exact_box_revalue(facets,pts,L,U,lam,mu)
            self.assertLessEqual(direct,minimum)
            self.assertLessEqual(box,minimum)
            self.assertGreater(checks['clipped_negative_inequality_multipliers'],0)

    def test_zero_initial_and_open_final_reference_are_guarded(self):
        self.assertEqual(self.Amin[0],0)
        self.assertEqual(self.Amax[0],0)
        with self.assertRaises(ValueError):reference_envelopes([0,1])

    def test_negative_outward_currency_format(self):
        self.assertEqual(floor_cents(F('-3506.530785')),'-3506.54')
        self.assertEqual(ceil_cents(F('-3506.530785')),'-3506.53')
        self.assertEqual(floor_cents(F('0.009')),'0.00')
        self.assertEqual(ceil_cents(F('-0.009')),'0.00')


if __name__=='__main__':unittest.main(verbosity=2)
