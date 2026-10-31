Mathematical audit utilities

These utilities add no new WorldMoE experimental results. They check contract
boundary logic and supply an exact repair algorithm for future verification of
the original annual witness when its arrays are supplied.

exact_witness_repair.py uses Python's standard-library Fraction arithmetic.
Run:
    python exact_witness_repair.py --self-test --output repair_self_test.json
    python exact_witness_repair.py --input repair_example.json --output example_result.json

repair_example.json is an explicitly constructed four-slot algorithm fixture,
not a research dataset or the annual 24 h hindsight witness. To verify actual
data, supply arrivals, candidate execution, capacity, deadline_slots and the
original terminal rule for one site. Arrays must cover the intended recovery
horizon. Decimal/rational strings are read exactly; floats are interpreted as
their exact binary values. Existing initial work is supplied as initial_cohorts,
each with remaining work and its original exclusive deadline boundary. See the
function documentation for its homogeneous-contract restriction.

The repair rejects raw negative service or capacity violations instead of
hiding them in a tolerance. Its guarantees do not include ramps, mixed service
classes or nonpreemptive jobs. Those require the corresponding original coupled
model. Monetary cost and terminal changes must be checked separately using the
original account. ../../results/conditional_money_bound.json is a conditional calculation
from reported diagnostic maxima; it is not an executed annual witness repair.

../source_contract_checks.py requires NumPy and the original public repository
checkout at commit 323c2084e7ee668ba512fb1e33dd56b8394b32d4. Run:
    python ../source_contract_checks.py --repo /path/to/repository --output-dir results

It extracts the original functions with AST to avoid model training/data-loading
side effects. Its 16 fixtures verify the observed implementation behavior,
including the limitations documented in ../../audit/mathematical_and_contract_audit.txt.
A passing fixture does not mean the implementation's limitation has been fixed
or that an annual scientific result has been reproduced. No source files are
changed by these checks.
