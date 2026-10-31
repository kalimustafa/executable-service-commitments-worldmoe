from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parent/'github_release_20261002/executable-service-commitments-worldmoe'
TARGET=REPO/'revisions/20261031_seven_windows'
assert (REPO/'.git').exists() and TARGET.resolve().is_relative_to(REPO.resolve())
shutil.copytree(ROOT/'publication_package',TARGET,dirs_exist_ok=True)
p=REPO/'README.md';s=p.read_text(encoding='utf-8')
a=s.index('## Current manuscript');b=s.index('## Contents',a)
new='''## Current manuscript and completed seven-window evidence

- [Current main manuscript](revisions/20261031_seven_windows/manuscript/EnergyEconomics_Seven_Window_Revision_20261031.docx), **18 pages**, with six readable figures in the original purple/pink palette and a concise economic literature comparison.
- [Current Supplementary Information](revisions/20261031_seven_windows/manuscript/EnergyEconomics_Seven_Window_Supplement_20261031.docx), **65 pages**, with S31–S32 and twelve retained original images. All current figure captions are short: at most 25 words in the main text and 23 in the supplement.
- [Complete evidence guide and reproduction commands](revisions/20261031_seven_windows/README.md).
- [Current figures and drawing source](revisions/20261031_seven_windows/figures/).
- [Previous recovered-admission revision](revisions/20261031_contract_value/), retained as a separate evidence vintage.

The extension completes **15 new annual policy evaluations** with the original three frozen checkpoints, including three 24 h anchors that reproduce the historical complete books and settlement trajectories bit for bit. It also computes **28 site–deadline certificates** for 8/12/16/20/24/36/48 h on the same recovered actual admissions and account. No new training or fitting was performed.

| Deadline | Mean policy cash saving US$ | Service status | Certified PI opportunity US$ |
|---:|---:|---|---:|
| 8 h | 50,735.34 | Substantive failure; about 3.01 IT MWh late | 78,751.60–78,882.41 |
| 12 h | 72,220.97 | Exact feasible counterpart | 97,639.33–97,796.41 |
| 16 h | 86,038.55 | Exact feasible counterpart | 113,119.89–113,305.19 |
| 20 h | 98,199.02 | Exact feasible counterpart | 124,528.17–124,733.99 |
| 24 h | 105,271.06 | Exact feasible counterpart | 132,376.82–132,598.65 |
| 36 h | 114,251.12 | Exact feasible counterpart | 141,996.61–142,256.41 |
| 48 h | 123,101.20 | Exact feasible counterpart | 152,798.28–153,070.23 |

The 8 h cash-only outcomes are retained without repair and are excluded from feasible policy value, capture and headroom. The other new executions have separately saved exact numerical witnesses. The 36/48 h implementation retains a fixed 24 h neural horizon and complete longer queues, with an explicitly declared deterministic continuation; it does not claim a new 48 h neural forecast or conservative price bound.

The current actual-admission 24 h remaining opportunity is **US$27,105.76–27,327.58**, capture is **79.39%–79.53%**, and the optimal 12→24 h rights increment is **US$34,580.41–34,959.31**. The historical same-checkpoint policy increment remains **US$33,050.09**. These are distinct economic objects. Earlier bounds uniform over all reference-compatible admissions remain valid robustness results in S27–S29.

Certified incremental opportunity per extra deadline hour rises from **US$783.16–823.30** over 24→36 h to **US$878.48–922.81** over 36→48 h, while the policy increments fall from **US$748.34** to **US$737.51**. Thus the operating curve alone does not identify the shape of the contractual opportunity curve. These results describe the observed simulation calendar, not promised future profit.

The earlier software-archive instructions below retain their historical scope and figure labels. The current manuscripts and S31–S32 supersede the previous status that additional windows had not yet been evaluated.

'''
s=s[:a]+new+s[b:]
s=s.replace('This release includes the selected **decision-module** checkpoints','The original v1.3 release includes the selected **decision-module** checkpoints')
s=s.replace('Full retraining and full-calendar replay are not self-contained in this release;', 'In that original release, full retraining and full-calendar replay are not self-contained;')
p.write_text(s,encoding='utf-8')
print('Prepared GitHub revision:',TARGET)
