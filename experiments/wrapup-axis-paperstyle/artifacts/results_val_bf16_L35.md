# Paper-style e: val results, bf16_L35 (8 held-out rollouts, position-matched)

| variant | model | AUROC | 90% CI | g-mean² | FPR tricky | FPR all neg | AUROC vs tricky | n tricky |
|---|---|---|---|---|---|---|---|---|
| any_call | e_paper (diffmean) | 0.741 | [0.701, 0.786] | 0.469 | 0.00 | 0.28 | nan | 1 |
| any_call | e_withinturn_nearest | 0.738 | [0.692, 0.780] | 0.461 | 0.00 | 0.34 | nan | 1 |
| any_call | e_withinturn_all | 0.730 | [0.686, 0.773] | 0.449 | 0.00 | 0.36 | nan | 1 |
| any_call | s | 0.746 | [0.712, 0.790] | 0.487 | 0.00 | 0.29 | nan | 1 |
| any_call | -u | 0.622 | [0.571, 0.661] | 0.361 | 0.00 | 0.37 | nan | 1 |
| any_call | position baseline (logreg) | 0.557 | [0.521, 0.627] | 0.322 | 0.00 | 0.36 | nan | 1 |
| any_call | logreg probe (C=1e-4) | 0.798 | [0.773, 0.832] | 0.525 | 0.00 | 0.22 | nan | 1 |

| cosine | value |
|---|---|
| cos(e_paper_any_call,u) | -0.239 |
| cos(e_paper_any_call,s) | +0.595 |
| cos(e_paper_any_call,e_withinturn_nearest_any_call) | +0.880 |
| cos(e_paper_any_call,e_withinturn_all_any_call) | +0.818 |
| cos(e_withinturn_nearest_any_call,u) | -0.230 |
| cos(e_withinturn_nearest_any_call,s) | +0.449 |
| cos(e_withinturn_nearest_any_call,e_paper_any_call) | +0.880 |
| cos(e_withinturn_nearest_any_call,e_withinturn_all_any_call) | +0.953 |
| cos(e_withinturn_all_any_call,u) | -0.220 |
| cos(e_withinturn_all_any_call,s) | +0.421 |
| cos(e_withinturn_all_any_call,e_paper_any_call) | +0.818 |
| cos(e_withinturn_all_any_call,e_withinturn_nearest_any_call) | +0.953 |
| cos(logreg_any_call,e_paper_any_call) | +0.260 |
