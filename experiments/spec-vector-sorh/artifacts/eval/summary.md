# Spec-vector and u-addition picks on the 309 held-out SoRH writing prompts (GLM 5)

Columns: hack = judge hack rate over all graded; hack|nd = among non-degenerate; empty = final answer empty (log); cap = output hit the proxy cap (log); degen = judge 'empty, cut off or loop'; h&c = neither hack nor degenerate; q = mean quality 1-5 over non-degenerate; p = paired exact McNemar test on hack vs base over the same prompts.

## all_309

| arm | n | hack | 95% CI | hack\|nd | empty | cap | degen | h&c | 95% CI | q | base-only / arm-only hacks | p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vspec_f40 | 309 | 0.30 | [0.25, 0.35] | 0.322 | 0.07 | 0.07 | 0.07 | 0.63 | [0.57, 0.68] | 3.79 | 0 / 0 |  |
| vspec_a32 | 309 | 0.29 | [0.24, 0.34] | 0.316 | 0.08 | 0.08 | 0.09 | 0.62 | [0.57, 0.68] | 3.91 | 0 / 0 |  |
| vspec_a24 | 309 | 0.27 | [0.22, 0.32] | 0.273 | 0.03 | 0.03 | 0.03 | 0.71 | [0.65, 0.75] | 4.0 | 0 / 0 |  |
| u_an24 | 309 | 0.28 | [0.23, 0.33] | 0.28 | 0.01 | 0.01 | 0.01 | 0.71 | [0.66, 0.76] | 4.03 | 0 / 0 |  |
