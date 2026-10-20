# plot_style

Figure style for this repo. The rules below are the source of truth;
`base.mplstyle` and `__init__.py` implement them for matplotlib.

```python
from steering_reasoning import plot_style
plot_style.use()                # experiment figures
plot_style.use("paper")         # paper figures: paper wheel, thinner lines and markers
plot_style.heatmap(ax, m)       # picks the scale by polarity, ink edges, numbers in cells
ax.imshow(m)                    # plain imshow defaults to paper_teal
```

`fmt_value()` and `nice_ceiling()` implement the number and scale-limit rules
below. `paper-artifacts/paper_style.py` builds on `use("paper")` and swaps the
font to Computer Modern Sans for the paper.

## Colors

| Name | Hex | Role |
|---|---|---|
| `PAPER` | `#fefdfc` | background; light end of heatmap scales |
| `INK` | `#111111` | text, axes, bar and cell edges, error bars |
| `MUTED` | `#4a4a4a` | secondary labels, tick marks, reference lines |
| `FAINT` | `#9a9a9a` | de-emphasized series |
| `RULE` | `#e5e5e5` | soft dividers |
| `TEAL` | `#3a7d85` | positive, treated, good; first series |
| `VERMILLION` | `#b8442b` | the baseline or bad case |
| `NEUTRAL` | `#7d7d7d` | a series that is neither: a control, a reference, chance |

Teal and vermillion were chosen over green and red so the pair stays distinct
under deuteranopia.

Two wheels for series with no good-or-bad reading, used in order:

- experiments (`CATEGORICAL`): teal `#3a7d85`, ochre `#9c6c24`, sage `#5a6f33`,
  slate `#356a7e`, plum `#76566f`.
- paper (`PAPER_CATEGORICAL`): teal `#3a7d85`, yellow `#b3952f`, green `#5a7a3a`,
  cool grey `#6b7a88`, purple `#78628e`. The tones are close in lightness so
  they differ by hue and none dominates; the cool grey is distinct from the
  warm `NEUTRAL`.

## Rules

**Text.** Every piece of text on a figure is in the mono face (Google Sans
Code, falling back to system monospace): title, axis labels, tick labels,
legend, annotations. The paper variant uses Computer Modern Sans instead, to
match the body text. Google Sans Code has no Greek or math glyphs; spell them
out or use mathtext.

**Titles.** Optional, and often absent. When present, a title is a short
left-aligned description of what is plotted, such as `loss by learning rate`.
It does not state the conclusion; that belongs in the caption.

**Axes and lines.** Axes in ink, 1px. No gridlines. Only the left and bottom
spines are drawn. Dashed `3 3` strokes mark projected or uncertain data.

**Series colors.** When the series carry valence, the color says which: teal
for the positive or treated case, vermillion for the baseline or bad case,
neutral grey for a control or reference. When the series are merely different
things, the wheel applies in order. Vermillion is never in the wheel. Five
series is the practical limit; beyond that, use small multiples or grey every
series except the one under discussion.

**Marks.** Lines are line-only, with no area fill except a translucent
uncertainty band. Bars and heatmap cells are filled with tinted paper and
outlined in 1px ink; a heatmap's outer frame is no heavier than the lines
between cells. Point markers are outlined in ink (about 0.6px; 0.5px in the
paper variant) so overlapping points stay distinct.

**Error bars.** Plain vertical strokes in ink, 1px, with no caps. The series
keeps its own color for its line and markers. Intervals are 90% unless stated
otherwise, and the caption says what they are (e.g. "90% bootstrap").

**Heatmap scales.** Unipolar data (counts, magnitudes, probabilities) runs
from paper to teal. Bipolar data (deltas, correlations, residuals) runs from
vermillion through paper to teal, centred on the meaningful zero. Viridis and
other spectral maps are not used.

**Scale limits.** Bipolar scales are symmetric about zero wherever the data
allows. Limits round the extreme up to the next round number: `1.8` becomes
`2`, `0.85` becomes `1`, `25.7` becomes `30`, `132` becomes `150`.

**Numbers on figures.** Heatmap cells show their values by default, in ink on
light cells and paper on dark ones; values are omitted only when the matrix is
too dense, and the caption says so. Any number printed on a figure has about
three significant digits and at most two decimals: `0.85`, `1.74`, `25.7`,
`132`.
