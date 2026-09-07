# Customizing the plots

This is a manual for **you** (the app owner) on changing how the profile
panels and corner plot look — both the knobs already exposed in the
browser UI, and the deeper things you change by editing two source
files. It's not meant for collaborators; the collaborator-facing doc is
`RUNNING_THE_APP.md`.

Everything here is about presentation only. None of it changes the
inference itself (the posterior, the Jeans profiles, the Wolf mass) —
those come from the model and your catalog.

## How edits take effect

- **UI controls** (everything in §1): live in the browser, no restart.
- **Code edits** (§2): the two files are
  `static/index.html` (all the interactive plotting) and
  `inference.py` (the corner PNG, made server-side by
  `matplotlib`/`corner`).
  - Running locally with `python app.py`: `index.html` is re-read from
    disk on every page load, so just **save the file and reload the
    browser tab**. Editing `inference.py` needs a server restart
    (`Ctrl+C`, run `python app.py` again).
  - Running from the Docker/Apptainer bundle: the files are baked into
    the image, so edit them in `webapp/` (or the unpacked bundle)
    **before** you build/package. See `DEPLOY.md`.

---

## 1. What you can change from the browser (no code)

These panels appear under the profile plot after a run.

### Panel layout

| Control | Effect |
|---|---|
| **Columns** | How many columns to arrange the 5 profile panels into. `1` = the original tall single column; `2` gives a 3×2 grid, etc. |
| **Rows** | Leave blank to auto-fit the columns you chose. Set it only if you want *more* rows than needed (e.g. extra vertical spread). |
| **Plot width [px]** | Blank = auto (widens with more columns). Set a number to control the aspect ratio directly — smaller = narrower/taller-looking, larger = wider. |
| **Plot height [px]** | Blank = auto (~300 px per row). |
| **Share x-axis** | On: all panels share one radius axis (zoom one, they all move; only the bottom panel of each column shows tick labels). Off: every panel gets its own independent radius axis and its own labels. |

Panels fill **left-to-right, top-to-bottom** in the order they're
defined (density, mass, β, σ_LOS, κ_LOS). On a wide screen, `Columns =
2` and a `Plot width` of ~1200–1400 is a good way to stop the 5×1 column
from being absurdly tall.

Note on the **Axis ranges** panel: the single *Radius* min/max box drives
every panel's x-axis, even when "Share x-axis" is off (it's broadcast to
all of them). The per-quantity y-boxes each drive their own panel.

### Corner plot

| Control | Effect |
|---|---|
| **Show corner plot** | Instant show/hide. Doesn't delete anything — re-check to bring it back. |
| **Smoothing** | Gaussian smoothing width (in histogram bins) for the 2-D and 1-D densities. Blank = none (raw histograms). Try `1.0`–`1.5` for smoother contours. |
| **Bins per parameter** | Histogram resolution. Fewer bins = coarser/smoother; more = finer/noisier. |
| **Contours** | `default` = corner's own contour lines; `filled` = shaded filled contours; `lines only`; `off` = no contours. |
| **Show individual data points** | Toggle the scatter of individual posterior draws. |

The corner options re-render from the **stored posterior**, so they're
cheap — no re-inference. They apply on the **Update corner** button
(not live), because each redraw is a server round-trip. In compare mode
both datasets' corners update together.

### Display options (recap)

Already documented by the tooltips in that panel, but for completeness:
credible-band levels (up to 3, each a toggle + editable central %),
band opacity, median-line width, per-series markers (median / binned /
Wolf), and a color picker per dataset.

---

## 2. Deeper changes (editing the source)

### 2a. Which panels, their titles, and log/linear — `static/index.html`

Near the top of the `<script>` block:

```js
// Profile panels: key, y-axis title, y scale. x (radius) is shared/log.
const PANELS = [
  ['rho',   'Density [M☉/kpc³]',   'log'],
  ['mass',  'Enclosed mass [M☉]',  'log'],
  ['beta',  'Anisotropy β',         'linear'],
  ['sigma', 'LOS dispersion [km/s]','linear'],
  ['kappa', 'LOS kurtosis',         'linear'],
];
```

- **Rename an axis**: edit the middle string.
- **Switch a panel between log/linear**: change `'log'`/`'linear'`.
- **Reorder panels**: reorder the rows (they render in this order).
- **Drop a panel**: delete its row. The layout, the axis-range boxes,
  and everything else adapt to `PANELS.length` automatically.

The first element (`'rho'`, `'mass'`, …) is the data key — it must stay
one of the keys the backend ships (`rho`, `mass`, `beta`, `sigma`,
`kappa`, defined as `PROFILE_KEYS` in `inference.py`). You can hide or
reorder these keys freely, but adding a brand-new profile means adding
it on the backend too (see 2c).

Two things are wired to specific panels by their data key, not position,
so they follow the panel when you reorder: the **binned data points**
(on `sigma` and `kappa`) and the **Wolf marker + dotted line** (on
`mass`). If you *remove* the `mass` or `sigma`/`kappa` panels, those
overlays simply won't have a panel to land on.

If you change how many panels there are, also bump the `max="5"` on the
Rows/Columns inputs in the `layoutBox` fieldset to match.

### 2b. Default colors and band levels — `static/index.html`

```js
// Default per-dataset colors (blue / orange). Editable per run in the UI.
const DS_COLOR = { A: '#2a78d6', B: '#e8710a' };
```

Change these hex values to set the **starting** colors (the UI color
pickers still let you override per run).

Default credible-band levels live in `buildBandRows()`:

```js
const defs = [[68, true], [90, true], [99, false]];
```

`[percent, checkedByDefault]`. Edit the numbers or the on/off flags.
There are three rows by design; the innermost-opacity math keys off how
many are enabled.

The default marker sizes/symbols are in `binnedTrace()` and
`wolfTraces()` if you want to touch those.

### 2c. Corner defaults and extra `corner.corner` options — `inference.py`

`plot_corner()` builds the corner figure. The UI already reaches
`smooth`, `bins`, `contours`, and `plot_datapoints` through the
`options` dict. To change a **default** (what a fresh run shows before
you touch the UI) or expose another `corner.corner` argument, edit the
`kwargs` dict there:

```python
kwargs = dict(
    labels=prior.ALL_PARAM_NAMES, color=o.get('color', '#2a78d6'),
    show_titles=True, title_fmt='.2f', quantiles=[0.16, 0.5, 0.84],
    range=_corner_ranges(posterior), bins=int(o.get('bins') or 20),
    smooth=smooth, smooth1d=smooth,
    plot_datapoints=bool(o.get('plot_datapoints', True)),
)
```

Common tweaks: `quantiles` (the dashed lines / title percentiles),
`title_fmt`, `color`, `label_kwargs={'fontsize': ...}`, `dpi` in the
`savefig` call below it. Any argument `corner.corner` accepts can go
here — see <https://corner.readthedocs.io>. To make one of them a live
UI control, add a field in the `cornerBox` fieldset in `index.html`,
read it in `cornerOptions()`, and pull it out of `options` in the
`/api/corner` handler in `app.py` (follow how `smooth`/`bins` are
threaded).

Note `_corner_ranges()` — leave it in. It exists so that a parameter
that collapses to a single value (which happens when `R_half` is given
with no uncertainty) doesn't make `corner` hard-error.

---

## Quick recipes

- **Wide screen, less scrolling:** UI → Panel layout → Columns `2`,
  Plot width `1300`.
- **Square-ish single panels for a talk:** Columns `1`, Plot width
  `900`, Plot height `2200`.
- **Independent zoom per panel:** uncheck Share x-axis, then drag-zoom
  each panel on its own.
- **Cleaner corner for a figure:** Smoothing `1.0`, Contours `filled`,
  uncheck Show individual data points, Update corner.
- **Only show density + mass:** in `PANELS`, delete the `beta`,
  `sigma`, `kappa` rows; reload.
