"""Fixed-format user catalog parsing for the posterior web app.

Defines the upload data format the web app (webapp/app.py) accepts and
turns an uploaded catalog into the `TargetData` snapshot the inference
stack consumes. The per-star format is fixed: `ra`, `dec` [deg], `vr`,
`vr_err` [km/s], plus a per-star `distance` [kpc] or `dm` [mag] column
(common alias spellings are tolerated, case-insensitive - see
COLUMN_ALIASES). Anything beyond that is optional and auto-detected: a
membership-probability column (threshold cut) and arbitrarily-named
boolean "flag" columns (each can be ignored, required True, or required
False).

Two further row filters apply on top, and combine with the above (all
cuts are ANDed): a free-form `DataFrame.query` expression over the
file's own columns (e.g. `key == "draco_1"` to pull one system out of a
multi-system catalog), and a projected-radius cut in kpc or arcmin.
Radius is derived rather than read from the file, and is exposed to the
query as the reserved `R_kpc`/`R_arcmin` columns - see `build_target`
for the ordering constraint this creates.

System-level metadata the model needs but a per-star table can't carry
(half-light radius + uncertainty for the conditioning prior, center,
systemic velocity, proper motion for the perspective correction) comes
in as explicit arguments to `build_target`, with sensible data-driven
defaults where possible (center/systemic velocity from the member stars
themselves).
"""

import re
import sys
from pathlib import Path

import astropy.table as at
import astropy.units as auni
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

# In the packaged bundle, `tsnpe`/`dsph_analysis` sit next to this
# file; in the repo, the tsnpe package lives in ../tsnpe.
_APP_DIR = Path(__file__).resolve().parent
for _p in (_APP_DIR, _APP_DIR.parent / 'tsnpe'):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from tsnpe.target import TargetData

from dsph_analysis import data_utils
from dsph_analysis.kinematic_io import DwarfMeta

# Canonical role -> accepted (normalized) column spellings. Normalization
# lowercases and strips '_'/'-'/' ' (see _normalize), so e.g. 'VR_ERR',
# 'vr err' and 'vrerr' all match 'vr_err'.
COLUMN_ALIASES = {
    'ra': ('ra', 'radeg', 'raj2000', 'alpha'),
    'dec': ('dec', 'decdeg', 'dej2000', 'de', 'delta'),
    'distance': ('distance', 'distancekpc', 'dist', 'distkpc', 'dkpc'),
    'dm': ('dm', 'dmod', 'distmod', 'distancemodulus', 'mu'),
    'vr': ('vr', 'vlos', 'vrad', 'rv', 'vhel', 'vhelio'),
    'vr_err': ('vrerr', 'vloserr', 'vraderr', 'evr', 'rverr', 'verr',
               'errvr', 'evlos', 'vrade'),
    'mem_prob': ('memprob', 'prob', 'pmem', 'membershipprob', 'memp',
                 'pmember', 'membership'),
}
REQUIRED_ROLES = ('ra', 'dec', 'vr', 'vr_err')

# Derived projected-radius columns injected into the catalog before the
# radius cut, so a custom query can filter on radius too. These names
# are reserved - a same-named column in the uploaded file is replaced.
RADIUS_KPC_COL = 'R_kpc'
RADIUS_ARCMIN_COL = 'R_arcmin'
RADIUS_UNITS = ('kpc', 'arcmin')

# Passes allowed when resolving a radius-based query against the center
# it depends on (see _select_by_radius_query); a sane filter settles in
# two or three.
_MAX_QUERY_PASSES = 6

_TRUE_STRINGS = {'true', 't', 'yes', 'y', '1'}
_FALSE_STRINGS = {'false', 'f', 'no', 'n', '0'}

# A catalog center within this angular distance of a known system's
# center is taken as a positional match for the prefill suggestion.
_SUGGEST_MAX_SEP_DEG = 0.5


def _normalize(name: str) -> str:
    """Normalize a column name for alias matching."""
    return (name.strip().lower()
            .replace('_', '').replace('-', '').replace(' ', ''))


def read_catalog(path: str) -> pd.DataFrame:
    """Read an uploaded catalog file into a DataFrame.

    Args:
        path: Path to the uploaded file. `.ecsv`/`.fits` go through
            astropy Table; anything else is parsed as CSV.

    Returns:
        The catalog as a pandas DataFrame.
    """
    suffix = Path(path).suffix.lower()
    if suffix in ('.ecsv', '.fits'):
        return at.Table.read(path).to_pandas()
    return pd.read_csv(path)


def _is_flag_column(series: pd.Series) -> bool:
    """Whether a column looks boolean: bool dtype, {0, 1} numerics, or
    true/false-like strings.
    """
    if pd.api.types.is_bool_dtype(series):
        return True
    values = series.dropna().unique()
    if len(values) == 0:
        return False
    if pd.api.types.is_numeric_dtype(series):
        return set(np.unique(values.astype(float))) <= {0.0, 1.0}
    if series.dtype == object:
        lowered = {str(v).strip().lower() for v in values}
        return lowered <= (_TRUE_STRINGS | _FALSE_STRINGS)
    return False


def _flag_values(series: pd.Series) -> np.ndarray:
    """Parse a flag column (see _is_flag_column) to a boolean array.

    NaN entries parse as False, so 'require True' drops them and
    'require False' keeps them - consistent with treating NaN as
    "flag not set".
    """
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy()
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).to_numpy().astype(float) == 1.0
    return np.array([
        str(v).strip().lower() in _TRUE_STRINGS for v in series
    ])


def inspect_catalog(df: pd.DataFrame) -> dict:
    """Auto-detect the catalog's column roles for the upload UI.

    Args:
        df: Catalog as returned by `read_catalog`.

    Returns:
        JSON-friendly dict with `n_rows`, `columns` (all column names),
        `mapping` (role -> matched column name for every recognized
        role), `missing` (required roles with no match), `has_distance`
        (a distance or dm column matched), and `flag_columns`
        (boolean-like columns not consumed by any role).
    """
    mapping = {}
    for column in df.columns:
        key = _normalize(str(column))
        for role, aliases in COLUMN_ALIASES.items():
            if role not in mapping and key in aliases:
                mapping[role] = column
                break

    consumed = set(mapping.values())
    flag_columns = [
        str(c) for c in df.columns
        if c not in consumed and _is_flag_column(df[c])
    ]
    missing = [r for r in REQUIRED_ROLES if r not in mapping]
    return dict(
        n_rows=int(len(df)),
        columns=[str(c) for c in df.columns],
        mapping={k: str(v) for k, v in mapping.items()},
        missing=missing,
        has_distance=('distance' in mapping or 'dm' in mapping),
        flag_columns=flag_columns,
    )


def _known_key_lookup(meta_df: pd.DataFrame) -> dict:
    """Normalized system key/name -> canonical database key."""
    lookup = {}
    for _, row in meta_df.iterrows():
        canonical = str(row['key'])
        for field in ('key', 'name'):
            value = row.get(field)
            if isinstance(value, str) and value.strip():
                lookup.setdefault(_normalize(value), canonical)
    return lookup


def _keys_from_columns(df: pd.DataFrame, lookup: dict) -> dict:
    """Known database keys appearing in any string column of `df`.

    A column counts as a system-identifier column only if at least half
    its non-null values are recognized keys, so a stray coincidental
    match in a free-text column doesn't register.

    Returns:
        {canonical key: number of rows carrying it}.
    """
    found = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        values = df[col].dropna().astype(str)
        if len(values) == 0:
            continue
        matched = values.map(_normalize).map(lookup).dropna()
        if len(matched) < 0.5 * len(values):
            continue
        for key, count in matched.value_counts().items():
            found[key] = found.get(key, 0) + int(count)
    return found


def _nearest_system(
    ra_deg: float, dec_deg: float, meta_df: pd.DataFrame,
) -> tuple:
    """Closest known system to a sky position.

    Returns:
        (canonical key, separation in degrees), or (None, None) if the
        table has no usable coordinates.
    """
    sub = meta_df.dropna(subset=['ra', 'dec'])
    if len(sub) == 0:
        return None, None
    catalog = SkyCoord(sub['ra'].to_numpy() * auni.deg,
                       sub['dec'].to_numpy() * auni.deg)
    here = SkyCoord(ra_deg * auni.deg, dec_deg * auni.deg)
    sep = here.separation(catalog).deg
    i = int(np.argmin(sep))
    return str(sub.iloc[i]['key']), float(sep[i])


def suggest_system(
    df: pd.DataFrame, mapping: dict, meta_df: pd.DataFrame,
    max_sep_deg: float = _SUGGEST_MAX_SEP_DEG,
) -> dict:
    """Guess which known system an uploaded catalog is, to prefill the
    metadata form (always overridable by the user).

    Two signals, tried in order:
      1. A column that names the system - e.g. a `key`/`name` column
         holding a database key like `draco_1`. This is also what flags
         a multi-system file: if several distinct known keys appear, no
         single system is suggested and they are returned as candidates.
      2. Sky position - the catalog's median center matched to the
         nearest known system within `max_sep_deg`.

    Args:
        df: Catalog as returned by `read_catalog`.
        mapping: Role -> column mapping (needs `ra`/`dec` for the
            positional match); from `inspect_catalog`/`resolve_mapping`.
        meta_df: The known-systems table (`key`, `ra`, `dec`, `name`),
            e.g. from `kinematic_io.load_meta_table`.
        max_sep_deg: Positional-match tolerance.

    Returns:
        JSON-friendly dict: `suggested_key` (canonical key or None),
        `reason` ('key' | 'position' | 'multi' | None), `sep_arcmin`
        (for a positional match, else None), and `candidates` (the
        distinct known keys found in a name column, when more than one).
    """
    none = dict(suggested_key=None, reason=None, sep_arcmin=None,
                candidates=[])
    lookup = _known_key_lookup(meta_df)

    found = _keys_from_columns(df, lookup)
    if len(found) == 1:
        key = next(iter(found))
        return dict(suggested_key=key, reason='key', sep_arcmin=None,
                    candidates=[key])
    if len(found) > 1:
        candidates = sorted(found, key=lambda k: -found[k])
        return dict(suggested_key=None, reason='multi', sep_arcmin=None,
                    candidates=candidates)

    if 'ra' not in mapping or 'dec' not in mapping:
        return none
    ra = pd.to_numeric(df[mapping['ra']], errors='coerce').to_numpy()
    dec = pd.to_numeric(df[mapping['dec']], errors='coerce').to_numpy()
    finite = np.isfinite(ra) & np.isfinite(dec)
    if not finite.any():
        return none
    c_ra, c_dec = _median_center(ra[finite], dec[finite])
    key, sep = _nearest_system(c_ra, c_dec, meta_df)
    if key is not None and sep <= max_sep_deg:
        return dict(suggested_key=key, reason='position',
                    sep_arcmin=round(sep * 60.0, 1), candidates=[key])
    return none


def _median_center(ra_deg: np.ndarray, dec_deg: np.ndarray) -> tuple:
    """Median sky position, handling the RA wrap at 0/360.

    A field straddling RA=0 has raw values near both 0 and 360, whose
    plain median lands near 180 - shifting by 180 deg first moves the
    wrap to RA=180, far from any such field's stars.
    """
    if np.ptp(ra_deg) > 180.0:
        ra_center = (np.median((ra_deg + 180.0) % 360.0) - 180.0) % 360.0
    else:
        ra_center = float(np.median(ra_deg))
    return float(ra_center), float(np.median(dec_deg))


def query_uses_radius(query: str) -> bool:
    """Whether a query expression references a derived radius column.

    Such a query can only run once the center/distance are known, so it
    is deferred until after the radius columns are injected (see
    `build_target`); queries on the file's own columns run before, since
    they are what selects the system an auto-center is derived from.
    """
    return any(
        re.search(rf'\b{re.escape(name)}\b', query)
        for name in (RADIUS_KPC_COL, RADIUS_ARCMIN_COL)
    )


def _query_rows(data: pd.DataFrame, query: str) -> pd.DataFrame:
    """Evaluate a query expression, allowing an empty result.

    Raises:
        ValueError: If the expression is invalid or names an unknown
            column.
    """
    try:
        return data.query(query)
    except Exception as e:
        raise ValueError(
            f'Could not evaluate the row filter {query!r}: {e}. '
            f'Available columns: {[str(c) for c in data.columns]}')


def apply_query(data: pd.DataFrame, query: str) -> pd.DataFrame:
    """Filter rows with a pandas `DataFrame.query` expression.

    Args:
        data: Catalog rows to filter.
        query: Expression over the catalog's columns, e.g.
            `key == "draco_1"` or `mem_prob > 0.8 and R_kpc < 5`.

    Returns:
        The matching rows.

    Raises:
        ValueError: If the expression is invalid, names an unknown
            column, or matches no rows.
    """
    out = _query_rows(data, query)
    if len(out) == 0:
        raise ValueError(f'No stars match the row filter {query!r}.')
    return out


def _apply_radius_cut(
    data: pd.DataFrame, radius_min: float, radius_max: float,
    radius_unit: str,
) -> pd.DataFrame:
    """Keep rows whose projected radius is within [min, max] in the
    chosen unit (either bound may be None). Assumes the derived radius
    columns are already present.
    """
    if radius_min is None and radius_max is None:
        return data
    unit = (radius_unit or 'kpc').lower()
    if unit not in RADIUS_UNITS:
        raise ValueError(
            f'Unknown radius unit {radius_unit!r} - use one of '
            f'{list(RADIUS_UNITS)}.')
    radius = data[
        RADIUS_KPC_COL if unit == 'kpc' else RADIUS_ARCMIN_COL
    ].to_numpy()
    keep = np.ones(len(data), dtype=bool)
    if radius_min is not None:
        keep &= radius >= radius_min
    if radius_max is not None:
        keep &= radius <= radius_max
    out = data[keep]
    if len(out) == 0:
        raise ValueError(
            f'No stars left after the radius cut '
            f'[{radius_min}, {radius_max}] {unit}.')
    return out


def resolve_mapping(df: pd.DataFrame, columns: dict = None) -> dict:
    """Final role -> column mapping: auto-detection plus user choices.

    Args:
        df: Catalog as returned by `read_catalog`.
        columns: Optional overrides from the UI's per-role column
            dropdowns - {role: column name}, where an empty/None column
            name unsets an auto-detected role (e.g. "don't use the
            membership column after all").

    Returns:
        {role: column name} with overrides applied.

    Raises:
        ValueError: If an override names an unknown role or a column
            not in the catalog, or a required role ends up unmapped.
    """
    mapping = inspect_catalog(df)['mapping']
    for role, column in (columns or {}).items():
        if role not in COLUMN_ALIASES:
            raise ValueError(f'Unknown column role: {role!r}')
        if column:
            if column not in df.columns:
                raise ValueError(
                    f'Column {column!r} (chosen for {role!r}) not in '
                    f'the catalog.')
            mapping[role] = column
        else:
            mapping.pop(role, None)

    missing = [r for r in REQUIRED_ROLES if r not in mapping]
    if missing:
        raise ValueError(
            f'No column assigned for required role(s): {missing}. '
            f'Catalog columns: {[str(c) for c in df.columns]}')
    return mapping


def _derive_metadata(
    data: pd.DataFrame, mapping: dict, center_ra_deg: float,
    center_dec_deg: float, distance_kpc: float,
    vlos_systemic_kms: float,
) -> tuple:
    """Fill the data-driven metadata defaults from `data`.

    Explicitly supplied values pass through untouched; only the None
    ones are estimated from the stars.

    Returns:
        (center_ra_deg, center_dec_deg, distance_kpc,
        vlos_systemic_kms).

    Raises:
        ValueError: If no distance is available at all.
    """
    if distance_kpc is None:
        if 'distance' in mapping:
            distance_kpc = float(np.nanmedian(
                data[mapping['distance']].to_numpy().astype(float)))
        elif 'dm' in mapping:
            dm = float(np.nanmedian(
                data[mapping['dm']].to_numpy().astype(float)))
            distance_kpc = 10.0 ** (dm / 5.0 - 2.0)
        else:
            raise ValueError(
                'Catalog has no distance/dm column - provide the '
                'system distance explicitly.')
    if center_ra_deg is None or center_dec_deg is None:
        auto_ra, auto_dec = _median_center(
            data[mapping['ra']].to_numpy().astype(float),
            data[mapping['dec']].to_numpy().astype(float))
        center_ra_deg = auto_ra if center_ra_deg is None else center_ra_deg
        center_dec_deg = (auto_dec if center_dec_deg is None
                          else center_dec_deg)
    if vlos_systemic_kms is None:
        vlos_systemic_kms = float(np.nanmedian(
            data[mapping['vr']].to_numpy().astype(float)))
    return center_ra_deg, center_dec_deg, distance_kpc, vlos_systemic_kms


def _add_radius_columns(
    data: pd.DataFrame, mapping: dict, center_ra_deg: float,
    center_dec_deg: float, distance_kpc: float,
) -> pd.DataFrame:
    """Return a copy of `data` with the derived radius columns, using
    the same small-angle convention as the preprocessing step that
    later recomputes R_proj for the surviving stars.
    """
    data = data.copy()
    data[RADIUS_KPC_COL] = data_utils.calc_projected_radius(
        data[mapping['ra']].to_numpy().astype(float),
        data[mapping['dec']].to_numpy().astype(float),
        center_ra_deg, center_dec_deg, distance_kpc)
    data[RADIUS_ARCMIN_COL] = np.rad2deg(
        data[RADIUS_KPC_COL].to_numpy() / distance_kpc) * 60.0
    return data


def _select_by_radius_query(
    base: pd.DataFrame, query: str, mapping: dict, center_ra_deg: float,
    center_dec_deg: float, distance_kpc: float,
    vlos_systemic_kms: float,
) -> pd.DataFrame:
    """Resolve a query that filters on radius, where the selection and
    the center are mutually dependent.

    Radius is measured from the center, but an auto center (and
    distance, and systemic velocity) is the median of whichever stars
    the query selects - so neither can be computed first. Iterate to a
    fixed point instead: estimate the metadata from the current
    selection, re-run the query with radii from it, and stop once the
    selection stops changing.

    The seed matters. Starting from every star would measure the center
    of a multi-system catalog off the majority system, putting a
    minority one thousands of arcmin away and letting an upper-bound
    radius term reject it outright - before the loop could correct the
    center. So seed by evaluating the query with radius standing in as
    zero, which admits everything on an upper-bound term and lets the
    query's other predicates (`key == "bootes_1"`) pick the system. A
    lower-bound-only radius term selects nothing that way, so fall back
    to every star and let the loop try.

    Raises:
        ValueError: If the selection never settles.
    """
    seed = base.copy()
    seed[RADIUS_KPC_COL] = 0.0
    seed[RADIUS_ARCMIN_COL] = 0.0
    selected = _query_rows(seed, query)
    if len(selected) == 0:
        selected = base
    for _ in range(_MAX_QUERY_PASSES):
        c_ra, c_dec, dist, _ = _derive_metadata(
            selected, mapping, center_ra_deg, center_dec_deg,
            distance_kpc, vlos_systemic_kms)
        nxt = apply_query(
            _add_radius_columns(base, mapping, c_ra, c_dec, dist), query)
        if nxt.index.equals(selected.index):
            return nxt
        selected = nxt
    raise ValueError(
        f'Could not settle on a center for the radius-based row filter '
        f'{query!r}: the stars it selects and the center they imply '
        f'keep changing. Set the center (and distance) explicitly, or '
        f'move the radius part into the radius-cut boxes.')


def build_target(
    df: pd.DataFrame,
    label: str,
    rhalf_kpc: float,
    rhalf_kpc_em: float,
    rhalf_kpc_ep: float,
    columns: dict = None,
    center_ra_deg: float = None,
    center_dec_deg: float = None,
    distance_kpc: float = None,
    vlos_systemic_kms: float = None,
    pmra_masyr: float = None,
    pmdec_masyr: float = None,
    vlos_abs_max: float = None,
    mem_prob_min: float = None,
    flag_requirements: dict = None,
    apply_perspective_corr: bool = True,
    query: str = None,
    radius_min: float = None,
    radius_max: float = None,
    radius_unit: str = 'kpc',
) -> tuple[TargetData, dict]:
    """Build a TargetData snapshot from a fixed-format user catalog.

    Args:
        df: Catalog as returned by `read_catalog`.
        label: Display name for this run (becomes TargetData.key).
        rhalf_kpc: Projected half-light radius [kpc] - required, the
            model conditions on it.
        rhalf_kpc_em: Lower (minus) uncertainty on rhalf_kpc [kpc].
        rhalf_kpc_ep: Upper (plus) uncertainty on rhalf_kpc [kpc].
        columns: Optional role -> column-name overrides on top of the
            auto-detected mapping (see `resolve_mapping`).
        center_ra_deg: System center RA [deg]; None = median of the
            selected member stars.
        center_dec_deg: System center Dec [deg]; None = median.
        distance_kpc: System distance [kpc]; None = median of the
            catalog's distance (or dm) column. Required if the catalog
            has neither column.
        vlos_systemic_kms: Systemic LOS velocity [km/s]; None = median
            of the selected stars' vr.
        pmra_masyr: Systemic proper motion in RA*cos(Dec) [mas/yr];
            needed (with pmdec_masyr) for the perspective correction.
        pmdec_masyr: Systemic proper motion in Dec [mas/yr].
        vlos_abs_max: Optional cut on |vr - v_sys| [km/s].
        mem_prob_min: Optional membership-probability threshold; only
            applied if the catalog has a membership column.
        flag_requirements: {column name: required bool value} for the
            catalog's flag columns; unlisted flags are ignored.
        apply_perspective_corr: Apply the full perspective-rotation
            correction (needs both proper motions); otherwise only the
            systemic velocity is subtracted.
        query: Optional `DataFrame.query` expression selecting the
            member stars, e.g. `key == "draco_1"`. It may reference the
            file's own columns and the derived `R_kpc`/`R_arcmin`
            columns. It defines the sample every data-driven default
            below is measured from, so picking one system out of a
            multi-system catalog here also gives that system's center,
            distance, and systemic velocity. A query that filters on
            radius is resolved against the center it implies (see
            `_select_by_radius_query`).
        radius_min: Optional lower projected-radius cut, in
            `radius_unit`.
        radius_max: Optional upper projected-radius cut.
        radius_unit: Unit for radius_min/radius_max: 'kpc' or 'arcmin'.

    Returns:
        (target, info) - the TargetData snapshot plus a JSON-friendly
        dict recording the star counts and the metadata values actually
        used (including defaults filled in from the data).

    Raises:
        ValueError: If required columns are missing, no distance is
            available, an unknown flag column is named, the query is
            invalid, or every star is cut.
    """
    mapping = resolve_mapping(df, columns)

    mask = np.ones(len(df), dtype=bool)
    for column, required in (flag_requirements or {}).items():
        if column not in df.columns:
            raise ValueError(f'Unknown flag column: {column!r}')
        mask &= _flag_values(df[column]) == bool(required)

    if 'mem_prob' in mapping and mem_prob_min is not None:
        mask &= np.nan_to_num(
            df[mapping['mem_prob']].to_numpy().astype(float)
        ) > mem_prob_min
    data = df[mask]
    if len(data) == 0:
        raise ValueError('All stars removed by flag/membership cuts.')

    # The query defines the member sample, so it runs before any
    # data-driven metadata: in a multi-system catalog it is what picks
    # the object the center/distance/systemic velocity describe.
    query = (query or '').strip()
    if query:
        data = (
            _select_by_radius_query(
                data, query, mapping, center_ra_deg, center_dec_deg,
                distance_kpc, vlos_systemic_kms)
            if query_uses_radius(query)
            else apply_query(data, query))

    center_ra_deg, center_dec_deg, distance_kpc, vlos_systemic_kms = (
        _derive_metadata(data, mapping, center_ra_deg, center_dec_deg,
                         distance_kpc, vlos_systemic_kms))

    # Radius comes last and never feeds back into the metadata above:
    # the center is a property of the member sample, not of whichever
    # annulus the cut happens to keep.
    data = _add_radius_columns(
        data, mapping, center_ra_deg, center_dec_deg, distance_kpc)
    data = _apply_radius_cut(data, radius_min, radius_max, radius_unit)

    ra = data[mapping['ra']].to_numpy().astype(float)
    dec = data[mapping['dec']].to_numpy().astype(float)
    vr = data[mapping['vr']].to_numpy().astype(float)
    vr_err = data[mapping['vr_err']].to_numpy().astype(float)
    mem_prob = (
        data[mapping['mem_prob']].to_numpy().astype(float)
        if 'mem_prob' in mapping else np.ones(len(data)))

    # The perspective correction needs a systemic proper motion; without
    # one, quietly fall back to plain systemic-velocity subtraction.
    have_pm = pmra_masyr is not None and pmdec_masyr is not None
    use_corr = bool(apply_perspective_corr and have_pm)

    meta = DwarfMeta(
        key=label,
        ra=center_ra_deg * auni.deg,
        dec=center_dec_deg * auni.deg,
        distance=distance_kpc * auni.kpc,
        pmra=(pmra_masyr or 0.0) * auni.mas / auni.yr,
        pmdec=(pmdec_masyr or 0.0) * auni.mas / auni.yr,
        vlos_systemic=vlos_systemic_kms * auni.km / auni.s,
        rhalf_arcmin=np.nan * auni.arcmin,
        rhalf_arcmin_em=np.nan * auni.arcmin,
        rhalf_arcmin_ep=np.nan * auni.arcmin,
        rhalf_kpc=rhalf_kpc * auni.kpc,
        rhalf_kpc_em=rhalf_kpc_em * auni.kpc,
        rhalf_kpc_ep=rhalf_kpc_ep * auni.kpc,
    )

    ra, dec, _, vr_err, mem_prob, vlos, r_proj = (
        data_utils.preprocess_kinematic_data(
            ra, dec, vr, vr_err, mem_prob, meta,
            vlos_abs_max=vlos_abs_max,
            apply_perspective_corr=use_corr,
        ))
    if len(ra) == 0:
        raise ValueError('All stars removed by NaN/vlos_abs_max cuts.')

    target = TargetData(
        key=label,
        ra_deg=ra.astype('float32'),
        dec_deg=dec.astype('float32'),
        vlos_kms=vlos.astype('float32'),
        vlos_err_kms=vr_err.astype('float32'),
        R_proj_kpc=r_proj.astype('float32'),
        rhalf_kpc=float(rhalf_kpc),
        rhalf_kpc_em=float(rhalf_kpc_em),
        rhalf_kpc_ep=float(rhalf_kpc_ep),
    )
    has_radius_cut = radius_min is not None or radius_max is not None
    info = dict(
        n_uploaded=int(len(df)),
        n_after_cuts=int(len(ra)),
        center_ra_deg=float(center_ra_deg),
        center_dec_deg=float(center_dec_deg),
        distance_kpc=float(distance_kpc),
        vlos_systemic_kms=float(vlos_systemic_kms),
        perspective_corr_applied=use_corr,
        rhalf_kpc=float(rhalf_kpc),
        query=query or None,
        radius_min=radius_min,
        radius_max=radius_max,
        radius_unit=(radius_unit or 'kpc') if has_radius_cut else None,
    )
    return target, info
