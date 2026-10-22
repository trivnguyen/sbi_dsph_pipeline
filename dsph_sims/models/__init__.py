"""The registered model specs, one module per family / prior variant.

Naming: spec name == dataset directory == wandb project, in the
`<N>p_<Halo><Tracer><DF>[_v<k>]` form (`8p_ZhaoPlumCOM_v4`); the module
file is the same tokens reordered so it is a valid Python identifier
(`ZhaoPlumCOM_8p_v4.py`). Add a model by writing `<Family>_<N>p[_v<k>].py`
next to these (see the three `ZhaoPlumCOM_8p*.py` for the explicit,
one-variant-per-file style) and listing its `SPEC` here.
"""

from . import (FlatZhaoPlumCOM_10p, ZhaoABGCOM_11p, ZhaoABGCOM_11p_v2,
               ZhaoPlumCOM_8p, ZhaoPlumCOM_8p_v3, ZhaoPlumCOM_8p_v4,
               ZhaoPlumGammaCOM_9p, ZhaoPlumJeansBVH_10p_v1,
               ZhaoPlumJeansBVH_14p_v1, ZhaoPlumJeansBVH_14p_v2)
from ..spec import ModelSpec

SPECS: dict[str, ModelSpec] = {
    s.name: s for s in (
        ZhaoPlumCOM_8p.SPEC, ZhaoPlumCOM_8p_v3.SPEC, ZhaoPlumCOM_8p_v4.SPEC,
        ZhaoPlumGammaCOM_9p.SPEC, FlatZhaoPlumCOM_10p.SPEC,
        ZhaoABGCOM_11p.SPEC, ZhaoABGCOM_11p_v2.SPEC,
        ZhaoPlumJeansBVH_10p_v1.SPEC_TWIN,
        ZhaoPlumJeansBVH_10p_v1.SPEC_KTWIN,
        ZhaoPlumJeansBVH_14p_v1.SPEC_TWIN,
        ZhaoPlumJeansBVH_14p_v1.SPEC_KTWIN,
        ZhaoPlumJeansBVH_14p_v2.SPEC_TWIN,
        ZhaoPlumJeansBVH_14p_v2.SPEC_KTWIN)
}
