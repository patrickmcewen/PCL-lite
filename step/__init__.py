"""PCL-lite `step` shim — re-exports the step_tl surface that benchmarks and
prompts target.

The old inlined `step/` package (base.py, ops.py) is gone; both PCL-lite test
codegen and prompts now speak the step_tl vocabulary directly. We keep the
`step.*` namespace because benchmarks, prompts, and `tools/yaml_to_code.py`
all reference it.
"""
import os
import sys

_step_tl_src = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "step_tl", "src")
)
assert os.path.isdir(_step_tl_src), (
    f"step_tl source tree not found at {_step_tl_src}. Expected step_tl to "
    f"be checked out as a sibling submodule of PCL-lite."
)
if _step_tl_src not in sys.path:
    sys.path.insert(0, _step_tl_src)

from networkx import MultiDiGraph as Graph  # noqa: E402

from step_py.datatype import (  # noqa: E402
    Buffer,
    DynDim,
    DynTile,
    Float16,
    Float32,
    MultiHot,
    Stream,
    Tile,
    Uint32,
    Uint64,
)
from step_py.ops import (  # noqa: E402
    Accum,
    BinaryMap,
    BinaryMapAccum,
    Broadcast,
    Bufferize,
    DynLinearOffChipLoad,
    DynOffChipStore,
    DynStreamify,
    EagerMerge,
    ExpandRef,
    Flatten,
    FlatPartition,
    FlatReassemble,
    FlatmapCounter,
    FlatmapFilterRowStreamify,
    LinearOffChipLoad,
    LinearOffChipLoadRef,
    MockStreamOp,
    OffChipStore,
    Parallelize,
    Promote,
    PromoteOuter,
    RandomOffChipLoad,
    RandomOffChipStore,
    RepeatRef,
    RepeatStatic,
    Reshape,
    ReshapePadStream,
    RetileStreamify,
    StaticReassemble,
    Streamify,
    UnaryMap,
)
from step_py.functions import accum_fn, init_fn, map_accum_fn, map_fn  # noqa: E402
from step_py.utility_ops import SelectGen  # noqa: E402
from timing_and_emulator.functional import execute  # noqa: E402
