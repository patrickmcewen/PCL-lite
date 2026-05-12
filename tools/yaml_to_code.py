"""YAML -> Python test codegen for the step_tl-backed PCL-lite flow.

The codegen emits a test file with this layout:

    import step, torch, sympy ...
    input_data = {'E0': torch.randn(...), ...}

    def test():
        graph = step.Graph()
        out = body(graph, input_data['E0'], input_data['E1'], ...)
        out = step.execute(graph, out, input_tensors={})
        ref0 = <data_transform[0]>
        assert out.numel() == ref0.numel(), ...
        torch.testing.assert_close(out.flatten(), ref0.flatten())

    # `impl:` from the YAML is appended verbatim as `def body(graph, ...):`.

Inputs are passed to the user-written ``body(graph, A, B, ...)`` as raw
``torch.Tensor`` underlyings (named after the YAML's ``inputs[*].name``).
The body is responsible for picking a source op for each input (e.g.
``step.LinearOffChipLoad`` for fp32/fp16/uint64 tile streams,
``step.SelectGen`` for Multihot/Index streams) and for returning either an
``OffChipStore`` (single output) or a tuple of them (multi output).
``step.execute`` is the functional emulator from step_tl.

A short ``# Inputs:`` comment precedes the body signature so the LLM sees
the underlying torch dtype + shape of each arg at the call site.
"""

import argparse
import os
import re
from functools import reduce

import yaml


HEADER = """
import step
from sympy import Symbol
import torch
from tools.get_indices import generate_multi_hot, generate_binary_tensor

torch.manual_seed(42)
"""

# Legacy fixed-symbol block used when a task YAML does not declare its own
# ``dims:`` section. PCL-lite benchmarks under benchmark/ rely on these
# concrete values (M=5, N=7, K=9, D=16); StepDB-derived YAMLs declare an
# explicit ``dims:`` and swap this block out.
LEGACY_DIMS_BLOCK = """E = Symbol("E")
M = Symbol("M")
N = Symbol("N")
K = Symbol("K")
D = Symbol("D")
M_value = 5
N_value = 7
K_value = 9
D_value = 16
ctx = {
    M: M_value,
    N: N_value,
    K: K_value,
    D: D_value
}
"""


def _render_dims_block(dims):
    """Render ``Symbol(...) + <name>_value`` lines for a task-supplied dims map.

    ``dims`` is ``{name: int}`` (preset values). Emits one
    ``<name> = Symbol("<name>")`` and one ``<name>_value = <int>`` per entry,
    plus a sympy ``ctx`` dict. Replaces the legacy M/N/K/D block whenever
    the task YAML carries an explicit ``dims:`` section.
    """
    assert dims, "empty dims map"
    sym_lines = "\n".join(f'{n} = Symbol("{n}")' for n in dims)
    val_lines = "\n".join(f"{n}_value = {int(v)}" for n, v in dims.items())
    ctx_entries = ", ".join(f"{n}: {n}_value" for n in dims)
    return f"{sym_lines}\n{val_lines}\nctx = {{{ctx_entries}}}\n"


def _prefix_for(data):
    """Prefix block for a task: header + dims constants."""
    dims = data.get("dims")
    return HEADER + (_render_dims_block(dims) if dims else LEGACY_DIMS_BLOCK)


# Back-compat alias: callers (helpers in this module + yaml_plan_to_code)
# that previously concatenated the legacy prefix continue to work.
prefix = HEADER + LEGACY_DIMS_BLOCK


# ---------------------------------------------------------------------------
# Helpers used by both yaml_to_code and yaml_plan_to_code
# ---------------------------------------------------------------------------


def replace_one_with_str(dims):
    """Normalize a PCL-lite ``dims`` list: ints stay, ``1`` becomes the
    string ``"1"`` so it can be templated alongside symbol names.
    """
    if isinstance(dims, list):
        return list(map(lambda x: str(x) if x == 1 else x, dims))
    elif dims == 1:
        return "1"
    elif isinstance(dims, str):
        return dims
    else:
        raise ValueError(f"Unknown dims: {dims}")


def insert_indent(line_list, indent):
    """Re-indent a multi-line block so embedded newlines pick up *indent*."""
    return line_list.replace("\n", indent)


def extract_func_lines(func):
    """Split a multi-line ``data_transform`` body into (stmts, final expr).

    The final non-empty line is treated as the expression that defines the
    reference tensor; preceding lines are intermediate statements (assigns,
    etc.) that run before the assignment.
    """
    lines = func.split('\n')
    id = len(lines) - 1
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip():
            id = i
            break
    intermediate_lines = lines[:id]
    result_line = lines[id]
    return intermediate_lines, result_line


def _torch_shape_for_dims(dims):
    """PCL-lite ``dims`` are innermost-first; torch shape is reversed."""
    return list(reversed(dims))


def torch_data_init(data_gen, dims, input=None):
    """Generate an input/parameter init expression from the YAML spec.

    ``dims`` is the PCL-lite list (innermost-first); we reverse it so the
    torch tensor's outermost dim corresponds to the leading shape entry.
    Symbols become ``<sym>_value`` (referencing the constants in *prefix*).
    """
    torch_dims = _torch_shape_for_dims(dims)
    shape_args = ", ".join(_value_of(d) for d in torch_dims)
    if data_gen in ("torch.rand", "torch.randn"):
        return f"{data_gen}({shape_args})"
    if data_gen == "torch.ones":
        return f"{data_gen}(({shape_args}), dtype=torch.float)"
    if data_gen == "binary":
        return f"generate_binary_tensor(({shape_args}))"

    dtype = (input or {}).get("dtype", {})
    assert "Multihot" in str(dtype), (
        f"Unsupported data_gen={data_gen!r} for dtype={dtype!r}; only "
        f"torch.rand[n], torch.ones, binary, and Multihot are recognised."
    )
    match = re.search(r"Multihot\((\w+),\s*(\w+)\)", dtype)
    assert match, f"Cannot decode Multihot dtype {dtype!r}"
    scalar_dtype, num_classes_symbol = match.group(1), match.group(2)
    assert scalar_dtype == "fp32", f"Only fp32 Multihot supported; got {scalar_dtype}"
    return (
        f"generate_multi_hot(({shape_args}),"
        f" {input['min']}, {input['max']}, {num_classes_symbol}_value)"
    )


def _load_params(dims):
    """Default LinearOffChipLoad params for a PCL-lite input.

    ``dims`` is innermost-first (e.g. ``[N, K, M]`` -> torch shape
    ``(M, K, N)``). The default tiling is ``tile_row=1, tile_col=<inner>``
    which makes every benchmark's input a stream of small tiles that the
    LLM's body can reshape on top of.

    Returns (underlying_expr_suffix, stride_expr, out_shape_tiled_expr,
    tile_row, tile_col). ``underlying_expr_suffix`` is "" by default; for
    1-D inputs it's a ``.reshape(1, K_value)`` so LinearOffChipLoad sees a
    2-D tensor (its underlying must have rank >= 2).
    """
    torch_dims = _torch_shape_for_dims(dims)  # outermost-first
    assert len(torch_dims) >= 1, f"empty dims {dims!r}"

    # 1-D inputs: LinearOffChipLoad requires a 2-D underlying. View as
    # (1, K) so the inner dim is still tile_col-covered and the outer dim
    # contributes a singleton to out_shape_tiled.
    if len(torch_dims) == 1:
        inner = torch_dims[0]
        underlying_suffix = f".reshape(1, {_value_of(inner)})"
        out_expr = "(1, 1)"
        stride_expr = "(1, 1)"
        return underlying_suffix, stride_expr, out_expr, "1", _value_of(inner)

    inner = torch_dims[-1]

    # out_shape_tiled mirrors the torch shape but the innermost dim collapses
    # to 1 because tile_col covers the whole inner row.
    out_shape_tiled = torch_dims[:-1] + ["1"]
    # stride is row-major over out_shape_tiled (advancing one tile in dim i
    # jumps prod(out_shape_tiled[i+1:]) tile slots in flat order). The
    # innermost out_shape_tiled entry is always "1" because tile_col covers
    # the whole inner row — drop those 1s from the product to keep the
    # generated source readable.
    stride = []
    for i in range(len(out_shape_tiled)):
        factors = [_value_of(d) for d in out_shape_tiled[i + 1:] if d != "1"]
        stride.append("*".join(factors) if factors else "1")
    out_factors = [_value_of(d) for d in out_shape_tiled]
    trailing_comma = "," if len(out_shape_tiled) == 1 else ""
    stride_expr = "(" + ", ".join(stride) + trailing_comma + ")"
    out_expr = "(" + ", ".join(out_factors) + trailing_comma + ")"
    tile_col_expr = _value_of(inner)
    return "", stride_expr, out_expr, "1", tile_col_expr


def _value_of(d):
    """Symbol/int dim -> Python source referencing the _value constant."""
    if d == "1" or d == 1:
        return "1"
    return f"{d}_value"


# ---------------------------------------------------------------------------
# Main codegen
# ---------------------------------------------------------------------------


def _input_table(inputs):
    """Render a one-line-per-input comment block describing the raw torch
    tensors that ``body()`` receives. Surfaces dtype + dim list so the LLM
    knows which source op to apply (e.g. fp32/uint64 -> LinearOffChipLoad;
    Multihot -> SelectGen)."""
    if not inputs:
        return ""
    lines = ["# Inputs (raw torch tensors passed positionally to body()):"]
    for inp in inputs:
        name = inp.get("name", "")
        dtype = inp.get("dtype", "?")
        dims = inp.get("dims", [])
        dims_str = "[" + ", ".join(str(d) for d in dims) + "]"
        lines.append(f"#   {name}: dtype={dtype}, dims={dims_str}")
    return "\n".join(lines) + "\n"


def yaml_to_code(data):
    inputs = data.get("inputs", [])
    parameters = data.get("parameters", [])
    outputs = data.get("outputs", [])

    # ---- input_data dict (inputs + parameters share this dict) ----
    data_dict = {}
    input_names = []
    for inp in inputs:
        name = inp.get("name", "")
        dims = replace_one_with_str(inp.get("dims", []))
        data_dict[name] = torch_data_init(inp.get("data_gen", ""), dims, inp)
        input_names.append(name)
    for param in parameters:
        name = param.get("name", "")
        dims = replace_one_with_str(param.get("dims", []))
        data_dict[name] = torch_data_init(param.get("data_gen", ""), dims, param)

    data_dict_str = "input_data = {\n"
    for key, value in data_dict.items():
        data_dict_str += f"    '{key}': {value},\n"
    data_dict_str += "}"

    listof_input_names = ", ".join(input_names)
    body_call_args = ", ".join(f"input_data['{n}']" for n in input_names)

    # ---- reference computation + compare (one block per output) ----
    ref_lines = []
    compare_lines = []
    n_outputs = len(outputs)
    for i, output in enumerate(outputs):
        name = output.get("name", "")
        data_transforms = output.get("data_transform", []) or []
        assert len(data_transforms) == 1, (
            f"Output {name!r} has {len(data_transforms)} data_transforms; "
            f"the new codegen pairs each OffChipStore with exactly one ref. "
            f"Split into multiple outputs in the YAML if needed."
        )
        intermediate, result_line = extract_func_lines(data_transforms[0])
        intermediate_block = "\n    ".join([l for l in intermediate if l.strip()])
        if intermediate_block:
            ref_lines.append(f"    {intermediate_block}")
        ref_lines.append(f"    {name}_ref = {result_line.strip()}")

        out_var = "out" if n_outputs == 1 else f"out[{i}]"
        compare_lines.append(
            f"    assert {out_var}.numel() == {name}_ref.numel(), "
            f"f'output {name} numel {{{out_var}.numel()}} != ref {{{name}_ref.numel()}}'"
        )
        compare_lines.append(
            f"    torch.testing.assert_close({out_var}.flatten(), {name}_ref.flatten())"
        )

    ref_block = "\n".join(ref_lines) if ref_lines else "    # no outputs"
    compare_block = "\n".join(compare_lines) if compare_lines else ""

    # ---- test() entrypoint ----
    test_str = (
        "def test():\n"
        "    graph = step.Graph()\n"
        f"    out = body(graph, {body_call_args})\n"
        "    out = step.execute(graph, out, input_tensors={})\n"
        f"{ref_block}\n"
    )
    if compare_block:
        test_str += compare_block + "\n"

    # ---- global stmts (YAML-level top-of-file injections) ----
    global_stmts = data.get("global", "")
    global_stmts_str = (
        f"\n{insert_indent(global_stmts, chr(10))}\n" if global_stmts else ""
    )

    # ---- body() — LLM-provided impl. body args are raw torch tensors; the
    # impl picks the appropriate source op per input.
    impl = data.get("impl", "")
    if impl:
        input_table = _input_table(inputs)
        impl_str = (
            f"{input_table}"
            f"def body(graph, {listof_input_names}):\n"
            f"    {insert_indent(impl, chr(10) + '    ')}\n"
        )
    else:
        impl_str = ""

    return reduce(
        lambda x, y: x + "\n" + y,
        [
            _prefix_for(data),
            global_stmts_str,
            data_dict_str,
            test_str,
            impl_str,
        ],
    )


# ---------------------------------------------------------------------------
# Auxiliary modes (plan, decode, deyaml)
# ---------------------------------------------------------------------------


def yaml_plan_to_code(data):
    """Plan-mode test: only sanity-checks the reference tensor's shape.

    No STeP graph is built or executed; this is used by ``scripts/validate.sh``
    to confirm the YAML's data_transform produces tensors of the declared
    output shape before the proposer is allowed to run.
    """
    inputs = data.get("inputs", [])
    parameters = data.get("parameters", [])

    data_dict = {}
    for inp in inputs:
        name = inp.get("name", "")
        dims = replace_one_with_str(inp.get("dims", []))
        data_dict[name] = torch_data_init(inp.get("data_gen", ""), dims, inp)
    for param in parameters:
        name = param.get("name", "")
        dims = replace_one_with_str(param.get("dims", []))
        data_dict[name] = torch_data_init(param.get("data_gen", ""), dims, param)

    data_dict_str = "input_data = {\n"
    for key, value in data_dict.items():
        data_dict_str += f"    '{key}': {value},\n"
    data_dict_str += "}"

    check_block = ""
    for output in data.get("outputs", []):
        name = output.get("name", "")
        dims = replace_one_with_str(output.get("dims", []))
        # PCL-lite dims are innermost-first; the ref tensor's torch shape
        # is the reverse.
        torch_dims = _torch_shape_for_dims(dims)
        expected = ", ".join(_value_of(d) for d in torch_dims)
        for i, func in enumerate(output.get("data_transform", []) or []):
            intermediate, result_line = extract_func_lines(func)
            intermediate_block = "\n    ".join(
                [l for l in intermediate if l.strip()]
            )
            if intermediate_block:
                check_block += f"    {intermediate_block}\n"
            check_block += f"    {name}_data_{i} = {result_line.strip()}\n"
            check_block += (
                f"    assert {name}_data_{i}.shape == ({expected},), "
                f"f'expected {name} shape ({expected},) but got "
                f"{{{name}_data_{i}.shape}}'\n"
            )

    test_str = "def test():\n" + (check_block or "    pass\n")

    return reduce(lambda x, y: x + "\n" + y, [_prefix_for(data), data_dict_str, test_str])


def decompose_step_yaml_to_code(input_file, output_dir):
    """Expand each example/helper in a doc YAML into its own .py file."""
    with open(input_file, "r") as f:
        data = yaml.safe_load(f)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ops = data.get("ops", [])
    patterns = data.get("patterns", [])
    for op in ops + patterns:
        op_name = op.get("name", "unknown_op").replace(" ", "_")
        for idx, example in enumerate(op.get("examples", [])):
            output_path = os.path.join(
                output_dir, f"{op_name}_example_{idx+1}.py"
            )
            with open(output_path, "w") as f:
                f.write(yaml_to_code(example))
            print(f"Extracted example {idx+1} for op '{op_name}' to {output_path}")

    for idx, helper in enumerate(data.get("helpers", [])):
        output_path = os.path.join(output_dir, f"helper_{idx+1}.py")
        with open(output_path, "w") as f:
            f.write(yaml_to_code(helper))
        print(f"Extracted helper {idx+1} to {output_path}")


# ---------------------------------------------------------------------------
# YAML pretty-printing helpers (kept exactly as before — used by reinforce/)
# ---------------------------------------------------------------------------


def literal_presenter(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


def flow_list_presenter(dumper, data):
    return dumper.represent_sequence(
        "tag:yaml.org,2002:seq", data, flow_style=True
    )


class literal(str):
    pass


class flow_list(list):
    pass


yaml.add_representer(literal, literal_presenter)
yaml.add_representer(flow_list, flow_list_presenter)


def should_be_flow_list(data):
    if not isinstance(data, list):
        return False
    return all(
        isinstance(x, (int, float))
        or (isinstance(x, str) and "\n" not in x and not isinstance(x, literal))
        for x in data
    )


def convert_to_literal(data):
    if isinstance(data, dict):
        return {key: convert_to_literal(value) for key, value in data.items()}
    elif isinstance(data, list):
        converted_list = [convert_to_literal(item) for item in data]
        has_literal = any(isinstance(x, literal) for x in converted_list)
        if should_be_flow_list(data) and not has_literal:
            return flow_list(converted_list)
        return converted_list
    elif isinstance(data, str):
        if "\n" in data or "input_data" in data:
            return literal(data)
    return data


def decompose_step_yaml(input_file, output_dir):
    with open(input_file, "r") as f:
        data = yaml.safe_load(f)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for op in data.get("ops", []):
        op_name = op.get("name", "unknown_op").replace(" ", "_")
        for idx, example in enumerate(op.get("examples", [])):
            output_path = os.path.join(
                output_dir, f"{op_name}_example_{idx+1}.yaml"
            )
            example_dict = {"examples": [convert_to_literal(example)]}
            with open(output_path, "w") as f:
                yaml.dump(
                    example_dict, f, default_flow_style=False, sort_keys=False,
                    width=float("inf"),
                )
            print(f"Extracted example {idx+1} for op '{op_name}' to {output_path}")

    for idx, helper in enumerate(data.get("helpers", [])):
        output_path = os.path.join(output_dir, f"helper_{idx+1}.yaml")
        helper_dict = {"helpers": [convert_to_literal(helper)]}
        with open(output_path, "w") as f:
            yaml.dump(
                helper_dict, f, default_flow_style=False, sort_keys=False,
                width=float("inf"),
            )
        print(f"Extracted helper {idx+1} to {output_path}")


# ---------------------------------------------------------------------------
# Misc utilities used by reinforce/
# ---------------------------------------------------------------------------


def clean_python_code(code_string):
    """Strip comments and blank lines from Python source while preserving strings."""
    lines = code_string.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        result = ""
        in_string = False
        string_char = None
        i = 0
        while i < len(line):
            char = line[i]
            if char in ['"', "'"] and (i == 0 or line[i - 1] != "\\"):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                result += char
            elif char == "#" and not in_string:
                break
            else:
                result += char
            i += 1
        if result.strip():
            cleaned_lines.append(result.rstrip())
    return "\n".join(cleaned_lines)


def batch_yaml_to_code(task_data, temp_dir, model_name, rounds, prefix):
    for id in range(rounds):
        temp_test_path = os.path.join(temp_dir, f"test_{id}_{model_name}.py")
        impl_path = os.path.join(temp_dir, f"{prefix}_{id}.yaml")
        if not os.path.exists(impl_path):
            continue
        with open(impl_path, "r") as f:
            impl_data = yaml.safe_load(f.read())
        data = {**task_data, **impl_data}
        code = yaml_to_code(data)
        with open(temp_test_path, "w") as file:
            file.write(code)


def remove_all_py(temp_dir):
    for root, _, files in os.walk(temp_dir):
        for file in files:
            if file.endswith(".py"):
                os.remove(os.path.join(root, file))


def clean_model_name(model_name):
    return model_name.replace(":", "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", help="The yaml file to convert")
    parser.add_argument("--output", help="The output file to write to")
    parser.add_argument("--mode", help="Mode to use", default="single")
    args = parser.parse_args()

    if args.mode == "single":
        with open(args.yaml, "r") as file:
            data = yaml.safe_load(file.read())
        with open(args.output, "w") as file:
            file.write(yaml_to_code(data))
    elif args.mode == "decode":
        decompose_step_yaml_to_code(args.yaml, args.output)
    elif args.mode == "deyaml":
        decompose_step_yaml(args.yaml, args.output)
    elif args.mode == "plan":
        with open(args.yaml, "r") as file:
            data = yaml.safe_load(file.read())
        with open(args.output, "w") as file:
            file.write(yaml_plan_to_code(data))
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
