"""Generate prompts/proposer_base.yaml and prompts/proposer_base_ws.yaml from
step_tl op docstrings + hand-curated examples.

Source-of-truth pipeline:
  - step_tl/src/timing_and_emulator/functional.py        (_exec_X docstring)
  - step_tl/src/step_py/ops.py                           (class docstring +
                                                          __init__ signature)
  - PCL-lite/prompts/_prompt_skeleton.yaml               (preamble, op order,
                                                          patterns)
  - PCL-lite/prompts/_op_examples.yaml                   (examples)

Run from the PCL-lite root:
    python tools/build_prompts.py

The two `proposer_base*.yaml` files are build artifacts. Don't hand-edit
them; edit the sources above and re-run this script.
"""

import argparse
import ast
import os
import re
import sys

import yaml

# Make the in-tree tools package importable when run as a script.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.yaml_to_code import (
    convert_to_literal,
    literal,
    literal_presenter,
    flow_list,
    flow_list_presenter,
    yaml_to_code,
)


_STEP_TL_SRC = os.path.normpath(
    os.path.join(_REPO_ROOT, "..", "step_tl", "src")
)
_FUNCTIONAL_PATH = os.path.join(
    _STEP_TL_SRC, "timing_and_emulator", "functional.py"
)
_OPS_PATH = os.path.join(_STEP_TL_SRC, "step_py", "ops.py")
_UTILITY_OPS_PATH = os.path.join(_STEP_TL_SRC, "step_py", "utility_ops.py")
_SKELETON_PATH = os.path.join(_REPO_ROOT, "prompts", "_prompt_skeleton.yaml")
_EXAMPLES_PATH = os.path.join(_REPO_ROOT, "prompts", "_op_examples.yaml")
_OUT_STRUCTURED = os.path.join(_REPO_ROOT, "prompts", "proposer_base.yaml")
_OUT_WS = os.path.join(_REPO_ROOT, "prompts", "proposer_base_ws.yaml")


def _camel_to_snake(name: str) -> str:
    """`RepeatStatic` -> `repeat_static`."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _extract_isinstance_classes(test):
    """Return the list of class names referenced by `isinstance(node, X)` or
    `isinstance(node, (A, B, ...))`. Empty list for any other test."""
    if not (isinstance(test, ast.Call)
            and getattr(test.func, "id", None) == "isinstance"
            and len(test.args) == 2):
        return []
    cls_arg = test.args[1]
    if isinstance(cls_arg, ast.Name):
        return [cls_arg.id]
    if isinstance(cls_arg, ast.Tuple):
        return [e.id for e in cls_arg.elts if isinstance(e, ast.Name)]
    return []


def _find_exec_call_name(if_body):
    """If the first statement of an `if isinstance(...)` body is
    `return _exec_X(...)`, return the called function's name; else None."""
    for stmt in if_body:
        if not isinstance(stmt, ast.Return):
            continue
        if not isinstance(stmt.value, ast.Call):
            return None
        func = stmt.value.func
        if isinstance(func, ast.Name) and func.id.startswith("_exec_"):
            return func.id
        return None
    return None


def _load_dispatch_mapping():
    """Parse the `_dispatch` function in functional.py to map each step_tl
    op class to the name of its `_exec_*` handler (or None for inline /
    identity dispatch).

    Built by AST-walking the body for `if isinstance(node, X):` (and tuple
    forms `isinstance(node, (A, B))`), so it stays in sync with the actual
    dispatch table — no snake_case heuristics that break on names like
    `LinearOffChipLoadRef -> _exec_load_ref`.
    """
    tree = ast.parse(open(_FUNCTIONAL_PATH).read())
    dispatch_fn = next(
        (n for n in tree.body
         if isinstance(n, ast.FunctionDef) and n.name == "_dispatch"),
        None,
    )
    assert dispatch_fn is not None, (
        "Expected a top-level `_dispatch` function in functional.py; "
        "the dispatch parser needs updating."
    )
    out = {}
    for stmt in dispatch_fn.body:
        if not isinstance(stmt, ast.If):
            continue
        classes = _extract_isinstance_classes(stmt.test)
        if not classes:
            continue
        exec_name = _find_exec_call_name(stmt.body)
        for cls in classes:
            out[cls] = exec_name  # None means inline dispatch
    return out


def _load_exec_docstrings():
    """Map `_exec_*` function-name -> docstring extracted from functional.py
    (only functions that actually carry a docstring)."""
    tree = ast.parse(open(_FUNCTIONAL_PATH).read())
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if not node.name.startswith("_exec_"):
            continue
        doc = ast.get_docstring(node)
        if doc:
            out[node.name] = doc.strip()
    return out


def _load_class_info():
    """Map class-name -> (class_docstring or None, __init__ signature str)
    pulled from step_tl's ops.py and utility_ops.py.

    Signature format mimics what the existing hand-written prompts use:
        step.OpName(arg0, arg1=<...>, ...)
    where positional args appear as bare names and keyword args (anything
    with a default) appear as `name=<...>`. The first parameter (`self`)
    and any `graph: MultiDiGraph` parameter are dropped — `graph` is always
    threaded by the harness, not by the LLM.
    """
    out = {}
    for path in (_OPS_PATH, _UTILITY_OPS_PATH):
        tree = ast.parse(open(path).read())
        _ingest_classes(tree, out)
    return out


def _ingest_classes(tree, out):
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        # locate __init__
        init = next(
            (
                n for n in node.body
                if isinstance(n, ast.FunctionDef) and n.name == "__init__"
            ),
            None,
        )
        if init is None:
            continue
        args = init.args
        n_args = len(args.args)
        n_defaults = len(args.defaults)
        default_offset = n_args - n_defaults
        parts = []
        for i, a in enumerate(args.args):
            if a.arg == "self":
                continue
            has_default = i >= default_offset
            if has_default:
                parts.append(f"{a.arg}=<...>")
            else:
                parts.append(a.arg)
        sig = f"step.{node.name}({', '.join(parts)})"
        out[node.name] = (ast.get_docstring(node), sig)
    return out


def _desc_for(class_name, exec_docs, class_info, dispatch_map):
    """Return the description body for a single op class (no Signature line).

    Lookup order: `_exec_X` docstring (where X is the actual handler name
    from the dispatch table) → class docstring in ops.py → AssertionError.
    """
    handler = dispatch_map.get(class_name)
    if handler and handler in exec_docs:
        return exec_docs[handler]
    class_doc, _ = class_info.get(class_name, (None, None))
    if class_doc:
        return class_doc.strip()
    handler_hint = (
        f"`{handler}` in step_tl/.../functional.py"
        if handler else
        f"`class {class_name}` in step_tl/.../ops.py (inline dispatch)"
    )
    raise AssertionError(
        f"No docstring found for op {class_name!r}. Add one to "
        f"{handler_hint}, or remove the op from op_order."
    )


def _build_op_desc(entry, exec_docs, class_info, dispatch_map):
    """Render the full per-op `desc:` body (semantic doc + Signature line(s))
    for one entry from the skeleton's `op_order`."""
    if isinstance(entry, str):
        body = _desc_for(entry, exec_docs, class_info, dispatch_map)
        _, sig = class_info[entry]
        return f"{body}\nSignature: {sig}"

    # combined entry
    assert isinstance(entry, dict) and "classes" in entry and "name" in entry, (
        f"op_order entry must be a string or {{name, classes, extra_desc?}} "
        f"dict, got {entry!r}"
    )
    paragraphs = []
    sig_lines = []
    for cls in entry["classes"]:
        paragraphs.append(_desc_for(cls, exec_docs, class_info, dispatch_map))
        _, sig = class_info[cls]
        sig_lines.append(f"Signature: {sig}")
    if entry.get("extra_desc"):
        paragraphs.append(entry["extra_desc"].rstrip())
    paragraphs.append("\n".join(sig_lines))
    return "\n\n".join(paragraphs)


def _entry_name_and_key(entry):
    """Return (display name, examples-lookup key) for an op_order entry."""
    if isinstance(entry, str):
        return entry, entry
    return entry["name"], entry["name"]


def _ws_example_text(example_stanza):
    """Codegen one example stanza to its full Python test scaffold."""
    return yaml_to_code(example_stanza).strip() + "\n"


# ---------------------------------------------------------------------------
# Prompt-file emitters
# ---------------------------------------------------------------------------

# YAML quirks: when a literal-block scalar contains trailing whitespace on a
# line, yaml.dump silently switches to a quoted style and the output becomes
# unreadable. Strip trailing spaces from any string before handing it to
# yaml.dump.
def _strip_trailing_spaces(s):
    return "\n".join(line.rstrip() for line in s.splitlines()) + (
        "\n" if s.endswith("\n") else ""
    )


def _as_literal(s):
    return literal(_strip_trailing_spaces(s))


def _ops_block_structured(skeleton, examples, exec_docs, class_info,
                          dispatch_map):
    out = []
    for entry in skeleton["op_order"]:
        name, key = _entry_name_and_key(entry)
        desc = _build_op_desc(entry, exec_docs, class_info, dispatch_map)
        stanza = {"name": name, "desc": _as_literal(desc + "\n")}
        ex_list = examples.get(key)
        if ex_list:
            stanza["examples"] = [convert_to_literal(ex) for ex in ex_list]
        out.append(stanza)
    return out


def _ops_block_ws(skeleton, examples, exec_docs, class_info, dispatch_map):
    out = []
    for entry in skeleton["op_order"]:
        name, key = _entry_name_and_key(entry)
        desc = _build_op_desc(entry, exec_docs, class_info, dispatch_map)
        stanza = {"name": name, "desc": _as_literal(desc + "\n")}
        ex_list = examples.get(key)
        if ex_list:
            stanza["examples"] = [
                _as_literal(_ws_example_text(ex)) for ex in ex_list
            ]
        out.append(stanza)
    return out


def _patterns_block_structured(skeleton, examples, mode):
    out = []
    for pat in skeleton.get("patterns", []):
        key = pat.get("example_key", pat["name"])
        ex_list = examples.get(key)
        assert ex_list, f"No example for pattern {key!r} in _op_examples.yaml."
        if mode == "structured":
            ex_blocks = [convert_to_literal(ex) for ex in ex_list]
        else:
            ex_blocks = [_as_literal(_ws_example_text(ex)) for ex in ex_list]
        out.append({
            "name": pat["name"],
            "desc": _as_literal(pat["desc"]),
            "examples": ex_blocks,
        })
    return out


def _dump(doc, path):
    # Make sure our literal/flow representers are registered. yaml_to_code
    # already registered them at import time, but be defensive.
    yaml.add_representer(literal, literal_presenter)
    yaml.add_representer(flow_list, flow_list_presenter)

    header = (
        "# Generated by tools/build_prompts.py. Do not hand-edit.\n"
        "# Sources of truth:\n"
        "#   - prompts/_prompt_skeleton.yaml\n"
        "#   - prompts/_op_examples.yaml\n"
        "#   - step_tl/.../functional.py (_exec_X docstrings)\n"
        "#   - step_tl/.../ops.py        (class docstrings + __init__ signatures)\n"
    )
    body = yaml.dump(
        doc,
        default_flow_style=False,
        sort_keys=False,
        width=float("inf"),
        allow_unicode=True,
    )
    with open(path, "w") as f:
        f.write(header + body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Build into temp paths and diff against the on-disk versions; "
             "exit nonzero if they differ. Used by CI to enforce that the "
             "generated prompts are in sync with the sources.",
    )
    args = parser.parse_args()

    skeleton = yaml.safe_load(open(_SKELETON_PATH))
    examples = yaml.safe_load(open(_EXAMPLES_PATH))
    exec_docs = _load_exec_docstrings()
    class_info = _load_class_info()
    dispatch_map = _load_dispatch_mapping()

    # In structured mode the prompt has separate `ops:` and `patterns:`
    # sections; in ws mode everything is folded into `ops:` (matching the
    # current hand-maintained layout of proposer_base_ws.yaml).
    structured_doc = {
        "general": {"desc": _as_literal(skeleton["general"]["desc"])},
        "function_inventory": {
            "desc": _as_literal(skeleton["function_inventory"]["desc"])
        },
        "ops": _ops_block_structured(
            skeleton, examples, exec_docs, class_info, dispatch_map
        ),
        "patterns": _patterns_block_structured(skeleton, examples, "structured"),
    }
    ws_doc = {
        "general": {"desc": _as_literal(skeleton["general"]["desc"])},
        "function_inventory": {
            "desc": _as_literal(skeleton["function_inventory"]["desc"])
        },
        "ops": (
            _ops_block_ws(
                skeleton, examples, exec_docs, class_info, dispatch_map
            )
            + _patterns_block_structured(skeleton, examples, "ws")
        ),
    }

    if args.check:
        import io
        for doc, path in [(structured_doc, _OUT_STRUCTURED),
                          (ws_doc, _OUT_WS)]:
            buf = io.StringIO()
            yaml.add_representer(literal, literal_presenter)
            yaml.add_representer(flow_list, flow_list_presenter)
            buf.write(
                "# Generated by tools/build_prompts.py. Do not hand-edit.\n"
                "# Sources of truth:\n"
                "#   - prompts/_prompt_skeleton.yaml\n"
                "#   - prompts/_op_examples.yaml\n"
                "#   - step_tl/.../functional.py (_exec_X docstrings)\n"
                "#   - step_tl/.../ops.py        (class docstrings + __init__ signatures)\n"
            )
            buf.write(yaml.dump(
                doc, default_flow_style=False, sort_keys=False,
                width=float("inf"), allow_unicode=True,
            ))
            on_disk = open(path).read() if os.path.exists(path) else ""
            if buf.getvalue() != on_disk:
                print(f"OUT OF DATE: {os.path.relpath(path, _REPO_ROOT)} "
                      f"differs from build output. Run "
                      f"`python tools/build_prompts.py` to regenerate.",
                      file=sys.stderr)
                sys.exit(1)
        print("prompts are in sync with sources.")
        return

    _dump(structured_doc, _OUT_STRUCTURED)
    _dump(ws_doc, _OUT_WS)
    print(f"wrote {os.path.relpath(_OUT_STRUCTURED, _REPO_ROOT)}")
    print(f"wrote {os.path.relpath(_OUT_WS, _REPO_ROOT)}")


if __name__ == "__main__":
    main()
