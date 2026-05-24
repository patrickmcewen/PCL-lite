import ast
import os
import time
import yaml
import pytest
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from tools.utils import query_anthropic, extract_code, query, clean_yaml, clean_code, query_text, print_elapsed_time
from tools import yaml_to_code, ast_analysis, count_usage
from reinforce.base import Agent, Storage


# --- Static "no torch-bypass" gate ---------------------------------------
# Source-of-truth list of step.* ops that legitimately consume the body's
# raw tensor inputs. Any other use of an input tensor (passing it to
# torch math, the `@` operator, helper functions, etc.) is a bypass —
# the model is computing the gold answer in plain torch and wrapping the
# result in a source op so the harness's identity load→store pipes a
# precomputed tensor straight through. View-creating ops (`.t()`,
# `.reshape(...)`, `[p]`) are permitted *inside* a source-op kwarg.
_SOURCE_OPS = frozenset({
    "LinearOffChipLoad",
    "RandomOffChipLoad",
    "DynLinearOffChipLoad",
    "SelectGen",
})
_SOURCE_KWARGS = frozenset({"underlying", "tensor"})


def _attach_parents(tree):
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child._parent = node


def _walk_no_nested(node):
    """ast.walk but does not descend into nested function defs / lambdas
    so legitimate ``input_data[...]`` access inside map_fn closures isn't
    treated as a body-level use."""
    yield node
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        yield from _walk_no_nested(child)


def _is_source_op_call(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _SOURCE_OPS
    )


def _classify_input_use(node):
    """Walk up from an input reference; True iff the chain terminates at
    an ``underlying=``/``tensor=`` kwarg of a source-op call, traversing
    attribute access, method calls, and subscript indexing (so
    ``Ei.reshape(...)`` and ``Ei[p]`` view ops inside the kwarg are
    allowed). Any input reference *inside* the subscript's index gets
    its own bypass check via ``_walk_no_nested``."""
    cur = node
    while True:
        parent = getattr(cur, "_parent", None)
        if parent is None:
            return False
        if isinstance(parent, ast.Attribute) and parent.value is cur:
            cur = parent
            continue
        if isinstance(parent, ast.Call) and parent.func is cur:
            cur = parent
            continue
        if isinstance(parent, ast.Subscript) and parent.value is cur:
            cur = parent
            continue
        if isinstance(parent, ast.keyword) and parent.value is cur:
            return (
                parent.arg in _SOURCE_KWARGS
                and _is_source_op_call(getattr(parent, "_parent", None))
            )
        return False


def _input_ref_load(node, input_set):
    """Return the input name if ``node`` is a Load-context reference to an
    input tensor — either bare ``Ei`` or ``input_data['Ei']`` — else None."""
    if (isinstance(node, ast.Name)
            and isinstance(getattr(node, "ctx", None), ast.Load)
            and node.id in input_set):
        return node.id
    if (isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "input_data"):
        idx = node.slice
        if isinstance(idx, ast.Constant) and idx.value in input_set:
            return idx.value
    return None


def _rhs_rebinds_input(rhs, name):
    """True iff ``rhs`` is a source-op call whose ``underlying=``/
    ``tensor=`` value references ``name`` (the standard rebind-input-as-
    stream pattern: ``A = step.LinearOffChipLoad(underlying=A, ...)``)."""
    if not _is_source_op_call(rhs):
        return False
    for kw in rhs.keywords:
        if kw.arg in _SOURCE_KWARGS:
            for n in ast.walk(kw.value):
                if isinstance(n, ast.Name) and n.id == name:
                    return True
    return False


def check_body_no_bypass(body_src, input_names):
    """Reject bodies that bypass STeP by computing the answer in plain
    torch and wrapping the final tensor in a source op. Returns ``None``
    if clean, else a one-line message describing the first violation."""
    indented = "\n".join("    " + line for line in body_src.split("\n"))
    wrapped = "def __body__():\n" + indented + "\n    pass\n"
    try:
        tree = ast.parse(wrapped)
    except SyntaxError as e:
        return f"body does not parse: {e}"
    _attach_parents(tree)
    input_set = set(input_names)
    name_rebound = set()
    for stmt in tree.body[0].body:
        new_rebinds = set()
        if (isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id in input_set
                and _rhs_rebinds_input(stmt.value, stmt.targets[0].id)):
            new_rebinds.add(stmt.targets[0].id)
        for node in _walk_no_nested(stmt):
            name = _input_ref_load(node, input_set)
            if name is None:
                continue
            if isinstance(node, ast.Name) and name in name_rebound:
                continue
            if not _classify_input_use(node):
                # Subtract 1 for the synthetic `def __body__():` wrapper line.
                lineno = max(node.lineno - 1, 1)
                return (
                    f"input `{name}` used outside a source-op load at body "
                    f"line {lineno}. Input tensors must be consumed as "
                    "`underlying=` / `tensor=` of step.LinearOffChipLoad / "
                    "SelectGen / RandomOffChipLoad / DynLinearOffChipLoad — "
                    "all compute must go through step.* ops."
                )
        name_rebound |= new_rebinds
    return None


_RESPONSES_SUBDIR = "responses"


def _split_message_content(content):
    """Return (reasoning_text, response_text) for either a plain string or
    an Anthropic-style content-block list. ``thinking`` blocks become
    reasoning; ``text`` blocks become response."""
    if content is None:
        return "", ""
    if isinstance(content, str):
        return "", content
    reasoning_parts, text_parts = [], []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "thinking" or "thinking" in block:
                reasoning_parts.append(block.get("thinking", ""))
            else:
                text_parts.append(block.get("text", ""))
        else:
            text_parts.append(str(block))
    return ("\n\n".join(p for p in reasoning_parts if p),
            "\n\n".join(p for p in text_parts if p))


def log_sample_response(temp_dir, sample_id, message):
    """Write the LLM reasoning + raw response for one sample to disk.

    Called immediately after the LLM returns so the trace is preserved
    even when downstream parsing/pytest raises. The matching
    `log_sample_outcome` call appends pass/fail + generated test code.
    Returns the sample log path so the caller can hand it back in.
    """
    sample_dir = os.path.join(temp_dir, _RESPONSES_SUBDIR)
    os.makedirs(sample_dir, exist_ok=True)
    path = os.path.join(sample_dir, f"sample_{sample_id}.md")
    message = message or {}
    block_reasoning, content = _split_message_content(message.get("content"))
    reasoning = (
        message.get("reasoning_content")
        or message.get("reasoning")
        or block_reasoning
    )
    sections = [f"# Sample {sample_id}", ""]
    if reasoning:
        sections += ["## Reasoning", "", str(reasoning).rstrip(), ""]
    sections += ["## Raw response", "", content.rstrip(), ""]
    with open(path, "w") as f:
        f.write("\n".join(sections))
    return path


def log_sample_outcome(path, passed, generated_code=None, error=None):
    """Append pass/fail + (optionally) generated test code to a sample log."""
    if not path or not os.path.exists(path):
        return
    with open(path, "a") as f:
        f.write(f"\n## Outcome\n\n{'PASS' if passed else 'FAIL'}\n")
        if error is not None:
            f.write(f"\n## Error\n\n```\n{error}\n```\n")
        if generated_code is not None:
            f.write("\n## Generated test code\n\n```python\n")
            f.write(generated_code)
            f.write("\n```\n")

def prepare_prompt_query_impl(key, config: Storage, storage: Storage):
    config_data = config.retrieve(key)
    storage_data = storage.retrieve(key)
    example_path = config_data["example_path"]
    with open(example_path, "r") as f:
        example = f.read()
        storage_data["example"] = example
    task_data = storage_data["task"]
    task = yaml.dump(yaml_to_code.convert_to_literal(task_data), default_flow_style=False, sort_keys=False, width=float("inf"))
    prompt = f"""
```
{example}
```
Given the above example, complete the `impl` for the test
```
{task}
```
Please output the code in this format:   
```
impl: |-
``` 
"""     
    storage_data["prompt"] = prompt
    return prompt

def prepare_prompt_query_impl_with_feedback(key, config: Storage, storage: Storage):
    config_data = config.retrieve(key)
    storage_data = storage.retrieve(key)
    example_path = config_data["example_path"]
    with open(example_path, "r") as f:
        example = f.read()
        storage_data["example"] = example
    task_data = storage_data["task"]
    feedback_data = storage_data["feedback"]
    task = yaml.dump(yaml_to_code.convert_to_literal(task_data), default_flow_style=False, sort_keys=False, width=float("inf"))
    feedback = yaml.dump(yaml_to_code.convert_to_literal(feedback_data), default_flow_style=False, sort_keys=False, width=float("inf"))
    prompt = f"""
```
{example}
```
Given the above example, complete the `impl` for the test
```
{task}

Feedback:
{feedback}
```
Please output the code in this format:   
```
impl: |-
``` 
"""     
    storage_data["prompt"] = prompt
    return prompt

def prepare_prompt_eliminate_identity(key, config: Storage, storage: Storage):
    storage_data = storage.retrieve(key)
    ex_impl_data = storage_data["ex_impl"]
    ex_impl = yaml.dump(yaml_to_code.convert_to_literal(ex_impl_data), default_flow_style=False, sort_keys=False, width=float("inf"))
    prompt = f"""

This program pattern:
```
Ex = ...
Ey = step.Bufferize(a=?).apply(Ex)
Ez = step.Streamify().apply(Ey)
``` can be simplified to:
```
Ez = ...
```
because Bufferize followed by a Streamify is an identity operation when Ex's element type is `step.Scalar`. 
Please detect this pattern in the below impl and simplify it.
```
{ex_impl}
```
Please output the code in this format:   
```
impl: |-
``` 
"""     
    storage_data["prompt"] = prompt
    return prompt

def single_query_impl(id, prompt, task_data, config_data):
    temp_dir = config_data["temp_dir"]
    model_name = config_data["model_name"]
    temperature = config_data["temperature"]
    max_tokens = config_data["max_tokens"]
    system_prompt = config_data["system_prompt"]
    temp = {"usage": {}}
    response = query_text(prompt, temperature=temperature, model_name=model_name, system=system_prompt, max_tokens=max_tokens, storage=temp)
    sample_log = log_sample_response(temp_dir, id, temp.get("message"))
    impl = extract_code(response)
    impl = clean_yaml(impl)
    temp_test_path = os.path.join(temp_dir, f"test_{id}_{yaml_to_code.clean_model_name(model_name)}.py")
    impl_data = yaml.safe_load(impl)
    # Same bypass gate as single_query_affine_impl — see check_body_no_bypass.
    input_names = [inp["name"] for inp in task_data.get("inputs", [])]
    bypass = check_body_no_bypass(impl_data.get("impl", ""), input_names)
    if bypass:
        log_sample_outcome(sample_log, False, error=f"Bypass: {bypass}")
        return (pytest.ExitCode.TESTS_FAILED, impl_data, temp["usage"])
    data = {**task_data, **impl_data}
    code = yaml_to_code.yaml_to_code(data)
    with open(temp_test_path, 'w') as file:
        file.write(code)
    result = pytest.main([temp_test_path], plugins=[])
    log_sample_outcome(sample_log, result == pytest.ExitCode.OK, generated_code=code)
    os.remove(temp_test_path) # Comment this line to keep the test file when debugging
    return (result, impl_data, temp["usage"])

class QueryImpl(Agent):
    def __init__(self, key, config: Storage):
        super().__init__(key, config)
    
    def run(self, storage: Storage):
        config_data = self.config.retrieve(self.key)
        num_samples = config_data["num_samples"]
        num_workers = config_data["num_workers"]
        storage_data = storage.retrieve(self.key)
        task_data = storage_data["task"]        
        prompt = storage_data["prompt"] # self.prepare_prompt(storage)
        success = []
        failure = []
        usages = []
        start = time.time()
        with tqdm(total=num_samples, smoothing=0) as pbar:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(
                        single_query_impl,
                        id,
                        prompt,
                        task_data,
                        config_data
                    )
                    for id in range(num_samples)
                }
                for future in as_completed(futures):
                    pbar.update(1)
                    try:
                        result = future.result()
                        if result is not None:
                            if result[0] == pytest.ExitCode.OK:
                                success.append(result[1])
                            else:
                                failure.append(result[1])
                            usages.append(result[2])
                    except Exception as e:
                        print("Got an error!", e)
                        continue
        end = time.time()
        print_elapsed_time(start, end)
        storage_data["success"] = success
        storage_data["failure"] = failure # Stores a list of failed impls (not task+impl)
        storage_data["usage"] = usages
    
    def deduplicate_failure(self, storage: Storage):
        storage_data = storage.retrieve(self.key)
        failure = storage_data["failure"]
        task_data = storage_data["task"]
        code_history = ast_analysis.EquivalentSet(ast_analysis.check_program_equivalence)
        rep_list = []
        rep_impl_list = []
        for (id, impl_data) in enumerate(failure):
            data = {**task_data, **impl_data}
            code = yaml_to_code.yaml_to_code(data)
            code_history.add(code, id)
        for (_, v) in code_history.item_map.items():
            rep_list.append({**failure[v[0]]})
            rep_impl_list.append(failure[v[0]])
        storage_data["failure_rep"] = rep_list
        storage_data["failure_rep_impl"] = rep_impl_list

    def deduplicate_success(self, storage: Storage):
        storage_data = storage.retrieve(self.key)
        failure = storage_data["success"]
        task_data = storage_data["task"]
        code_history = ast_analysis.EquivalentSet(ast_analysis.check_program_equivalence)
        rep_list = []
        rep_impl_list = []
        for (id, impl_data) in enumerate(failure):
            data = {**task_data, **impl_data}
            code = yaml_to_code.yaml_to_code(data)
            code_history.add(code, id)
        for (_, v) in code_history.item_map.items():
            rep_list.append({**failure[v[0]]})
            rep_impl_list.append(failure[v[0]])
        storage_data["success_rep"] = rep_list
        storage_data["success_rep_impl"] = rep_impl_list

def prepare_prompt_query_py_body(key, config: Storage, storage: Storage):
    config_data = config.retrieve(key)
    storage_data = storage.retrieve(key)
    example_path = config_data["example_path"]
    with open(example_path, "r") as f:
        example = f.read()
        storage_data["example"] = example
    task_data = storage_data["task"]
    prompt = f"""
```
{example}
```
Ops define all the primitive you can use. Please implement the `body` function for the test.
```
{task_data}
```
Please output the code in this format:   
```
def body
``` 
"""     
    storage_data["prompt"] = prompt
    return prompt

def single_query_py_body(id, prompt, task_str, config_data):
    temp_dir = config_data["temp_dir"]
    model_name = config_data["model_name"]
    temperature = config_data["temperature"]
    max_tokens = config_data["max_tokens"]
    system_prompt = config_data["system_prompt"]
    temp = {"usage": {}}
    if "claude" in model_name:
        response = query_anthropic(prompt, temperature=temperature, model_name=model_name, system=system_prompt, max_tokens=max_tokens, storage=temp)
        sample_log = log_sample_response(temp_dir, id, temp.get("message"))
        impl = extract_code(response['text'])
    else:
        response = query(prompt, temperature=temperature, model_name=model_name, system=system_prompt, max_tokens=max_tokens, storage=temp)
        sample_log = log_sample_response(temp_dir, id, temp.get("message"))
        impl = extract_code(response)
    impl_str = clean_code(impl)
    temp_test_path = os.path.join(temp_dir, f"test_{id}_{yaml_to_code.clean_model_name(model_name)}.py")
    code = task_str + "\n" + impl_str
    with open(temp_test_path, 'w') as file:
        file.write(code)
    result = pytest.main([temp_test_path], plugins=[])
    log_sample_outcome(sample_log, result == pytest.ExitCode.OK, generated_code=code)
    os.remove(temp_test_path) # Comment this line to keep the test file when debugging
    return (result, impl_str, temp["usage"])

class QueryPyBody(Agent):
    def __init__(self, key, config: Storage):
        super().__init__(key, config)

    def run(self, storage: Storage):
        config_data = self.config.retrieve(self.key)
        num_samples = config_data["num_samples"]
        num_workers = config_data["num_workers"]
        storage_data = storage.retrieve(self.key)
        task_str = storage_data["task"]        
        prompt = storage_data["prompt"]
        success = []
        failure = []
        usages = []
        start = time.time()
        with tqdm(total=num_samples, smoothing=0) as pbar:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(
                        single_query_py_body,
                        id,
                        prompt,
                        task_str,
                        config_data
                    )
                    for id in range(num_samples)
                }
                for future in as_completed(futures):
                    pbar.update(1)
                    try:
                        result = future.result()
                        if result is not None:
                            if result[0] == pytest.ExitCode.OK:
                                success.append(result[1])
                            else:
                                failure.append(result[1])
                            usages.append(result[2])
                    except Exception as e:
                        print("Got an error!", e)
                        continue
        end = time.time()
        print_elapsed_time(start, end)
        storage_data["success"] = success
        storage_data["failure"] = failure # Stores a list of failed impls (not task+impl)
        storage_data["usage"] = usages
    
    def deduplicate_failure(self, storage: Storage):
        storage_data = storage.retrieve(self.key)
        failure = storage_data["failure"]
        task_str = storage_data["task"]
        code_history = ast_analysis.EquivalentSet(ast_analysis.check_program_equivalence)
        rep_list = []
        rep_impl_list = []
        for (id, impl_str) in enumerate(failure):
            code = task_str + "\n" + impl_str
            code_history.add(code, id)
        for (_, v) in code_history.item_map.items():
            rep_list.append(failure[v[0]])
            rep_impl_list.append(failure[v[0]])
        storage_data["failure_rep"] = rep_list
        storage_data["failure_rep_impl"] = rep_impl_list

    def deduplicate_success(self, storage: Storage):
        storage_data = storage.retrieve(self.key)
        failure = storage_data["success"]
        task_str = storage_data["task"]
        code_history = ast_analysis.EquivalentSet(ast_analysis.check_program_equivalence)
        rep_list = []
        rep_impl_list = []
        for (id, impl_str) in enumerate(failure):
            code = task_str + "\n" + impl_str
            code_history.add(code, id)
        for (_, v) in code_history.item_map.items():
            rep_list.append(failure[v[0]])
            rep_impl_list.append(failure[v[0]])
        storage_data["success_rep"] = rep_list
        storage_data["success_rep_impl"] = rep_impl_list

def single_query_affine_py_body(id, prompt, task_str, config_data):
    temp_dir = config_data["temp_dir"]
    model_name = config_data["model_name"]
    temperature = config_data["temperature"]
    max_tokens = config_data["max_tokens"]
    system_prompt = config_data["system_prompt"]
    temp = {"usage": {}}
    response = query_text(prompt, temperature=temperature, model_name=model_name, system=system_prompt, max_tokens=max_tokens, storage=temp)
    sample_log = log_sample_response(temp_dir, id, temp.get("message"))
    impl = extract_code(response)
    impl_str = clean_code(impl)
    temp_test_path = os.path.join(temp_dir, f"test_{id}_{yaml_to_code.clean_model_name(model_name)}.py")
    code = task_str + "\n" + impl_str
    with open(temp_test_path, 'w') as file:
        file.write(code)
    result = pytest.main([temp_test_path], plugins=[])
    usage_all_once = count_usage.check_affine_type(temp_test_path)
    passed = usage_all_once and result == pytest.ExitCode.OK
    log_sample_outcome(sample_log, passed, generated_code=code)
    os.remove(temp_test_path) # Comment this line to keep the test file when debugging
    return (passed, impl_str, temp["usage"])

class QueryAffinePyBody(QueryPyBody):
    def __init__(self, key, config: Storage):
        super().__init__(key, config)

    def run(self, storage: Storage):
        config_data = self.config.retrieve(self.key)
        num_samples = config_data["num_samples"]
        num_workers = config_data["num_workers"]
        storage_data = storage.retrieve(self.key)
        task_str = storage_data["task"]        
        prompt = storage_data["prompt"]
        success = []
        failure = []
        usages = []
        start = time.time()
        with tqdm(total=num_samples, smoothing=0) as pbar:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(
                        single_query_affine_py_body,
                        id,
                        prompt,
                        task_str,
                        config_data
                    )
                    for id in range(num_samples)
                }
                for future in as_completed(futures):
                    pbar.update(1)
                    try:
                        result = future.result()
                        if result is not None:
                            if result[0]:
                                success.append(result[1])
                            else:
                                failure.append(result[1])
                            usages.append(result[2])
                    except Exception as e:
                        print("Got an error!", e)
                        continue
        end = time.time()
        print_elapsed_time(start, end)
        storage_data["success"] = success
        storage_data["failure"] = failure # Stores a list of failed impls (not task+impl)
        storage_data["usage"] = usages

def prepare_prompt_query_affine_rewrite(key, config: Storage, storage: Storage):
    config_data = config.retrieve(key)
    storage_data = storage.retrieve(key)
    example_path = config_data["example_path"]
    with open(example_path, "r") as f:
        example = f.read()
        storage_data["example"] = example
    test_str = storage_data["test"]
    prompt = f"""
```
{example}
```
Based on the above instruction, please add Copy and adjust the stream variables for the `impl` function.
```
{test_str}
```
Please output the code in this format:   
```
impl: |-
``` 
"""     
    storage_data["prompt"] = prompt
    return prompt

def single_query_affine_impl(id, prompt, task_data, config_data):
    temp_dir = config_data["temp_dir"]
    model_name = config_data["model_name"]
    temperature = config_data["temperature"]
    max_tokens = config_data["max_tokens"]
    system_prompt = config_data["system_prompt"]
    temp = {"usage": {}}
    response = query_text(prompt, temperature=temperature, model_name=model_name, system=system_prompt, max_tokens=max_tokens, storage=temp)
    # Persist the raw response + reasoning trace before parsing so the
    # log survives format errors / pytest crashes downstream.
    sample_log = log_sample_response(temp_dir, id, temp.get("message"))
    impl = extract_code(response)
    impl = clean_yaml(impl)
    temp_test_path = os.path.join(temp_dir, f"test_{id}_{yaml_to_code.clean_model_name(model_name)}.py")
    impl_data = yaml.safe_load(impl)
    if impl_data == None:
        log_sample_outcome(sample_log, False, error="Response Format Error: empty/unparsable YAML")
        raise Exception("Response Format Error!")
    # Static gate: reject torch-to-step bypasses before pytest can be
    # fooled by an identity load->store of a precomputed tensor.
    input_names = [inp["name"] for inp in task_data.get("inputs", [])]
    bypass = check_body_no_bypass(impl_data.get("impl", ""), input_names)
    if bypass:
        log_sample_outcome(sample_log, False, error=f"Bypass: {bypass}")
        return (False, impl_data, temp["usage"])
    data = {**task_data, **impl_data}
    code = yaml_to_code.yaml_to_code(data)
    with open(temp_test_path, 'w') as file:
        file.write(code)
    result = pytest.main([temp_test_path], plugins=[])
    usage_all_once = count_usage.check_affine_type(temp_test_path)
    passed = usage_all_once and result == pytest.ExitCode.OK
    log_sample_outcome(sample_log, passed, generated_code=code)
    os.remove(temp_test_path) # Comment this line to keep the test file when debugging
    return (passed, impl_data, temp["usage"])

class QueryAffineImpl(QueryImpl):
    def __init__(self, key, config: Storage):
        super().__init__(key, config)
    
    def run(self, storage: Storage):
        config_data = self.config.retrieve(self.key)
        num_samples = config_data["num_samples"]
        num_workers = config_data["num_workers"]
        storage_data = storage.retrieve(self.key)
        task_data = storage_data["task"]        
        prompt = storage_data["prompt"] # self.prepare_prompt(storage)
        success = []
        failure = []
        usages = []
        start = time.time()
        with tqdm(total=num_samples, smoothing=0) as pbar:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(
                        single_query_affine_impl,
                        id,
                        prompt,
                        task_data,
                        config_data
                    )
                    for id in range(num_samples)
                }
                for future in as_completed(futures):
                    pbar.update(1)
                    try:
                        result = future.result()
                        if result is not None:
                            if result[0]:
                                success.append(result[1])
                            else:
                                failure.append(result[1])
                            usages.append(result[2])
                    except Exception as e:
                        print("Got an error at QueryAffineImpl!", e)
                        continue
        end = time.time()
        print_elapsed_time(start, end)
        storage_data["success"] = success
        storage_data["failure"] = failure # Stores a list of failed impls (not task+impl)
        storage_data["usage"] = usages
    
