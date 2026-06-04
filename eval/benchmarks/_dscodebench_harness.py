"""DSCodeBench official test harness — vendored and made dependency-tolerant.

This is a faithful port of the scoring logic from the upstream repo:
  https://github.com/ShuyinOuyang/DSCodeBench/blob/main/benchmark_construction_evaluation/run_test.py

How DSCodeBench scores a solution
---------------------------------
Each benchmark problem ships three things:
  • ``ground_truth_code`` — a reference implementation whose LAST function is the
    "main" function the task asks for (helper functions come first).
  • ``test_script``       — defines ``test_case_input_generator(n)`` returning a
    list of input tuples for the main function.
  • ``code_problem``      — the natural-language task (includes the signature).

To score a candidate solution we:
  1. Append ``test_script`` to BOTH the ground-truth and the candidate code,
     seed every RNG (numpy/torch/tf/random) to 42, inject ``random_state`` into
     known sklearn estimators, and force the matplotlib ``Agg`` backend.
  2. Generate the SAME ``n`` test-case inputs (identical seed) and call the main
     function on each, collecting an ``output_list`` for both implementations.
  3. Compare the two output lists element-by-element with ``exec_test`` — a
     type-aware deep comparison covering 30+ types (ndarray/Tensor/DataFrame/
     sklearn estimator/keras model/scipy sparse/…). Plot problems (those whose
     ground truth writes ``output.png``) are compared by RGB-channel SSIM > 0.5.
  4. Each test case yields 1 (match) or 0 (mismatch). The per-problem result is
     the list of 1/0 — a problem is "solved" iff every entry is 1.

Difference from upstream
------------------------
The upstream ``exec_test`` does ``import torch / tensorflow / keras / lightgbm``
unconditionally, so a numpy-only problem would crash if those libs are absent.
Here the heavy imports are *guarded*: a missing library resolves to a sentinel
type that nothing is an instance of, so its ``isinstance`` branches simply never
match. Problems whose own library is missing can't run (ground truth fails to
import) and therefore score 0 — the correct, expected outcome for that env.

Run as a subprocess (so the 200s ``SIGALRM`` timeout and any segfault in exec'd
model code stay isolated from the evaluator process)::

    python -m eval.benchmarks._dscodebench_harness <task.json> <result.json>

``task.json``   : {ground_truth_code, solution_code, test_script,
                   test_case_number, random_seed}
``result.json`` : {"evaluation_result": [1, 0, 1, ...], "error": "<optional>"}
"""
from __future__ import annotations

import ast
import importlib
import json
import os
import re
import signal
import sys
import tempfile


# ── dependency-tolerant module access ───────────────────────────────────────────

class _NeverMeta(type):
    """Metaclass so attribute access on a Never-type yields another Never-type.

    This lets arbitrary attribute chains used in ``exec_test`` (e.g.
    ``scipy.sparse.bsr_matrix``, ``torch.utils.data.TensorDataset``) resolve to a
    *type* at every depth, which is what ``isinstance``'s second argument requires.
    """

    def __getattr__(cls, name):  # noqa: N805
        return cls


class _Never(metaclass=_NeverMeta):
    """A type that nothing is ever an instance of."""


class _SafeModule:
    """Stand-in for a missing module: every attribute is the unsatisfiable type."""

    def __getattr__(self, name):
        return _Never


def _imp(name: str):
    """Import ``name`` (and make it usable) or return a :class:`_SafeModule`."""
    try:
        return importlib.import_module(name)
    except Exception:
        return _SafeModule()


# ── code preparation (verbatim logic from upstream run_test.py) ─────────────────

def get_main_function_name_and_parameter_count_brief(code):
    try:
        tree = ast.parse(code)
        functions = [n for n in tree.body
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        function_name = functions[-1].name if functions else None
        function_parameter_count = len(functions[-1].args.args)
        return function_name, function_parameter_count
    except Exception:
        return None, None


def classify_code_ast(code):
    classified = {'imports': []}
    try:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                classified['imports'].append(ast.unparse(node))
    except Exception:
        pass
    return classified


def extract_imports(code):
    imports = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append((alias.name, alias.asname or alias.name))
            elif isinstance(node, ast.ImportFrom):
                module = node.module
                for alias in node.names:
                    imports.append((f"{module}.{alias.name}", alias.asname or alias.name))
    except Exception:
        pass
    return imports


def get_library_from_code(code):
    classified = classify_code_ast(code)
    import_libs = []
    for import_line in classified['imports']:
        import_libs += extract_imports(import_line.strip())
    return import_libs


def add_random_seed_code(import_libs, random_seed):
    random_seed_code = ''
    for import_lib in import_libs:
        if 'random' in import_lib[0]:
            random_seed_code += 'import random\nrandom.seed(%s)\n' % random_seed
        elif import_lib[0] in ['numpy', 'pandas', 'scipy', 'matplotlib',
                               'matplotlib.pyplot', 'seaborn']:
            random_seed_code += 'import numpy as np\nnp.random.seed(%s)\n' % random_seed
        elif 'torch' in import_lib[0]:
            random_seed_code += (
                'import random\nimport numpy as np\nimport torch\n'
                'torch.manual_seed(%s)\nrandom.seed(%s)\nnp.random.seed(%s)\n'
                % (random_seed, random_seed, random_seed))
        elif import_lib[0] in ['tensorflow', 'keras']:
            random_seed_code += (
                'import tensorflow as tf\nimport random\nimport numpy as np\n'
                'tf.random.set_seed(%s)\nrandom.seed(%s)\nnp.random.seed(%s)\n'
                % (random_seed, random_seed, random_seed))
    return random_seed_code


def add_random_seed_into_functions(code, random_seed):
    class AddRandomStateTransformer(ast.NodeTransformer):
        def __init__(self):
            self.api_list = [
                'RatioUniforms', 'RandomForestClassifier', 'RandomForestRegressor',
                'KFold', 'StratifiedKFold', 'LinearSVC', 'MLPRegressor',
                'train_test_split', 'make_regression', 'make_classification',
            ]

        def visit_Call(self, node):
            node.args = [self.visit(arg) for arg in node.args]
            node.keywords = [self.visit(keyword) for keyword in node.keywords]
            for randomness_api in self.api_list:
                if ((isinstance(node.func, ast.Name) and node.func.id == randomness_api) or
                        (isinstance(node.func, ast.Attribute) and node.func.attr == randomness_api)):
                    if randomness_api in ('KFold', 'StratifiedKFold'):
                        shuffle_exist = False
                        for keyword in node.keywords:
                            if keyword.arg == 'shuffle':
                                keyword.value = ast.Constant(value=True)
                                shuffle_exist = True
                            if keyword.arg == 'random_state':
                                keyword.value = ast.Constant(value=random_seed)
                                break
                        else:
                            if not shuffle_exist:
                                node.keywords.append(
                                    ast.keyword(arg='shuffle', value=ast.Constant(value=True)))
                            node.keywords.append(
                                ast.keyword(arg='random_state', value=ast.Constant(value=random_seed)))
                    else:
                        for keyword in node.keywords:
                            if keyword.arg == 'random_state':
                                keyword.value = ast.Constant(value=random_seed)
                                break
                        else:
                            node.keywords.append(
                                ast.keyword(arg='random_state', value=ast.Constant(value=random_seed)))
            return node

    tree = ast.parse(code)
    modified = AddRandomStateTransformer().visit(tree)
    return ast.unparse(modified)


def add_matplotlib_agg(code):
    lines = code.splitlines()
    result = []
    added_agg = False
    for line in lines:
        result.append(line)
        if 'matplotlib' in line and 'import' in line and not added_agg:
            result.append("import matplotlib\nmatplotlib.use('Agg')")
            added_agg = True
    return '\n'.join(result)


def get_additional_code(code, is_matplotlib_or_seaborn, test_case_number=0):
    tcn = '' if test_case_number == 0 else str(test_case_number)
    main_function_name, function_parameter_count = \
        get_main_function_name_and_parameter_count_brief(code)
    if is_matplotlib_or_seaborn:
        if function_parameter_count is None or function_parameter_count <= 2:
            call = f'    {main_function_name}(test_cases[i])'
        else:
            call = f'    {main_function_name}(*test_cases[i])'
        additional_code = f'''
from PIL import Image
import numpy as np

test_cases = test_case_input_generator({tcn})
output_list = []
for i in range(len(test_cases)):
{call}
    img = np.array(Image.open("output.png").convert("RGB"))
    output_list.append(img)
'''
    else:
        if function_parameter_count == 1:
            call = f'    output_list.append({main_function_name}(test_cases[i]))'
        else:
            call = f'    output_list.append({main_function_name}(*test_cases[i]))'
        additional_code = f'''
test_cases = test_case_input_generator({tcn})
output_list = []
for i in range(len(test_cases)):
{call}
'''
    return additional_code


def prepare_exec_code(code, test_case_script, is_ground_truth_code=True, random_seed=42,
                      is_matplotlib_or_seaborn=False, test_case_number=0):
    import_libs = get_library_from_code(code)
    try:
        code = add_random_seed_into_functions(code, random_seed)
    except Exception:
        pass
    code = add_matplotlib_agg(code)
    test_case_script = add_random_seed_into_functions(test_case_script, random_seed)
    if is_ground_truth_code:
        import_libs += get_library_from_code(test_case_script)
    import_libs = list(set(import_libs))
    random_seed_code = add_random_seed_code(import_libs, random_seed)
    additional_code = get_additional_code(code, is_matplotlib_or_seaborn, test_case_number)
    return code + '\n\n' + test_case_script + '\n\n' + random_seed_code + '\n\n' + additional_code


def _timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out!")


def get_code_output_list(exec_code, time_limit):
    if time_limit:
        signal.signal(signal.SIGALRM, _timeout_handler)
    local_namespace = {}
    test_case_output_list = []
    test_case_input_list = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        old_dir = os.getcwd()
        try:
            if time_limit:
                signal.alarm(200)
            os.chdir(tmp_dir)
            exec(exec_code, local_namespace)
            os.chdir(old_dir)
            if time_limit:
                signal.alarm(0)
            test_case_output_list = local_namespace['output_list']
            test_case_input_list = local_namespace['test_cases']
        except Exception as e:  # noqa: BLE001 — match upstream (swallow + empty result)
            print(e, file=sys.stderr)
            if time_limit:
                signal.alarm(0)
            test_case_output_list = []
            test_case_input_list = []
            os.chdir(old_dir)
    return test_case_input_list, test_case_output_list


def get_exec_output(ground_truth_code, test_solution_code, test_case_script,
                    time_limit=True, is_matplotlib_or_seaborn=False,
                    test_case_number=0, random_seed=42):
    exec_code_gt = prepare_exec_code(
        ground_truth_code, test_case_script, is_ground_truth_code=True,
        random_seed=random_seed, is_matplotlib_or_seaborn=is_matplotlib_or_seaborn,
        test_case_number=test_case_number)
    test_case_input_list, gt_output_list = get_code_output_list(exec_code_gt, time_limit=False)

    exec_code_sol = prepare_exec_code(
        test_solution_code, test_case_script, is_ground_truth_code=True,
        random_seed=random_seed, is_matplotlib_or_seaborn=is_matplotlib_or_seaborn,
        test_case_number=test_case_number)
    _, sol_output_list = get_code_output_list(exec_code_sol, time_limit=time_limit)

    return test_case_input_list, gt_output_list, sol_output_list


# ── type-aware comparison (logic faithful to upstream; imports guarded) ─────────

def exec_test(result, ans):
    import math

    np = _imp('numpy')
    pd = _imp('pandas')
    torch = _imp('torch')
    tf = _imp('tensorflow')
    keras = _imp('keras')
    lgb = _imp('lightgbm')
    sklearn = _imp('sklearn')
    scipy = _imp('scipy')
    # Ensure the submodules used below are bound on their packages when available.
    for _sub in ('scipy.sparse', 'scipy.stats', 'sklearn.base',
                 'torch.nn', 'torch.utils.data', 'keras.models', 'lightgbm.basic'):
        try:
            importlib.import_module(_sub)
        except Exception:
            pass

    if isinstance(result, tuple):
        assert len(result) == len(ans)
        for i in range(len(result)):
            assert type(result[i]) == type(ans[i])
            exec_test(result[i], ans[i])
    elif isinstance(result, sklearn.base.BaseEstimator):
        assert str(result.get_params()) == str(ans.get_params())
    elif isinstance(result, lgb.basic.Booster):
        assert result.dump_model() == ans.dump_model()
    elif isinstance(result, lgb.basic.Dataset):
        assert np.allclose(result.get_data(), ans.get_data(), equal_nan=True)
    elif isinstance(result, tf.Tensor):
        assert np.allclose(result.numpy(), ans.numpy(), equal_nan=True)
    elif isinstance(result, tf.Variable):
        assert np.allclose(result.numpy(), ans.numpy(), equal_nan=True)
    elif isinstance(result, scipy.sparse.bsr_matrix):
        assert result.shape == ans.shape
        assert result.nnz == ans.nnz
        assert np.allclose(result.indices, ans.indices)
        assert np.allclose(result.indptr, ans.indptr)
        assert np.allclose(result.data, ans.data)
    elif isinstance(result, scipy.sparse.coo_matrix):
        assert np.allclose(result.row, ans.row)
        assert np.allclose(result.col, ans.col)
        assert np.allclose(result.data, ans.data)
        assert result.shape == ans.shape
        assert result.nnz == ans.nnz
    elif isinstance(result, scipy.sparse.csc_matrix):
        assert result.shape == ans.shape
        assert result.nnz == ans.nnz
        assert np.allclose(result.indices, ans.indices)
        assert np.allclose(result.indptr, ans.indptr)
        assert np.allclose(result.data, ans.data)
    elif isinstance(result, scipy.sparse.csr_matrix):
        assert result.shape == ans.shape
        assert result.nnz == ans.nnz
        assert np.allclose(result.indices, ans.indices)
        assert np.allclose(result.indptr, ans.indptr)
        assert np.allclose(result.data, ans.data)
    elif isinstance(result, scipy.sparse.dia_matrix):
        assert result.shape == ans.shape
        assert result.nnz == ans.nnz
        assert np.allclose(result.data, ans.data)
        assert np.allclose(result.offsets, ans.offsets)
    elif isinstance(result, scipy.sparse.dok_matrix):
        assert result.shape == ans.shape
        assert result.nnz == ans.nnz
        result_dict, ans_dict = dict(result), dict(ans)
        for i in result_dict:
            exec_test(result_dict[i], ans_dict[i])
    elif isinstance(result, scipy.sparse.lil_matrix):
        assert result.shape == ans.shape
        assert result.nnz == ans.nnz
        assert all(np.allclose(result.rows[i], ans.rows[i]) for i in range(result.shape[0]))
        assert all(np.allclose(result.data[i], ans.data[i]) for i in range(result.shape[0]))
    elif isinstance(result, dict):
        for i in result:
            exec_test(result[i], ans[i])
    elif isinstance(result, list):
        assert len(result) == len(ans)
        for i in range(len(result)):
            exec_test(result[i], ans[i])
    elif isinstance(result, np.ma.MaskedArray):
        np.ma.allclose(result, ans)
    elif isinstance(result, np.ndarray):
        assert np.allclose(result, ans, equal_nan=True)
    elif isinstance(result, torch.Tensor):
        torch.allclose(result, ans, equal_nan=True)
    elif isinstance(result, torch.nn.Sequential):
        assert len(result) == len(ans)
        for p1, p2 in zip(result.parameters(), ans.parameters()):
            assert torch.allclose(p1, p2, equal_nan=True)
    elif isinstance(result, torch.nn.Linear):
        assert torch.allclose(result.weight, ans.weight)
        assert torch.allclose(result.bias, ans.bias)
    elif isinstance(result, scipy.stats._multivariate.multivariate_normal_frozen):
        np.allclose(result.mean, ans.mean)
        np.allclose(result.cov, ans.cov)
        assert result.dim == ans.dim
        assert result.random_state == ans.random_state
    elif isinstance(result, pd.DataFrame):
        pd.testing.assert_frame_equal(result, ans)
    elif isinstance(result, float):
        math.isclose(result, ans)
    elif isinstance(result, pd.Series):
        pd.testing.assert_series_equal(result, ans)
    elif isinstance(result, keras.models.Model):
        def normalize_layer_names(obj):
            if isinstance(obj, list):
                return [normalize_layer_names(item) for item in obj]
            elif isinstance(obj, str):
                return re.sub(r'_(\d+)$', '', obj)
            return obj

        def normalize_config(config):
            keys_to_remove = ['name', 'dtype', 'trainable']
            keys_to_modify = ['inbound_nodes', 'input_layers']

            def recursive_clean(obj):
                if isinstance(obj, dict):
                    for key in list(obj.keys()):
                        if key in keys_to_remove:
                            obj.pop(key)
                        elif key in keys_to_modify:
                            if obj[key]:
                                obj[key] = normalize_layer_names(obj[key])
                        else:
                            recursive_clean(obj[key])
                elif isinstance(obj, list):
                    for item in obj:
                        recursive_clean(item)
            recursive_clean(config)
            return config
        config1 = normalize_config([layer.get_config() for layer in result.layers])
        config2 = normalize_config([layer.get_config() for layer in ans.layers])
        assert config1 == config2
    elif isinstance(result, torch.utils.data.TensorDataset):
        assert len(result.tensors) == len(ans.tensors)
        for i, (t1, t2) in enumerate(zip(result.tensors, ans.tensors)):
            assert torch.allclose(t1, t2, equal_nan=True)
    else:
        assert result == ans
    return 1


def exec_test_img(result, ans):
    from skimage.metrics import structural_similarity as ssim
    ssim_r, _ = ssim(result[:, :, 0], ans[:, :, 0], full=True)
    ssim_g, _ = ssim(result[:, :, 1], ans[:, :, 1], full=True)
    ssim_b, _ = ssim(result[:, :, 2], ans[:, :, 2], full=True)
    ssim_avg = (ssim_r + ssim_g + ssim_b) / 3
    assert ssim_avg > 0.5
    return 1


def evaluate_outputs(test_case_input_list, gt_output_list, sol_output_list,
                     is_matplotlib_or_seaborn=False):
    # NB: faithful to upstream — plots are loaded as RGB ndarrays in the exec'd
    # code and compared here via exec_test's np.ndarray branch (pixel-wise
    # np.allclose), NOT via exec_test_img/SSIM. ``is_matplotlib_or_seaborn`` is
    # accepted for parity with upstream but is unused in the scoring path.
    evaluation_result_list = []
    for i in range(len(test_case_input_list)):
        try:
            gt_output = gt_output_list[i]
            sol_output = sol_output_list[i]
            if type(gt_output) == type(sol_output):
                evaluation_result_list.append(exec_test(gt_output, sol_output))
            else:
                evaluation_result_list.append(0)
        except Exception:
            evaluation_result_list.append(0)
    return evaluation_result_list


# ── subprocess entry point ──────────────────────────────────────────────────────

def evaluate(ground_truth_code, solution_code, test_script,
             test_case_number=50, random_seed=42):
    """Return the per-test-case 1/0 list for a single problem."""
    is_plot = 'output.png' in ground_truth_code
    inputs, gt_out, sol_out = get_exec_output(
        ground_truth_code, solution_code, test_script,
        is_matplotlib_or_seaborn=is_plot,
        test_case_number=test_case_number, random_seed=random_seed)
    return evaluate_outputs(inputs, gt_out, sol_out, is_matplotlib_or_seaborn=is_plot)


def main() -> None:
    task_path, result_path = sys.argv[1], sys.argv[2]
    with open(task_path) as f:
        task = json.load(f)
    out = {"evaluation_result": []}
    try:
        out["evaluation_result"] = evaluate(
            ground_truth_code=task["ground_truth_code"],
            solution_code=task["solution_code"],
            test_script=task["test_script"],
            test_case_number=int(task.get("test_case_number", 50)),
            random_seed=int(task.get("random_seed", 42)),
        )
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    with open(result_path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
