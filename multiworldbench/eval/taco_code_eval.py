import subprocess
import tempfile
import os
import re
import json
from tqdm import tqdm  # 导入进度条库

# ------------------ 新增：代码清洗函数 ------------------
def clean_code(code: str) -> str:
    """
    移除代码中可能包含的```python开头和```结尾标识符
    """
    # 移除开头的```python标记（忽略大小写，允许前后有空白）
    code = re.sub(r'^\s*```python\s*', '', code, flags=re.IGNORECASE)
    # 移除结尾的```标记（允许前后有空白）
    code = re.sub(r'\s*```\s*$', '', code)
    return code


# ------------------ 核心执行函数 ------------------
def run_python_code(code: str, input_str: str, timeout=2):
    """
    执行 Python 代码，返回是否成功、标准输出和标准错误
    """
    # 新增：执行前先清洗代码
    code = clean_code(code)
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        file_path = f.name

    try:
        result = subprocess.run(
            ["python", file_path],
            input=input_str,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True
        )
        success = (result.returncode == 0)
        return success, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", "TLE"
    finally:
        os.remove(file_path)

# ------------------ 输入归一化 ------------------
def normalize_input(inp):
    """
    保证传给 subprocess 的输入一定是字符串
    """
    if isinstance(inp, list):
        return "\n".join(map(str, inp)) + "\n"
    return str(inp)

def normalize_io(inputs, outputs):
    """
    对整个输入输出列表做归一化
    """
    norm_inputs = [normalize_input(inp) for inp in inputs]
    norm_outputs = [str(out).strip() for out in outputs]  # 输出去掉多余空格和换行
    return norm_inputs, norm_outputs

# ------------------ 代码正确性检查 ------------------
def check_correctness(code, inputs, outputs):
    """
    返回：
        executable (bool): 代码能否成功运行
        correct (bool): 代码是否完全正确
    """
    for inp, gt in zip(inputs, outputs):
        inp_norm = normalize_input(inp)
        success, out, err = run_python_code(code, inp_norm)
        if not success:
            return False, False
        if out.strip() != gt.strip():
            return True, False
    return True, True

# ------------------ 单题评测 ------------------
def evaluate_problem(generated_codes, problem, k):
    # 解析 input_output
    input_output = problem["input_output"]
    if isinstance(input_output, str):
        input_output = json.loads(input_output)

    inputs = input_output["inputs"]
    outputs = input_output["outputs"]

    # 归一化
    inputs, outputs = normalize_io(inputs, outputs)

    executable = 0
    correct = 0

    # 单题内代码校验添加子进度条（leave=False 不保留进度条，避免刷屏）
    for code in tqdm(
        generated_codes[:k], 
        desc=f"Checking solutions (q: {problem['question'][:20]}...)", 
        unit="code", 
        leave=False,
        ncols=80
    ):
        is_exec, is_correct = check_correctness(code, inputs, outputs)
        if is_exec:
            executable += 1
        if is_correct:
            correct += 1

    return {
        "pass@k": 1 if correct > 0 else 0,
        "execution_rate": executable / k
    }

# ------------------ 全数据集评测 ------------------
def evaluate_dataset(dataset, generated_map, k):
    total_pass = 0
    total_exec = 0
    num_problems = 0

    # 数据集遍历添加总进度条
    for problem in tqdm(
        dataset, 
        desc=f"Evaluating dataset (k={k})", 
        unit="problem", 
        ncols=100
    ):
        q = problem["question"]
        if q not in generated_map:
            tqdm.write(f"[WARN] No generated solutions for question: {q[:50]}...")  # 用tqdm.write避免覆盖进度条
            continue

        r = evaluate_problem(
            generated_map[q]["generated_solutions"],
            problem,
            k
        )

        total_pass += r["pass@k"]
        total_exec += r["execution_rate"]
        num_problems += 1

    return {
        f"pass@{k}": total_pass / num_problems if num_problems > 0 else 0.0,
        "execution_rate": total_exec / num_problems if num_problems > 0 else 0.0
    }

# ------------------ 主入口 ------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate generated code on TACO.")
    parser.add_argument("--dataset_path", required=True, help="TACO evaluation JSON file.")
    parser.add_argument("--generated_path", required=True, help="JSON file containing generated solutions.")
    args = parser.parse_args()

    with open(args.dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    with open(args.generated_path, "r", encoding="utf-8") as f:
        generated = json.load(f)

    generated_map = {item["question"]: item for item in generated}

    result_pass1 = evaluate_dataset(dataset, generated_map, k=1)
    result_pass5 = evaluate_dataset(dataset, generated_map, k=5)

    print("\n==== Evaluation Results ====")
    print(f"Pass@1: {result_pass1['pass@1']:.4f}")
    print(f"Pass@5: {result_pass5['pass@5']:.4f}")
    print(f"Execution Rate: {result_pass1['execution_rate']:.4f}")
