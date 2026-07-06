import subprocess
import tempfile
import os
import re
import json
from tqdm import tqdm
import ast

# ------------------ 代码清洗函数 ------------------
def clean_code(code: str) -> str:
    """
    移除代码中可能包含的```python开头和```结尾标识符
    """
    code = re.sub(r'^\s*```python\s*', '', code, flags=re.IGNORECASE)
    code = re.sub(r'\s*```\s*$', '', code)
    return code

# ------------------ 解析assert语句 ------------------
def parse_assert_to_io(assert_statement: str):
    """
    从assert语句中提取输入参数和期望输出
    
    例如: assert check("01010101010") == "Yes"
    返回: (["01010101010"], "Yes")
    
    支持多种格式:
    - assert func(arg) == expected
    - assert func(arg1, arg2) == expected
    - assert func(arg) == expected
    """
    try:
        # 移除 "assert " 前缀
        statement = assert_statement.strip()
        if statement.startswith("assert "):
            statement = statement[7:].strip()
        
        # 分割 == 两边
        parts = statement.split("==")
        if len(parts) != 2:
            return None, None
        
        func_call = parts[0].strip()
        expected = parts[1].strip()
        
        # 解析期望输出（去除引号）
        try:
            expected_value = ast.literal_eval(expected)
        except:
            expected_value = expected
        
        # 提取函数参数
        # 匹配函数调用: func_name(args)
        match = re.match(r'(\w+)\((.*)\)', func_call)
        if not match:
            return None, None
        
        func_name = match.group(1)
        args_str = match.group(2)
        
        # 解析参数
        try:
            # 使用ast.literal_eval安全地解析参数
            if args_str.strip():
                # 将参数字符串包装成tuple来解析
                args = ast.literal_eval(f"({args_str},)")
                if not isinstance(args, tuple):
                    args = (args,)
            else:
                args = ()
        except:
            return None, None
        
        return list(args), str(expected_value)
        
    except Exception as e:
        return None, None

# ------------------ 核心执行函数（IO模式）------------------
def run_python_code_io(code: str, test_inputs: list, expected_outputs: list, timeout=2):
    """
    以IO模式执行Python代码
    
    参数:
        code: 生成的完整代码（包含input/output逻辑）
        test_inputs: 测试输入列表，每个元素是一个参数列表
        expected_outputs: 期望输出列表
        timeout: 超时时间
    
    返回:
        executable (bool): 代码是否可执行
        all_passed (bool): 所有测试是否通过
        pass_count (int): 通过的测试数量
    """
    code = clean_code(code)
    
    # 写入临时文件
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding='utf-8') as f:
        f.write(code)
        file_path = f.name
    
    passed = 0
    executable = True
    
    try:
        # 构建输入：第一行是测试用例数量，后面是每个测试用例的输入
        stdin_content = f"{len(test_inputs)}\n"
        for inputs in test_inputs:
            # 将每个参数转换为一行输入
            for inp in inputs:
                stdin_content += f"{inp}\n"
        
        # 执行代码
        result = subprocess.run(
            ["python", file_path],
            input=stdin_content,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True
        )
        
        # 检查是否成功执行
        if result.returncode != 0:
            return False, False, 0
        
        # 解析输出
        output_lines = result.stdout.strip().split('\n')
        
        # 比较每个输出
        for i, (expected, actual) in enumerate(zip(expected_outputs, output_lines)):
            # 标准化输出（去除空格，统一大小写比较）
            expected_normalized = str(expected).strip().lower()
            actual_normalized = actual.strip().lower()
            
            if expected_normalized == actual_normalized:
                passed += 1
        
        all_passed = (passed == len(expected_outputs))
        return True, all_passed, passed
        
    except subprocess.TimeoutExpired:
        return False, False, 0
    except Exception as e:
        return False, False, 0
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

# ------------------ 单题评测 ------------------
def evaluate_problem(generated_codes, problem, k):
    """
    评测单个问题的生成代码（IO模式）
    """
    test_cases = problem.get("test_list", [])
    
    if not test_cases:
        return {"pass@k": 0, "execution_rate": 0, "avg_pass_rate": 0}
    
    # 解析所有测试用例
    test_inputs = []
    expected_outputs = []
    
    for test in test_cases:
        inputs, output = parse_assert_to_io(test)
        if inputs is not None and output is not None:
            test_inputs.append(inputs)
            expected_outputs.append(output)
    
    if not test_inputs:
        return {"pass@k": 0, "execution_rate": 0, "avg_pass_rate": 0}
    
    executable = 0
    correct = 0
    total_pass_rate = 0.0
    
    for code in tqdm(
        generated_codes[:k],
        desc=f"Testing task {problem.get('task_id', 'unknown')}",
        unit="code",
        leave=False,
        ncols=80
    ):
        is_exec, is_correct, pass_count = run_python_code_io(
            code, test_inputs, expected_outputs
        )
        
        if is_exec:
            executable += 1
            pass_rate = pass_count / len(expected_outputs)
            total_pass_rate += pass_rate
            
            if is_correct:
                correct += 1
    
    return {
        "pass@k": 1 if correct > 0 else 0,
        "execution_rate": executable / k if k > 0 else 0,
        "avg_pass_rate": total_pass_rate / executable if executable > 0 else 0
    }

# ------------------ 全数据集评测 ------------------
def evaluate_dataset(dataset, generated_map, k):
    """
    评测整个数据集
    """
    total_pass = 0
    total_exec = 0
    total_pass_rate = 0
    num_problems = 0
    
    for problem in tqdm(
        dataset,
        desc=f"Evaluating dataset (k={k})",
        unit="problem",
        ncols=100
    ):
        question_text = problem["text"]
        
        if question_text not in generated_map:
            tqdm.write(f"[WARN] No generated solutions for: {question_text[:50]}...")
            continue
        
        result = evaluate_problem(
            generated_map[question_text]["generated_solutions"],
            problem,
            k
        )
        
        total_pass += result["pass@k"]
        total_exec += result["execution_rate"]
        total_pass_rate += result["avg_pass_rate"]
        num_problems += 1
    
    return {
        f"pass@{k}": total_pass / num_problems if num_problems > 0 else 0.0,
        "execution_rate": total_exec / num_problems if num_problems > 0 else 0.0,
        "avg_pass_rate": total_pass_rate / num_problems if num_problems > 0 else 0.0
    }

# ------------------ 主入口 ------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate generated code on MBPP.")
    parser.add_argument("--dataset_path", required=True, help="MBPP evaluation JSONL file.")
    parser.add_argument("--generated_path", required=True, help="JSON file containing generated solutions.")
    args = parser.parse_args()

    dataset = []
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))

    print(f"Loaded {len(dataset)} problems from dataset")

    with open(args.generated_path, "r", encoding="utf-8") as f:
        generated = json.load(f)

    generated_map = {item["question"]: item for item in generated}
    print(f"Loaded {len(generated_map)} generated solutions")

    result_pass1 = evaluate_dataset(dataset, generated_map, k=1)
    result_pass5 = evaluate_dataset(dataset, generated_map, k=5)

    print("\n==== Evaluation Results ====")
    print(f"Pass@1:           {result_pass1['pass@1']:.4f} ({result_pass1['pass@1']*100:.2f}%)")
    print(f"Pass@5:           {result_pass5['pass@5']:.4f} ({result_pass5['pass@5']*100:.2f}%)")
    print(f"Execution Rate:   {result_pass1['execution_rate']:.4f} ({result_pass1['execution_rate']*100:.2f}%)")
    print(f"Avg Pass Rate:    {result_pass1['avg_pass_rate']:.4f} ({result_pass1['avg_pass_rate']*100:.2f}%)")
