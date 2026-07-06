import json
import argparse
import numpy as np
import os
import hashlib
from typing import List, Dict, Any, Optional, Union
from bert_score import score as bert_score

# 新增导入
import requests
import base64
from pathlib import Path

# ==================== LLM解析相关函数 ===================

def _require_token(value: str, env_name: str = "MWB_EVAL_API_KEY") -> str:
    if not value:
        raise RuntimeError(f"Missing evaluator API token. Please set {env_name} or pass --token.")
    return value

def _require_endpoint(value: str, env_name: str = "MWB_EVAL_API_URL") -> str:
    if not value:
        raise RuntimeError(f"Missing evaluator API endpoint. Please set {env_name} or pass --api_url.")
    return value

def encode_base64(image_path: str) -> str:
    """
    将图片文件编码为base64字符串
    """
    with open(image_path, "rb") as _f:
        data = _f.read()
    return base64.b64encode(data).decode()


def llm(prompt: str,
        model: str = "gpt-5.1-2025-11-13",
        image_path: Optional[Union[str, List[str]]] = None,
        max_tokens: int = 1688,
        temperature: float = 0.5,
        api_url: str = None,
        token: str = None
        ) -> str:
    """
    调用LLM API获取响应
    """
    if token is None:
        raise ValueError("API token is required. Please provide it via --token argument or MWB_EVAL_API_KEY environment variable.")

    content = [{"type": "text", "text": prompt}]

    if image_path is not None:
        if isinstance(image_path, str):
            image_path = [image_path]

        for img_path in image_path:
            image_base64 = encode_base64(img_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                }
            })

    payload = json.dumps({
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    })

    headers = {
        "Authorization": f"Bearer {_require_token(token)}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(_require_endpoint(api_url), headers=headers, data=payload, timeout=60)
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"]
        return result
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"LLM API call failed: {str(e)}")


def get_default_parse_prompt() -> str:
    """
    获取默认的解析prompt
    """
    return """You are a data format parser. Your task is to convert irregular function call outputs into a standardized JSON format.

## Input Format
You will receive a possibly irregular text output that represents one or more function calls. The output may be:
- A malformed JSON string
- A natural language description of function calls
- A mix of JSON and text
- Any other irregular format

## Output Format
You MUST output a JSON array containing function call objects in the EXACT format below:
[
  {
    "name": "function_name",
    "arguments": {
      "param1": value1,
      "param2": value2
    }
  }
]

## Requirements
1. Output ONLY valid JSON, no additional text, explanations, or markdown
2. If no function calls are found, output an empty array: []
3. Infer parameter values from the input text when possible
4. Use appropriate data types (string, number, boolean, null) for parameter values
5. If a parameter value is unclear, use null or an empty string

## Examples

### Example 1
Input: "I need to calculate the area of a circle with radius 10 meters"
Output: [{"name": "geometry.area_circle", "arguments": {"radius": 10, "units": "meters"}}]

### Example 2
Input: "calculate_circle_area(5, 'cm')"
Output: [{"name": "geometry.area_circle", "arguments": {"radius": 5, "units": "cm"}}]

### Example 3
Input: "No function calls needed"
Output: []

Now, parse the following input and output ONLY the JSON array:"""


def parse_with_llm(raw_output: str,
                   model: str,
                   token: str,
                   prompt_template: str,
                   max_tokens: int = 1688,
                   temperature: float = 0.1  # 降低温度以获得更稳定的输出
                   ) -> List[Dict[str, Any]]:
    """
    使用LLM解析不规则的输出为规整的JSON格式
    """
    prompt = f"{prompt_template}\n\n{raw_output}"
    
    try:
        response = llm(
            prompt=prompt,
            model=model,
            token=token,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # 提取JSON部分（移除可能的markdown代码块标记）
        response = response.strip()
        if response.startswith("```json"):
            response = response[len("```json"):].strip()
        if response.startswith("```"):
            response = response[len("```"):].strip()
        if response.endswith("```"):
            response = response[:-len("```")].strip()
        
        # 解析JSON
        parsed_data = json.loads(response)
        
        # 验证格式
        if not isinstance(parsed_data, list):
            raise ValueError("Parsed output is not a JSON array")
        
        for item in parsed_data:
            if not isinstance(item, dict):
                raise ValueError("Array items must be objects")
            if "name" not in item or "arguments" not in item:
                raise ValueError("Each object must contain 'name' and 'arguments' fields")
            if not isinstance(item["arguments"], dict):
                raise ValueError("'arguments' must be an object")
        
        return parsed_data
        
    except Exception as e:
        print(f"Warning: LLM parsing failed: {str(e)}")
        # 返回一个标记解析失败的特殊格式
        return [{"name": "parsing_failed", "arguments": {"error": str(e), "raw_output": raw_output}}]


def load_or_parse_predictions(pred_data: List[Dict],
                              cache_file: Optional[str],
                              prompt_file: Optional[str],
                              model: str,
                              token: str,
                              max_tokens: int,
                              temperature: float,
                              force_reparse: bool = False
                              ) -> List[Dict]:
    """
    加载预测数据，使用缓存或LLM解析
    """
    # 加载prompt模板
    if prompt_file and os.path.exists(prompt_file):
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read().strip()
    else:
        prompt_template = get_default_parse_prompt()
        if prompt_file:
            print(f"Warning: Prompt file '{prompt_file}' not found, using default prompt")
    
    # 加载缓存
    cache = {}
    if cache_file and os.path.exists(cache_file) and not force_reparse:
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        cache[item["id"]] = item
            print(f"Loaded {len(cache)} cached parsing results")
        except Exception as e:
            print(f"Warning: Failed to load cache: {str(e)}")
    
    # 处理每个样本
    parsed_data = []
    need_cache_update = False
    
    for i, sample in enumerate(pred_data):
        sid = sample["id"]
        
        # 跳过失败的样本
        status = sample.get("status", "success")
        if status == "error":
            parsed_data.append(sample)
            continue
        
        # 检查是否有 llm_answers 字段
        if "llm_answers" not in sample:
            parsed_data.append(sample)
            continue
        
        raw_output = sample["llm_answers"]
        
        # 如果已经是标准格式且不需要重新解析，直接保留
        if not force_reparse and isinstance(raw_output, list) and len(raw_output) > 0:
            if all(isinstance(item, dict) and "name" in item and "arguments" in item for item in raw_output):
                parsed_data.append(sample)
                continue
        
        # 检查缓存
        if sid in cache and not force_reparse:
            cached_item = cache[sid]
            # 验证缓存项的有效性
            if (cached_item.get("status") == "success" and 
                "parsed_answers" in cached_item and
                isinstance(cached_item["parsed_answers"], list)):
                sample["llm_answers"] = cached_item["parsed_answers"]
                parsed_data.append(sample)
                continue
        
        # 使用LLM解析
        print(f"[{i+1}/{len(pred_data)}] Parsing sample {sid} with LLM...")
        try:
            if isinstance(raw_output, (list, dict)):
                # 如果已经是某种JSON格式，先序列化为字符串
                raw_output_str = json.dumps(raw_output, ensure_ascii=False)
            else:
                raw_output_str = str(raw_output)
            
            parsed_answers = parse_with_llm(
                raw_output=raw_output_str,
                model=model,
                token=token,
                prompt_template=prompt_template,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # 更新样本
            sample["llm_answers"] = parsed_answers
            sample["original_answers"] = raw_output  # 保留原始输出
            sample["parsed_at"] = "2026-01-10"  # 添加时间戳
            
            # 添加到缓存
            if cache_file:
                cache[sid] = {
                    "id": sid,
                    "status": "success",
                    "parsed_answers": parsed_answers,
                    "original_answers": raw_output,
                    "parsed_at": "2026-01-10"
                }
                need_cache_update = True
                
        except Exception as e:
            print(f"Error parsing sample {sid}: {str(e)}")
            # 保留原始数据，但标记为解析失败
            sample["parsing_error"] = str(e)
        
        parsed_data.append(sample)
    
    # 保存缓存
    if cache_file and need_cache_update:
        try:
            # 将缓存写入临时文件，然后原子替换
            temp_file = cache_file + ".tmp"
            with open(temp_file, "w", encoding="utf-8") as f:
                for item in cache.values():
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            os.replace(temp_file, cache_file)
            print(f"Cache saved to {cache_file}")
        except Exception as e:
            print(f"Warning: Failed to save cache: {str(e)}")
    
    return parsed_data


# ==================== 原有评估函数（保持不变） ====================


def load_jsonl(file_path: str) -> List[Dict]:
    """加载 JSONL 格式文件（每行一个 JSON）"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def normalize_value(value):
    """归一化值，处理空字符串等边界情况"""
    if value == "":
        return None
    return value


def check_argument_match(pred_args: Dict, gold_options: Dict) -> bool:
    """
    检查预测的参数是否匹配标准答案
    gold_options 格式: {"radius": [10], "units": ["meters", ""]}
    pred_args 格式: {"radius": 10, "units": "meters"}
    """
    for param, acceptable_values in gold_options.items():
        pred_value = pred_args.get(param)
        
        # 归一化预测值
        pred_value = normalize_value(pred_value)
        
        # 归一化可接受值列表
        normalized_acceptable = [normalize_value(v) for v in acceptable_values]
        
        # 检查预测值是否在可接受值列表中
        if pred_value not in normalized_acceptable:
            return False
    
    # 检查是否有多余的参数（可选：根据需求决定是否严格检查）
    # 这里采用宽松策略，只要必需参数正确即可
    return True


def exact_match_new_format(pred: List[Dict], gold: List[Dict]) -> int:
    """
    新格式的精确匹配
    pred: [{"name": "geometry.area_circle", "arguments": {"radius": 10, "units": "meters"}}]
    gold: [{"geometry.area_circle": {"radius": [10], "units": ["meters", ""]}}]
    """
    # 如果数量不匹配，直接返回0
    if len(pred) != len(gold):
        return 0
    
    # 将 gold 转换为更容易匹配的格式
    gold_calls = {}
    for g in gold:
        for func_name, params in g.items():
            gold_calls[func_name] = params
    
    # 检查每个预测的函数调用
    matched = 0
    for p in pred:
        func_name = p.get("name")
        pred_args = p.get("arguments", {})
        
        if func_name in gold_calls:
            if check_argument_match(pred_args, gold_calls[func_name]):
                matched += 1
    
    # 只有全部匹配才返回1
    return int(matched == len(pred) == len(gold))


def serialize_for_bertscore(tool_calls: List[Dict], is_gold: bool = False) -> str:
    """
    将多个 tool call 序列化成自然语言友好的字符串
    """
    lines = []
    
    if is_gold:
        # gold 格式: [{"geometry.area_circle": {"radius": [10], "units": ["meters", ""]}}]
        for tc in tool_calls:
            for func_name, params in tc.items():
                # 对于 gold，取每个参数的第一个值作为代表
                simplified_params = {k: v[0] if v else None for k, v in params.items()}
                lines.append(
                    f"Tool: {func_name}, Arguments: {json.dumps(simplified_params, sort_keys=True)}"
                )
    else:
        # pred 格式: [{"name": "geometry.area_circle", "arguments": {"radius": 10}}]
        for tc in tool_calls:
            lines.append(
                f"Tool: {tc.get('name')}, Arguments: {json.dumps(tc.get('arguments', {}), sort_keys=True)}"
            )
    
    return "\n".join(lines) if lines else "No tool calls"


def main(args):
    # 加载预测结果
    print("Loading prediction data...")
    pred_data = load_jsonl(args.result_json)
    
    # 如果使用LLM解析，先处理数据
    if args.parse_with_llm:
        print("Starting LLM parsing process...")
        pred_data = load_or_parse_predictions(
            pred_data=pred_data,
            cache_file=args.cache_file,
            prompt_file=args.parse_prompt_file,
            model=args.model,
            token=args.token,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            force_reparse=args.force_reparse
        )
        
        # 保存解析后的结果（可选）
        if args.save_parsed_results:
            parsed_output_file = args.result_json.replace(".json", "_parsed.json")
            with open(parsed_output_file, "w", encoding="utf-8") as f:
                for item in pred_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"Parsed results saved to: {parsed_output_file}")
    
    # 加载标准答案
    print("Loading ground truth data...")
    gold_data = load_jsonl(args.answer_json)
    
    # 创建 id 到答案的映射
    gold_map = {item["id"]: item["ground_truth"] for item in gold_data}
    
    em_scores = []
    pred_texts = []
    gold_texts = []
    
    # 统计信息
    total_samples = len(pred_data)
    evaluated_samples = 0
    skipped_samples = 0
    skipped_ids = []
    parsing_failed_ids = []

    for sample in pred_data:
        sid = sample["id"]
        
        # 检查样本状态，跳过失败的样本
        status = sample.get("status", "success")
        if status == "error":
            skipped_samples += 1
            skipped_ids.append(sid)
            print(f"[Skipped] ID: {sid} (status: error)")
            continue
        
        # 检查是否有 llm_answers 字段
        if "llm_answers" not in sample:
            skipped_samples += 1
            skipped_ids.append(sid)
            print(f"[Skipped] ID: {sid} (missing llm_answers)")
            continue
        
        pred = sample["llm_answers"]
        
        # 检查解析是否失败
        if (isinstance(pred, list) and len(pred) > 0 and 
            pred[0].get("name") == "parsing_failed"):
            parsing_failed_ids.append(sid)
            print(f"[Warning] ID: {sid} parsing failed: {pred[0]['arguments']['error']}")
        
        # 获取对应的标准答案
        if sid not in gold_map:
            skipped_samples += 1
            skipped_ids.append(sid)
            print(f"[Skipped] ID: {sid} (no ground truth)")
            continue
        
        gold = gold_map[sid]
        
        try:
            em_score = exact_match_new_format(pred, gold)
            em_scores.append(em_score)
            pred_texts.append(serialize_for_bertscore(pred, is_gold=False))
            gold_texts.append(serialize_for_bertscore(gold, is_gold=True))
            evaluated_samples += 1
            
        except Exception as e:
            skipped_samples += 1
            skipped_ids.append(sid)
            print(f"[Skipped] ID: {sid} (evaluation error: {str(e)})")
            continue

    # 如果没有有效样本可评估，提前退出
    if evaluated_samples == 0:
        print("\n" + "="*60)
        print("ERROR: No valid samples to evaluate!")
        print(f"Total samples in result file: {total_samples}")
        print(f"All samples were skipped.")
        if parsing_failed_ids:
            print(f"Parsing failed IDs: {', '.join(parsing_failed_ids)}")
        print("="*60)
        return

    # 计算 BERTScore
    try:
        P, R, F1 = bert_score(
            pred_texts,
            gold_texts,
            lang="en",
            model_type=args.bert_model
        )
        
        bertscore_success = True
    except Exception as e:
        print(f"\nWarning: BERTScore calculation failed: {str(e)}")
        bertscore_success = False

    # 输出评估结果
    print("\n" + "="*60)
    print("========== Evaluation Results ==========")
    print(f"Total samples in file: {total_samples}")
    print(f"Evaluated samples: {evaluated_samples}")
    print(f"Skipped samples: {skipped_samples}")
    if skipped_ids:
        print(f"Skipped IDs: {', '.join(skipped_ids)}")
    if parsing_failed_ids:
        print(f"Parsing failed IDs: {len(parsing_failed_ids)}")
    print("-" * 60)
    print(f"Exact Match: {np.mean(em_scores):.4f}")
    
    if bertscore_success:
        print(f"BERTScore Precision: {P.mean():.4f}")
        print(f"BERTScore Recall:    {R.mean():.4f}")
        print(f"BERTScore F1:        {F1.mean():.4f}")
    else:
        print("BERTScore: N/A (calculation failed)")
    
    print("="*60 + "\n")
    
    # 如果需要，保存详细的评估结果
    if args.save_detailed:
        detailed_results = []
        for i in range(evaluated_samples):
            detailed_results.append({
                "exact_match": int(em_scores[i]),
                "bertscore_precision": float(P[i]) if bertscore_success else None,
                "bertscore_recall": float(R[i]) if bertscore_success else None,
                "bertscore_f1": float(F1[i]) if bertscore_success else None,
            })
        
        output_file = args.result_json.replace(".json", "_detailed_eval.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "summary": {
                    "total_samples": total_samples,
                    "evaluated_samples": evaluated_samples,
                    "skipped_samples": skipped_samples,
                    "parsing_failed_count": len(parsing_failed_ids),
                    "exact_match": float(np.mean(em_scores)),
                    "bertscore_precision": float(P.mean()) if bertscore_success else None,
                    "bertscore_recall": float(R.mean()) if bertscore_success else None,
                    "bertscore_f1": float(F1.mean()) if bertscore_success else None,
                },
                "parsing_failed_ids": parsing_failed_ids,
                "detailed_results": detailed_results
            }, f, indent=2, ensure_ascii=False)
        print(f"Detailed results saved to: {output_file}")


if __name__ == "__main__":
    MODEL = "gpt-5.1"
    parser = argparse.ArgumentParser(description="Tool Use Evaluation with LLM-based Output Parsing")
    
    # 原有参数
    parser.add_argument("--result_json", type=str, required=True,
                        help="LLM生成的结果文件（JSONL格式）")
    parser.add_argument("--answer_json", type=str, required=True,
                        help="标准答案文件（JSONL格式）")
    parser.add_argument("--bert_model", type=str, default="microsoft/deberta-xlarge-mnli")
    parser.add_argument("--save_detailed", default=False, action="store_true",
                        help="是否保存详细的评估结果到JSON文件")
    
    # 新增LLM解析相关参数
    parser.add_argument("--parse_with_llm", default=True, action="store_true",
                        help="是否使用LLM解析不规则的输出格式")
    parser.add_argument("--cache_file", type=str, default=None,
                        help="LLM解析结果的缓存文件路径（JSONL格式）")
    parser.add_argument("--parse_prompt_file", type=str,
                        help="LLM解析prompt文件路径（可选，使用默认prompt若未指定）")
    parser.add_argument("--save_parsed_results", default=False, action="store_true",
                        help="是否保存解析后的结果到文件")
    parser.add_argument("--force_reparse", default=False, action="store_true",
                        help="强制重新解析所有样本，忽略缓存")
    
    # LLM API参数
    parser.add_argument("--token", type=str, 
                        default=os.environ.get("MWB_EVAL_API_KEY", ""),
                        help="LLM API的token（也可以通过环境变量MWB_EVAL_API_KEY设置）")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="用于解析的LLM模型名称")
    parser.add_argument("--max_tokens", type=int, default=1688,
                        help="LLM生成的最大token数")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="LLM的temperature参数（推荐0.1-0.3以获得稳定输出）")
    parser.add_argument("--api_url", type=str, default=os.environ.get("MWB_EVAL_API_URL", ""),
                        help="LLM API的URL")

    args = parser.parse_args()
    
    # 验证必要参数
    if args.parse_with_llm and not args.token:
        parser.error("--parse_with_llm requires --token or MWB_EVAL_API_KEY environment variable")
    
    main(args)
