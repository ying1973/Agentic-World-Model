import os
import json
from typing import Dict, Any

from awm.tools import get_tools, get_tools_new
from awm.react_gpt import MultiToolReActAgent


def load_taco_dataset(json_path: str) -> list:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("Expected JSON file to contain a list of samples")
    
    return data


def build_prompt(sample: Dict[str, Any]) -> str:
    question = sample.get("question", "")
    input_output = sample.get("input_output", "{}")
    starter_code = sample.get("starter_code", "")
    difficulty = sample.get("difficulty", "UNKNOWN")
    tags = sample.get("tags", "[]")
    time_limit = sample.get("time_limit", "N/A")
    memory_limit = sample.get("memory_limit", "N/A")
    
    # Build prompt
    prompt = f"""You are given a programming problem. Generate Python code to solve it.

Problem Statement:
{question}

Difficulty: {difficulty}
Tags: {tags}
Time Limit: {time_limit}
Memory Limit: {memory_limit}
"""
    
    if starter_code:
        prompt += f"\nStarter Code:\n{starter_code}\n"
    
    prompt += """
Instructions:
- Solve the problem directly and deterministically.
- Do NOT perform iterative exploration or repeated tool usage.
- Reason internally, then write the final solution once.
- Assume standard competitive programming input/output format unless specified otherwise.

The solution MUST:
1. Be correct for all valid inputs
2. Handle edge cases implicitly through correct logic
3. Be efficient enough to satisfy the given constraints
4. Include all necessary imports
5. Be executable as-is

Output Rules:
- Output ONLY valid Python code.
- Do NOT include explanations, comments outside the code, or markdown.
- Place the entire solution in FINAL_ANSWER.
- Once FINAL_ANSWER is produced, the task is complete.

Use available tools to analyze the problem and generate the solution.
Provide your final code solution in FINAL_ANSWER.
"""
    
    return prompt


def infer_taco(json_path: str,
               out_dir: str = None,
               start_sample=0,
               max_samples: int = None,
               output_json: str = "raw_trajectories_taco.json"):
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), "infer_taco")
    os.makedirs(out_dir, exist_ok=True)

    # Load dataset
    print(f"Loading TACO dataset from: {json_path}")
    samples = load_taco_dataset(json_path)
    print(f"Total samples in dataset: {len(samples)}")
    
    if max_samples:
        samples = samples[start_sample:start_sample + max_samples]
        print(f"Processing from {start_sample} to {start_sample + max_samples} samples")

    # Prepare agent
    tools = get_tools()
    agent = MultiToolReActAgent(tools=tools, max_iterations=10, verbose=True)

    trajectories = []
    trajs_path = os.path.join(out_dir, output_json)

    def append_to_json_list(path: str, item: dict):
        """Append an item to a JSON file that contains a list. Create file if missing."""
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as rf:
                    try:
                        data = json.load(rf)
                        if not isinstance(data, list):
                            data = []
                    except Exception:
                        data = []
            else:
                data = []

            data.append(item)

            # Write atomically
            tmp_path = path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as wf:
                json.dump(data, wf, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
        except Exception as e:
            print(f"Failed appending trajectory to {path}: {e}")

    # Process each sample
    for idx, sample in enumerate(samples, 1):
        try:
            # Build prompt
            question = build_prompt(sample)
            
            # Get sample metadata
            source = sample.get("source", "unknown")
            difficulty = sample.get("difficulty", "UNKNOWN")
            url = sample.get("url", "")
            date = sample.get("date", "")
            
            print(f"\n[{idx}/{len(samples)}] Processing sample from {source} (Difficulty: {difficulty})")
            
            # Run agent (no image for code generation task)
            answer, traj = agent.run(question, image=None)

            traj_record = {
                "sample_id": idx - 1,  # 0-indexed
                "source": source,
                "difficulty": difficulty,
                "url": url,
                "date": date,
                "question": sample.get("question", "")[:200] + "...",  # Truncate for brevity
                "tags": sample.get("tags", "[]"),
                "trajectory": traj,
                "generated_code": answer,  # The final answer from agent
            }
            trajectories.append(traj_record)
            
            # Append immediately to trajectories.json to avoid data loss
            try:
                append_to_json_list(trajs_path, traj_record)
                print(f"[{idx}/{len(samples)}] ✓ Successfully processed and saved")
            except Exception as e:
                print(f"[{idx}/{len(samples)}] ✗ Failed to save trajectory: {e}")

        except Exception as e:
            print(f"[{idx}/{len(samples)}] ✗ Error processing sample: {e}")

    print(f"\n{'='*60}")
    print(f"Total trajectories processed: {len(trajectories)}")
    print(f"Results saved to: {trajs_path}")
    print(f"{'='*60}")

    return trajs_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run AWM inference on TACO/code-world samples.")
    parser.add_argument("--json_path", required=True, help="TACO JSON file.")
    parser.add_argument("--out_dir", default="./output/infer_taco", help="Directory for AWM trajectory outputs.")
    parser.add_argument("--start_sample", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--output_json", default="raw_trajectories_taco.json")
    args = parser.parse_args()

    infer_taco(
        json_path=args.json_path,
        out_dir=args.out_dir,
        start_sample=args.start_sample,
        max_samples=args.max_samples,
        output_json=args.output_json,
    )
