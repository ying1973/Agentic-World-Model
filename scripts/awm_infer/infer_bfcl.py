import os
import json
from typing import Dict, Any, List, Tuple

from awm.tools import get_tools, get_tools_new
from awm.react_gpt import MultiToolReActAgent


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def merge_question_groundtruth(questions: List[Dict], groundtruths: List[Dict]) -> List[Dict]:
    # Create a dictionary for quick lookup
    gt_dict = {item["id"]: item for item in groundtruths}
    
    merged = []
    for q in questions:
        sample_id = q["id"]
        if sample_id in gt_dict:
            merged.append({
                "id": sample_id,
                "question": q.get("question", []),
                "function": q.get("function", []),
                "ground_truth": gt_dict[sample_id].get("ground_truth", [])
            })
        else:
            print(f"Warning: No ground truth found for ID {sample_id}")
            merged.append({
                "id": sample_id,
                "question": q.get("question", []),
                "function": q.get("function", []),
                "ground_truth": []
            })
    
    return merged


def build_prompt(sample: Dict[str, Any]) -> str:
    sample_id = sample.get("id", "unknown")
    question_data = sample.get("question", [[]])
    functions = sample.get("function", [])
    
    # Extract user question from the nested structure
    user_question = ""
    if question_data and len(question_data) > 0:
        for msg in question_data[0]:
            if msg.get("role") == "user":
                user_question = msg.get("content", "")
                break
    
    # Format available functions
    functions_str = ""
    if functions:
        functions_str = "\n\nAvailable Functions:\n"
        for i, func in enumerate(functions, 1):
            func_name = func.get("name", "unknown")
            func_desc = func.get("description", "No description")
            func_params = func.get("parameters", {})
            
            functions_str += f"\n{i}. Function: {func_name}\n"
            functions_str += f"   Description: {func_desc}\n"
            functions_str += f"   Parameters:\n"
            
            properties = func_params.get("properties", {})
            required = func_params.get("required", [])
            
            for param_name, param_info in properties.items():
                param_type = param_info.get("type", "unknown")
                param_desc = param_info.get("description", "No description")
                is_required = "required" if param_name in required else "optional"
                
                functions_str += f"     - {param_name} ({param_type}, {is_required}): {param_desc}\n"
                
                # Handle array items
                if param_type == "array" and "items" in param_info:
                    items_type = param_info["items"].get("type", "unknown")
                    functions_str += f"       (array of {items_type})\n"


    prompt = f"""You are a ROUTING MODEL in a multi-agent system. Your ONLY role is to analyze tasks and delegate them to specialized world models (tools). You MUST NOT solve the task directly.

CRITICAL INSTRUCTIONS:
1. You are NOT allowed to directly answer or solve the user's task
2. You MUST select an appropriate world model (tool) to handle the task
3. You MUST pass the complete task information (user request + available functions) to the selected world model
4. The world model will analyze the functions and generate the function call
5. You will receive the world model's response and can iterate if needed

WORKFLOW:
Step 1: Analyze the user request and available functions
Step 2: Select the most appropriate world model:
   - Text generation model: for pure text-based function calling tasks
   - Image editing model: for image manipulation tasks
   - Video generation model: for video-related tasks
Step 3: Delegate the ENTIRE task to that world model by using the tool
Step 4: Wait for the world model's response
Step 5: If the response is complete, provide FINAL_ANSWER; otherwise, continue reasoning

USER REQUEST:
{user_question}
{functions_str}

TASK FOR THE WORLD MODEL:
You must pass this complete information to the selected world model:
- User Request: {user_question}
- Available Functions: (all functions listed above)
- Required Output: JSON format function call(s)

The world model should analyze which function(s) to call and generate output in this format:
{{
    "function_name": {{
        "parameter1": value1,
        "parameter2": value2
    }}
}}

REMEMBER: 
- DO NOT generate the function call yourself
- DO NOT directly answer the user's question
- You MUST use a tool (world model) to process this task
- Pass the complete context to the world model

Now, select the appropriate world model and delegate this function calling task to it."""

    return prompt


def infer_bfcl(question_json_path: str,
               groundtruth_json_path: str,
               out_dir: str = None,
               start_sample = 0,
               max_samples: int = None,
               output_json: str = "raw_trajectories_bfcl.json"):
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), "infer_bfcl")
    os.makedirs(out_dir, exist_ok=True)

    # Load datasets
    print(f"Loading questions from: {question_json_path}")
    questions = load_jsonl(question_json_path)
    print(f"Total questions: {len(questions)}")
    
    print(f"Loading ground truths from: {groundtruth_json_path}")
    groundtruths = load_jsonl(groundtruth_json_path)
    print(f"Total ground truths: {len(groundtruths)}")
    
    # Merge question and ground truth
    samples = merge_question_groundtruth(questions, groundtruths)
    print(f"Total merged samples: {len(samples)}")
    
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
            sample_id = sample.get("id", "unknown")
            
            # Build prompt
            question = build_prompt(sample)
            
            # Extract user request for display
            question_data = sample.get("question", [[]])
            user_request = ""
            if question_data and len(question_data) > 0:
                for msg in question_data[0]:
                    if msg.get("role") == "user":
                        user_request = msg.get("content", "")[:100]
                        break
            
            print(f"\n[{idx}/{len(samples)}] Processing: {sample_id}")
            print(f"  User request: {user_request}...")
            
            # Run agent (no image for function calling task)
            answer, traj = agent.run(question, image=None)

            traj_record = {
                "id": sample_id,
                "user_request": user_request,
                "available_functions": [f.get("name") for f in sample.get("function", [])],
                "ground_truth": sample.get("ground_truth", []),
                "trajectory": traj,
                "predicted_call": answer,  # The final answer from agent
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

    parser = argparse.ArgumentParser(description="Run AWM inference on BFCL/tool-world samples.")
    parser.add_argument("--question_json_path", required=True, help="BFCL question JSONL file.")
    parser.add_argument("--groundtruth_json_path", required=True, help="BFCL ground-truth JSONL file.")
    parser.add_argument("--out_dir", default="./output/infer_bfcl", help="Directory for AWM trajectory outputs.")
    parser.add_argument("--start_sample", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--output_json", default="raw_trajectories_bfcl.json")
    args = parser.parse_args()

    infer_bfcl(
        question_json_path=args.question_json_path,
        groundtruth_json_path=args.groundtruth_json_path,
        out_dir=args.out_dir,
        start_sample=args.start_sample,
        max_samples=args.max_samples,
        output_json=args.output_json,
    )
