import os
import json
from typing import Dict, List
import random

from awm.tools import get_tools
from awm.react_gpt import MultiToolReActAgent


def load_webdreamer_data(json_path: str) -> List[Dict]:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle both list format and dict format
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "samples" in data:
            return data["samples"]
        else:
            print(f"Warning: Unexpected data format in {json_path}")
            return []
    except Exception as e:
        print(f"Error loading JSON file {json_path}: {e}")
        return []


def build_webdreamer_prompt(sample_prompt: str, action: str) -> str:
    id_options = [
        "only media_id of the image",
        "only text description"
    ]
    question = (
        f"You are analyzing a web page interaction scenario. "
        f"{sample_prompt} "
        # f"Based on the current web page state shown in the image and the described action, "
        # f"predict the subsequent web page state after this action is executed. "
        f"Your prediction should include: "
        f"(1) Visual changes on the webpage (new elements appearing, elements disappearing, or moving); "
        f"(2) Changes in the page layout or structure; "
        f"(3) Any new interactive elements or content that would become visible; "
        f"Always use available tools to support your prediction and provide a clear FINAL_ANSWER."
        f"Note that in the FINAL_ANSWER, you must return {random.choice(id_options)} as the final answer content."
    )
    
    return question


def infer_webdreamer(json_path: str,
                     data_dir: str = None,
                     out_dir: str = None,
                     start_sample = 0,
                     max_samples: int = None,
                     output_json: str = "raw_trajectories_webdreamer.json"):

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), "infer_webdreamer")
    os.makedirs(out_dir, exist_ok=True)
    
    # Determine data directory
    if data_dir is None:
        data_dir = os.path.dirname(json_path)
    
    # Prepare agent
    tools = get_tools()
    agent = MultiToolReActAgent(tools=tools, max_iterations=10, verbose=True)
    
    # Load samples
    print(f"Loading samples from: {json_path}")
    samples = load_webdreamer_data(json_path)
    print(f"Loaded {len(samples)} samples")
    
    if max_samples:
        samples = samples[start_sample:start_sample + max_samples]
        print(f"Processing from {start_sample} to {start_sample + max_samples} samples")
    
    trajectories = []
    trajs_path = os.path.join(out_dir, output_json)
    
    def append_to_json_list(path: str, item: dict):
        """Append an item to a JSON file that contains a list. Create file if missing."""
        try:
            if os.path.exists(path):
                # read existing
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

            # write atomically
            tmp_path = path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as wf:
                json.dump(data, wf, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
        except Exception as e:
            print(f"Failed appending trajectory to {path}: {e}")
    
    # Process each sample
    for idx, sample in enumerate(samples):
        try:
            sample_id = sample.get("id", f"sample_{idx}")
            sample_prompt = sample.get("prompt", "")
            action = sample.get("action", "UNKNOWN")
            image_rel_path = sample.get("image", "")
            
            # Construct absolute image path
            if os.path.isabs(image_rel_path):
                image_path = image_rel_path
            else:
                image_path = os.path.join(data_dir, image_rel_path)
            
            # Check if image exists
            if not os.path.exists(image_path):
                print(f"Warning: Image not found for sample {sample_id}: {image_path}")
                continue
            
            # Build the inference prompt
            question = build_webdreamer_prompt(sample_prompt, action)
            
            print(f"\n{'='*60}")
            print(f"Processing sample {idx+1}/{len(samples)}: {sample_id}")
            print(f"Action: {action}")
            print(f"Image: {image_path}")
            print(f"{'='*60}")
            
            # Run agent inference
            answer, traj = agent.run(question, image=image_path)
            
            traj_record = {
                "id": sample_id,
                "original_prompt": sample_prompt,
                "action": action,
                "image_path": image_path,
                "inference_prompt": question,
                "trajectory": traj,
            }
            trajectories.append(traj_record)
            
            # Append immediately to trajectories.json to avoid data loss
            try:
                append_to_json_list(trajs_path, traj_record)
                print(f"✓ Saved trajectory for {sample_id}")
            except Exception as e:
                print(f"✗ Failed to append trajectory for {sample_id}: {e}")
        
        except Exception as e:
            print(f"✗ Error processing sample {sample.get('id', idx)}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"Processed {len(trajectories)} samples successfully")
    print(f"Trajectories saved to: {trajs_path}")
    print(f"{'='*60}")
    
    return trajs_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run AWM inference on WebDreamer/GUI-world samples.")
    parser.add_argument("--json_path", required=True, help="WebDreamer samples JSON file.")
    parser.add_argument("--data_dir", default=None, help="Base directory for relative image paths in the JSON file.")
    parser.add_argument("--out_dir", default="./output/infer_webdreamer", help="Directory for AWM trajectory outputs.")
    parser.add_argument("--start_sample", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--output_json", default="raw_trajectories_webdreamer.json")
    args = parser.parse_args()

    infer_webdreamer(
        json_path=args.json_path,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        start_sample=args.start_sample,
        max_samples=args.max_samples,
        output_json=args.output_json,
    )
