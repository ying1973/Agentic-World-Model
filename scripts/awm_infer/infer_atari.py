import os
import json
from typing import Tuple

from awm.tools import get_tools, get_tools_new
from awm.react_gpt import MultiToolReActAgent


def parse_atari_filename(filename: str) -> Tuple[str, str, str, str]:
    name = os.path.splitext(os.path.basename(filename))[0]
    parts = name.split("-")
    
    if len(parts) < 5:
        raise ValueError(f"Invalid filename format: {filename}")
    
    game_name = parts[0]
    sample_id = parts[1]
    # parts[2] should be 'prev'
    action_name = parts[3]
    action_id = parts[4]
    
    return game_name, sample_id, action_name, action_id


def infer_atari(dataset_dir: str,
                out_dir: str = None,
                start_sample=0,
                max_samples: int = None,
                output_json: str = "raw_trajectories_atari.json"):

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), "infer_atari")
    os.makedirs(out_dir, exist_ok=True)

    # Prepare agent
    tools = get_tools()
    agent = MultiToolReActAgent(tools=tools, max_iterations=10, verbose=True)

    # Path to prev directory
    prev_dir = os.path.join(dataset_dir, "prev")
    
    if not os.path.exists(prev_dir):
        raise RuntimeError(f"prev directory not found: {prev_dir}")
    
    # Collect all prev images
    prev_images = []
    for f in os.listdir(prev_dir):
        if f.lower().endswith(".png") and "-prev-" in f:
            prev_images.append(os.path.join(prev_dir, f))
    
    prev_images.sort()
    
    if max_samples:
        prev_images = prev_images[start_sample:start_sample + max_samples]
    
    print(f"Found {len(prev_images)} samples to process")

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

    # Process each image
    for idx, img_path in enumerate(prev_images, 1):
        try:
            game_name, sample_id, action_name, action_id = parse_atari_filename(img_path)
            
            # Build question for agent
            question = (
                f"You are given the current world state (image) from the Atari game '{game_name}'. "
                f"The player is about to execute the action '{action_name}' (action ID: {action_id}). "
                f"Given the current game state and the action to be performed, predict the subsequent game world state after this action is executed. "
                f"Show expected visual changes, game objects' new positions, score changes, and any consequences relevant to the game. "
                f"Always use available tools to support your prediction and provide a clear FINAL_ANSWER."
                f"Note that in the FINAL_ANSWER, you must return the media_id of the image as the final answer content."
            )

            answer, traj = agent.run(question, image=img_path)

            traj_record = {
                "game": game_name,
                "sample_id": sample_id,
                "action_name": action_name,
                "action_id": action_id,
                "image_path": img_path,
                "trajectory": traj,
            }
            trajectories.append(traj_record)
            
            # Append immediately to trajectories.json to avoid data loss
            try:
                append_to_json_list(trajs_path, traj_record)
                print(f"[{idx}/{len(prev_images)}] ✓ Processed: {os.path.basename(img_path)}")
            except Exception as e:
                print(f"[{idx}/{len(prev_images)}] ✗ Failed to save trajectory for {img_path}: {e}")

        except Exception as e:
            print(f"[{idx}/{len(prev_images)}] ✗ Error processing {img_path}: {e}")

    print(f"\nTotal trajectories processed: {len(trajectories)}")
    print(f"Results saved to: {trajs_path}")

    return trajs_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run AWM inference on Atari game-world samples.")
    parser.add_argument("--dataset_dir", required=True, help="Atari dataset directory containing a prev/ subdirectory.")
    parser.add_argument("--out_dir", default="./output/infer_atari", help="Directory for AWM trajectory outputs.")
    parser.add_argument("--start_sample", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--output_json", default="raw_trajectories_atari.json")
    args = parser.parse_args()

    infer_atari(
        dataset_dir=args.dataset_dir,
        out_dir=args.out_dir,
        start_sample=args.start_sample,
        max_samples=args.max_samples,
        output_json=args.output_json,
    )
