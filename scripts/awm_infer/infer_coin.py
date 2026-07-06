import os
import json
import cv2
import tempfile
from typing import Tuple
import random

from awm.tools import get_tools, get_tools_new
from awm.react_gpt import MultiToolReActAgent


def parse_filename(filename: str) -> Tuple[str, str, str]:
    """Parse filenames of form id-task-action.mp4
    Returns (id, task, action)
    If task contains additional dashes, they are preserved.
    """
    name = os.path.splitext(os.path.basename(filename))[0]
    parts = name.split("-")
    if len(parts) < 3:
        # Fallback: put entire name as task, empty action
        return parts[0] if parts else "", "", ""

    vid_id = parts[0]
    action = parts[-1]
    task = "-".join(parts[1:-1])
    return vid_id, task, action


def extract_initial_frame(video_path: str, out_dir: str) -> str:
    """Extract the first readable frame from video and save as PNG.
    Returns path to saved image.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError(f"Cannot read frame from: {video_path}")

    # Convert BGR -> RGB for saving consistently
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    out_dir = out_dir + "/coin_first_frame"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(video_path) + ".png")
    # cv2.imwrite expects BGR, convert back
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, bgr)
    return out_path


def infer_coin(videos_dir: str,
               out_dir: str = None,
               start_sample=0,
               max_videos: int = 5,
               output_json: str = "raw_trajectories_coin.json"):
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), "infer_coin")
    os.makedirs(out_dir, exist_ok=True)

    # Prepare agent
    tools = get_tools()
    agent = MultiToolReActAgent(tools=tools, max_iterations=10, verbose=True)

    videos = []
    for root, _, files in os.walk(videos_dir):
        for f in files:
            if f.lower().endswith((".mp4", ".avi", ".mov")):
                videos.append(os.path.join(root, f))

    videos.sort()
    if max_videos:
        videos = videos[start_sample:start_sample + max_videos]

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


    for vid in videos:
        try:
            vid_id, task, action = parse_filename(vid)
            img_path = extract_initial_frame(vid, out_dir)

            id_options = [
                "media_id of the image",
                "video_id of the video"
            ]
            # Build question for agent
            question = (
                f"You are given the current physical world state (image) which shows the beginning of a action in a task. "
                f"The current task is '{task}' and current action is '{action}'. "
                f"Given the current state and that action, predict the subsequent world state after this action completes. "
                f"Describe expected visual changes, objects' new positions, and any consequences relevant to the task. "
                f"Always use available tools to support your prediction and provide a clear FINAL_ANSWER."
                f"Note that in the FINAL_ANSWER, you must return the {random.choice(id_options)} as the final answer content."
            )

            answer, traj = agent.run(question, image=img_path)

            traj_record = {
                "id": vid_id,
                "video_path": vid,
                "task": task,
                "action": action,
                "trajectory": traj,
            }
            trajectories.append(traj_record)
            # append immediately to trajectories.json to avoid data loss
            try:
                append_to_json_list(trajs_path, traj_record)
            except Exception as e:
                print(f"Failed to append trajectory for {vid}: {e}")

        except Exception as e:
            print(f"Error processing {vid}: {e}")

    # Write aggregated trajectories to a single JSON file
    # try:
    #     with open(trajs_path, "w", encoding="utf-8") as tf:
    #         json.dump(trajectories, tf, ensure_ascii=False, indent=2)
    #     print(f"Trajectories written to: {trajs_path}")
    # except Exception as e:
    #     print(f"Failed writing trajectories: {e}")

    # return trajs_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run AWM inference on COIN/general-world videos.")
    parser.add_argument("--videos_dir", required=True, help="Directory containing COIN benchmark or training videos.")
    parser.add_argument("--out_dir", default="./output/infer_coin", help="Directory for AWM trajectory outputs.")
    parser.add_argument("--start_sample", type=int, default=0)
    parser.add_argument("--max_videos", type=int, default=5)
    parser.add_argument("--output_json", default="raw_trajectories_coin.json")
    args = parser.parse_args()

    infer_coin(
        videos_dir=args.videos_dir,
        out_dir=args.out_dir,
        start_sample=args.start_sample,
        max_videos=args.max_videos,
        output_json=args.output_json,
    )
