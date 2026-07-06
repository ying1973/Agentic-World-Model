import os
import cv2
import torch
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm


import base64

from typing import Optional, Union, List
import json
import requests
import tempfile
# ======================
# Global config
# ======================
DEVICE = "cpu"
CLIP_MODEL_NAME = "ViT-B/32"
IMAGE_SIZE = 256


# ======================
# Load CLIP
# ======================
print("Loading CLIP model...")
clip_model, clip_preprocess = clip.load(CLIP_MODEL_NAME, device=DEVICE)
clip_model.eval()


# ======================
# Utilities
# ======================
def extract_last_k_frames(
    video_path: str,
    k: int = 8,
    resize=(224, 224)
):
    """Extract last K frames from a video."""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, resize)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames extracted from {video_path}")

    return frames[-k:]


@torch.no_grad()
def encode_image_pil(img: Image.Image) -> torch.Tensor:
    """Encode a PIL image into a normalized CLIP feature."""
    img_input = clip_preprocess(img).unsqueeze(0).to(DEVICE)
    feat = clip_model.encode_image(img_input)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0)


def encode_video_last_k_frames(
    video_path: str,
    k: int = 8
) -> torch.Tensor:
    """Encode a video by mean-pooling CLIP features of last K frames."""
    frames = extract_last_k_frames(video_path, k=k, resize=(IMAGE_SIZE, IMAGE_SIZE))

    feats = []
    for frame in frames:
        img = Image.fromarray(frame)
        feat = encode_image_pil(img)
        feats.append(feat)

    feats = torch.stack(feats, dim=0)  # (K, D)
    video_feat = feats.mean(dim=0)
    video_feat = video_feat / video_feat.norm()

    return video_feat


def encode_generated_image(image_path: str) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    return encode_image_pil(img)


def cosine_similarity(f1: torch.Tensor, f2: torch.Tensor) -> float:
    return torch.dot(f1, f2).item()


# ======================
# Dataset-level evaluation
# ======================
def evaluate_video_image_dirs(
    video_dir: str,
    image_dir: str,
    k: int = 8,
    video_exts=(".mp4", ".avi", ".mov"),
    image_exts=(".png", ".jpg", ".jpeg")
):
    scores = []

    video_files = [
        f for f in os.listdir(video_dir)
        if f.lower().endswith(video_exts)
    ]
    video_files.sort()

    for vf in tqdm(video_files, desc="Evaluating"):
        stem = os.path.splitext(vf)[0]

        # find matching image
        img_path = None
        for ext in image_exts:
            candidate = os.path.join(image_dir, stem + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            print(f"[Warning] No matching image for {vf}, skipped.")
            continue

        video_path = os.path.join(video_dir, vf)

        try:
            video_feat = encode_video_last_k_frames(video_path, k)
            image_feat = encode_generated_image(img_path)
            score = cosine_similarity(video_feat, image_feat)
            scores.append(score)
        except Exception as e:
            print(f"[Error] {vf}: {e}")

    if len(scores) == 0:
        raise RuntimeError("No valid video-image pairs evaluated.")

    return {
        "num_samples": len(scores),
        "mean_similarity": float(np.mean(scores)),
        "std_similarity": float(np.std(scores)),
        "all_scores": scores
    }



def build_gpt_video_image_prompt(num_frames: int) -> str:
    return f"""
You are an expert evaluator for action understanding and world-state prediction.

You will be given {num_frames + 1} images in total:

- The FIRST {num_frames} images are frames sampled from the REAL (ground-truth) video,
  ordered chronologically, showing an action being performed.
- The LAST image is a GENERATED image, which represents the FINAL world state
  after the action has been completed.

Your task is to evaluate how well the GENERATED image matches the REAL video
in terms of the completed action and final outcome.

Please focus on the following aspects (in order of importance):

1. Action completion and goal achievement:
   - Does the generated image reflect the SAME completed action as in the real video?
   - Does it reach the same final goal or outcome?
   - If the action is incomplete, incorrect, or deviates from the real video,
     the score should be LOW, even if the image looks realistic.

2. Object and state correctness:
   - Are the key objects present and in the correct final states?
   - Are object positions, relationships, and states consistent with the real video outcome?

3. Physical plausibility:
   - Is the final state physically plausible given the observed action?

4. Visual realism:
   - Is the generated image visually realistic and coherent?

Important rules:
- Judge based on the FINAL outcome implied by the real video.
- Do NOT reward images that look realistic but imply a different action.
- Do NOT hallucinate outcomes not supported by the video.

Give a single numeric score from 0 to 10:
- 10: The generated image correctly reflects the completed action and final state.
- 7–9: Mostly correct with minor deviations.
- 4–6: Partially correct or incomplete.
- 1–3: Largely incorrect.
- 0: Completely wrong or unrelated.

Only output the number. Do not explain.
"""

token = os.environ.get("MWB_EVAL_API_KEY", "")

def _require_token(value: str, env_name: str = "MWB_EVAL_API_KEY") -> str:
    if not value:
        raise RuntimeError(f"Missing evaluator API token. Please set {env_name}.")
    return value

def _require_endpoint(value: str, env_name: str = "MWB_EVAL_API_URL") -> str:
    if not value:
        raise RuntimeError(f"Missing evaluator API endpoint. Please set {env_name}.")
    return value

def encode_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def llm(prompt: str,model: str = "gpt-5.1-2025-11-13",image_path: Optional[Union[str, List[str]]] = None,max_tokens: int = 1688,temperature: float = 0.5
):
    url = _require_endpoint(os.environ.get("MWB_EVAL_API_URL", ""))

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

    response = requests.post(url, headers=headers, data=payload)
    response.raise_for_status()

    result = response.json()["choices"][0]["message"]["content"]
    return result


def extract_uniform_frames(
    video_path: str,
    num_frames: int,
    resize=(224, 224)
):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise RuntimeError(f"Invalid video: {video_path}")

    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)

    frame_id = 0
    target_ptr = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id == indices[target_ptr]:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, resize)
            frames.append(frame)
            target_ptr += 1
            if target_ptr >= len(indices):
                break

        frame_id += 1

    cap.release()

    if len(frames) != num_frames:
        raise RuntimeError(f"Expected {num_frames} frames, got {len(frames)}")

    return frames


def save_frames_to_temp_images(frames):
    temp_paths = []
    for frame in frames:
        img = Image.fromarray(frame)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(tmp.name)
        temp_paths.append(tmp.name)
    return temp_paths


def compute_gptscore_video_image(
    video_path: str,
    image_path: str,
    num_frames: int = 4,
    model: str = "gpt-5.1-2025-11-13"
) -> float:
    """
    GPT-based score in [0, 10], using:
      - num_frames real video frames
      - 1 generated final-state image
    """
    # 1. sample real video frames
    real_frames = extract_uniform_frames(video_path, num_frames=num_frames)
    real_frame_paths = save_frames_to_temp_images(real_frames)

    # 2. build prompt
    prompt = build_gpt_video_image_prompt(num_frames)

    # 3. image list: real frames + ONE generated image
    image_paths = real_frame_paths + [image_path]

    # 4. call LLM
    result = llm(
        prompt=prompt,
        model=model,
        image_path=image_paths
    )

    # 5. parse score
    try:
        score = float(result.strip())
    except ValueError:
        raise RuntimeError(f"Invalid GPT output: {result}")

    return score


def evaluate_gptscore_video_image_dirs(
    video_dir: str,
    image_dir: str,
    num_frames: int = 4,
    model: str = "gpt-5.1-2025-11-13",
    video_exts=(".mp4", ".avi", ".mov"),
    image_exts=(".png", ".jpg", ".jpeg"),
):
    """
    Directory-level GPTScore evaluation.

    Returns:
        {
            "num_samples": int,
            "mean_gptscore": float,
            "std_gptscore": float,
            "scores": List[float],
            "per_sample": Dict[str, float]
        }
    """
    scores = []
    per_sample_scores = {}

    video_files = [
        f for f in os.listdir(video_dir)
        if f.lower().endswith(video_exts)
    ]
    video_files.sort()

    for vf in tqdm(video_files, desc="Evaluating GPTScore"):
        stem = os.path.splitext(vf)[0]
        video_path = os.path.join(video_dir, vf)

        # find matching generated image
        image_path = None
        for ext in image_exts:
            candidate = os.path.join(image_dir, stem + ext)
            if os.path.exists(candidate):
                image_path = candidate
                break

        if image_path is None:
            print(f"[Warning] No generated image for {vf}, skipped.")
            continue

        try:
            score = compute_gptscore_video_image(
                video_path=video_path,
                image_path=image_path,
                num_frames=num_frames,
                model=model
            )
            score = float(score)
            print(f"[Info] GPTScore for {stem}: {score}")
            scores.append(score)
            per_sample_scores[stem] = score

        except Exception as e:
            print(f"[Error] GPTScore failed for {stem}: {e}")

    if len(scores) == 0:
        raise RuntimeError("No valid samples evaluated for GPTScore.")

    return {
        "num_samples": len(scores),
        "mean_gptscore": float(np.mean(scores)),
        "std_gptscore": float(np.std(scores)),
        "scores": scores,
        "per_sample": per_sample_scores
    }

# ======================
# Main
# ======================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()

    results = evaluate_video_image_dirs(
        video_dir=args.video_dir,
        image_dir=args.image_dir,
        k=args.k
    )

    print("\n===== Evaluation Results =====")
    print(f"Samples      : {results['num_samples']}")
    print(f"Mean Similarity : {results['mean_similarity']:.4f}")
    print(f"Std Similarity  : {results['std_similarity']:.4f}")

    gpt_result = evaluate_gptscore_video_image_dirs(
        video_dir=args.video_dir,
        image_dir=args.image_dir,
        num_frames=5
    )
    print("===== GPTScore Results =====")
    print("Samples:", gpt_result["num_samples"])
    print("Mean GPTScore:", gpt_result["mean_gptscore"])
    print("Std GPTScore:", gpt_result["std_gptscore"])