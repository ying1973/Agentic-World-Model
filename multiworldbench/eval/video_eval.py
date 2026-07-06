import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import base64

from typing import Optional, Union, List
import json
import requests
import tempfile


from multiworldbench.eval.common_metrics_on_video_quality.calculate_fvd import calculate_fvd
from multiworldbench.eval.common_metrics_on_video_quality.calculate_psnr import calculate_psnr
from multiworldbench.eval.common_metrics_on_video_quality.calculate_ssim import calculate_ssim
from multiworldbench.eval.common_metrics_on_video_quality.calculate_lpips import calculate_lpips

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ====== 参数区 ======
print("当前任务是：droid数据集")
REAL_DIR = None
GEN_DIR = None

NUM_FRAMES = 32              # 统一帧数（推荐 32）
IMG_SIZE = (64, 64)          # (W, H)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
only_final = True

token = os.environ.get("MWB_EVAL_API_KEY", "")

def _require_token(value: str, env_name: str = "MWB_EVAL_API_KEY") -> str:
    if not value:
        raise RuntimeError(f"Missing evaluator API token. Please set {env_name}.")
    return value

def _require_endpoint(value: str, env_name: str = "MWB_EVAL_API_URL") -> str:
    if not value:
        raise RuntimeError(f"Missing evaluator API endpoint. Please set {env_name}.")
    return value
# ====================


def uniform_sample_frames(frames, num_frames):
    """
    frames: np.ndarray [T_orig, H, W, 3]
    return: np.ndarray [num_frames, H, W, 3]
    """
    T_orig = len(frames)

    if T_orig >= num_frames:
        # 均匀采样（关键修改点）
        indices = np.linspace(0, T_orig - 1, num_frames)
        indices = np.round(indices).astype(np.int64)
        sampled = frames[indices]
    else:
        # padding：重复最后一帧
        pad_len = num_frames - T_orig
        pad = np.repeat(frames[-1][None], pad_len, axis=0)
        sampled = np.concatenate([frames, pad], axis=0)

    return sampled


def load_and_process_video(
    video_path,
    num_frames=32,
    target_size=(64, 64),
):
    """
    返回：
        Tensor [T, C, H, W], float32 in [0,1]
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"Failed to read video: {video_path}")

    frames = np.array(frames)  # [T_orig, H, W, 3]

    # ===== 核心修改：统一用均匀采样 =====
    frames = uniform_sample_frames(frames, num_frames)

    # ===== resize + normalize =====
    processed = []
    for f in frames:
        f = cv2.resize(f, target_size)
        f = f.astype(np.float32) / 255.0
        processed.append(f)

    video = np.stack(processed)            # [T, H, W, 3]
    video = torch.from_numpy(video)
    video = video.permute(0, 3, 1, 2)      # [T, C, H, W]

    return video


def load_video_pairs(real_dir, gen_dir):
    """
    返回：
        real_videos: [B, T, C, H, W]
        gen_videos:  [B, T, C, H, W]
    """
    filenames = sorted(os.listdir(real_dir))
    filenames = [
        f for f in filenames
        if os.path.isfile(os.path.join(gen_dir, f))
    ]

    assert len(filenames) > 0, "No matched video pairs found."

    real_videos = []
    gen_videos = []

    for fname in tqdm(filenames, desc="Loading video pairs"):
        real_path = os.path.join(real_dir, fname)
        gen_path  = os.path.join(gen_dir, fname)

        real_vid = load_and_process_video(
            real_path,
            num_frames=NUM_FRAMES,
            target_size=IMG_SIZE,
        )
        gen_vid = load_and_process_video(
            gen_path,
            num_frames=NUM_FRAMES,
            target_size=IMG_SIZE,
        )

        real_videos.append(real_vid)
        gen_videos.append(gen_vid)

    real_videos = torch.stack(real_videos, dim=0)  # [B, T, C, H, W]
    gen_videos  = torch.stack(gen_videos,  dim=0)

    return real_videos, gen_videos


def main(real_dir: str, gen_dir: str, num_frames: int = NUM_FRAMES, img_size=IMG_SIZE, device: str = DEVICE, only_final_frame: bool = only_final):
    print("Preparing video tensors...")
    real_videos, gen_videos = load_video_pairs(real_dir, gen_dir)

    print(f"Video tensor shape: {real_videos.shape}")

    real_videos = real_videos.to(device)
    gen_videos  = gen_videos.to(device)

    # ===== 调用 common_metrics_on_video_quality =====
    # 确保你已将该 repo 加入 PYTHONPATH
    print("Computing metrics...")

    lpips_score = calculate_lpips(real_videos, gen_videos, device=device, only_final=only_final_frame)
    psnr_score  = calculate_psnr(real_videos, gen_videos, only_final=only_final_frame)
    ssim_score  = calculate_ssim(real_videos, gen_videos, only_final=only_final_frame)
    fvd_score   = calculate_fvd(real_videos, gen_videos, device=device, method='styleganv', only_final=only_final_frame)

    def metric_to_float(x):
        """Try to convert metric result to a float for formatting.

        Supports: float, int, torch.Tensor, dict (tries common keys or mean of values).
        Returns None if conversion not possible.
        """
        # torch tensor
        try:
            import torch as _torch
        except Exception:
            _torch = None

        if _torch is not None and isinstance(x, _torch.Tensor):
            try:
                return float(x.detach().cpu().item())
            except Exception:
                return None

        # numeric
        if isinstance(x, (int, float)):
            return float(x)

        # dict: try common keys then mean of numeric values
        if isinstance(x, dict):
            for k in ("mean", "score", "value", "val"):
                if k in x:
                    try:
                        return float(x[k])
                    except Exception:
                        try:
                            v = x[k]
                            if hasattr(v, "item"):
                                return float(v.item())
                        except Exception:
                            pass
            # fallback: mean of values
            try:
                vals = []
                for v in x.values():
                    if hasattr(v, "item"):
                        vals.append(float(v.item()))
                    else:
                        vals.append(float(v))
                if len(vals) > 0:
                    return float(np.mean(vals))
            except Exception:
                return None

        # last attempt
        try:
            return float(x)
        except Exception:
            return None

    lpips_v = metric_to_float(lpips_score)
    psnr_v = metric_to_float(psnr_score)
    ssim_v = metric_to_float(ssim_score)
    fvd_v = metric_to_float(fvd_score)

    print("\n===== Evaluation Results =====")
    if lpips_v is None:
        print(f"LPIPS: {lpips_score}")
    else:
        print(f"LPIPS: {lpips_v:.4f}")

    if psnr_v is None:
        print(f"PSNR : {psnr_score}")
    else:
        print(f"PSNR : {psnr_v:.4f}")

    if ssim_v is None:
        print(f"SSIM : {ssim_score}")
    else:
        print(f"SSIM : {ssim_v:.4f}")

    if fvd_v is None:
        print(f"FVD  : {fvd_score}")
    else:
        print(f"FVD  : {fvd_v:.4f}")




def encode_base64(image_path: str) -> str:
    """
    Encode an image file (given by path) to a base64 string.
    """
    with open(image_path, "rb") as _f:
        data = _f.read()
    return base64.b64encode(data).decode()

def llm(prompt: str,model: str = "gpt-5.1",image_path: Optional[Union[str, List[str]]] = None,max_tokens: int = 1688,temperature: float = 0.5
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

def extract_frames(video_path: str,num_frames: int = 4,resize=(256, 256)) -> List[str]:
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Cannot open video: {video_path}"

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert total > 0, f"Empty video: {video_path}"

    indices = [
        int((i + 0.5) * total / num_frames)
        for i in range(num_frames)
    ]

    tmp_dir = tempfile.mkdtemp(prefix="gpt_frames_")
    frame_paths = []

    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, resize)

        out_path = os.path.join(tmp_dir, f"{i}.png")
        cv2.imwrite(out_path, frame[:, :, ::-1])
        frame_paths.append(out_path)

    cap.release()
    return frame_paths

def build_gpt_video_prompt(num_frames: int) -> str:
    return f"""
You are an expert evaluator for video generation quality, especially for
action-conditional and world-model-based video generation.

You will be given {num_frames * 2} images in total:

- The FIRST {num_frames} images are frames sampled from the REAL (ground-truth) video,
  ordered chronologically.
- The NEXT {num_frames} images are frames sampled from the GENERATED video,
  at the same timestamps and in the same order.

Your task is to evaluate how well the GENERATED video matches the REAL video
by considering ALL frames together.

Please focus on the following aspects (in order of importance):

1. Action completion and goal achievement:
   - Does the generated video perform the SAME action as the real video?
   - Does it reach the same final goal or outcome?
   - If the action is incomplete, incorrect, or deviates from the real video,
     the score should be LOW, even if the visuals look realistic.

2. Temporal and object consistency:
   - Are objects, identities, and their states consistent across frames?
   - Are there unnatural jumps, disappearances, or contradictions over time?

3. Physical and motion plausibility:
   - Are motions physically plausible and coherent across frames?
   - Do interactions with objects follow common-sense physics?

4. Visual realism:
   - Are the frames visually realistic in terms of texture, lighting, and detail?

Important rules:
- Judge the video as a whole, not individual frames.
- Do NOT reward videos that look realistic but perform a different action.
- Do NOT infer missing actions that are not supported by the frames.

Give a single numeric score from 0 to 10:
- 10: The generated video matches the real video very well AND completes the same action.
- 7–9: The action is mostly correct with minor deviations.
- 4–6: The action is partially correct or incomplete.
- 1–3: The action is largely incorrect or inconsistent.
- 0: The action is completely wrong or absent.

Only output the number. Do not explain.
"""

def compute_gpt_score_for_video(
    real_video_path: str,
    gen_video_path: str,
    num_frames: int = 4,
    resize=(256, 256),
    model: str = "gpt-5.1-2025-11-13"
) -> Optional[float]:
    real_frames = extract_frames(real_video_path, num_frames, resize)
    gen_frames = extract_frames(gen_video_path, num_frames, resize)

    assert len(real_frames) == len(gen_frames), "Frame count mismatch"

    prompt = build_gpt_video_prompt(num_frames)
    all_images = real_frames + gen_frames

    try:
        result = llm(
            prompt=prompt,
            model=model,
            image_path=all_images,
            temperature=0.0
        )
        raw = result.strip()
        print(f"GPT-Score raw output: {raw}")
        score = float(raw)
        score = max(0.0, min(10.0, score))
        return score

    except Exception:
        print("[GPT-Score] Failed to parse score, returning None")
        return None


def compute_gpt_score_dataset(
    real_dir: str,
    gen_dir: str,
    num_frames: int = 5,
    resize=(256, 256)
) -> float:
    names = sorted(os.listdir(real_dir))
    scores = []
    num_failed = 0

    for name in names:
        real_path = os.path.join(real_dir, name)
        gen_path = os.path.join(gen_dir, name)

        if not os.path.exists(gen_path):
            print(f"[GPT-Score] Missing: {name}")
            continue

        print(f"[GPT-Score] Evaluating {name}")
        score = compute_gpt_score_for_video(
            real_path,
            gen_path,
            num_frames=num_frames,
            resize=resize
        )

        if score is None:
            continue

        scores.append(score)

    if len(scores) == 0:
        print("[GPT-Score] No valid scores!")
        return 0.0

    avg_score = sum(scores) / len(scores)

    print(
        f"[GPT-Score] Valid videos: {len(scores)}, "
        f"Failed parses: {num_failed}, "
        f"Average score: {avg_score:.4f}"
    )

    return avg_score


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate generated videos against reference videos.")
    parser.add_argument("--real_dir", required=True, help="Directory containing reference videos.")
    parser.add_argument("--gen_dir", required=True, help="Directory containing generated videos.")
    parser.add_argument("--num_frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--img_size", type=int, nargs=2, default=list(IMG_SIZE))
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--all_frames", action="store_true", help="Use all sampled frames for frame metrics instead of only the final frame.")
    parser.add_argument("--gpt_score", action="store_true", help="Also compute LLM semantic score. Requires MWB_EVAL_API_KEY.")
    parser.add_argument("--gpt_num_frames", type=int, default=5)
    parser.add_argument("--gpt_resize", type=int, nargs=2, default=[256, 256])
    args = parser.parse_args()

    main(
        real_dir=args.real_dir,
        gen_dir=args.gen_dir,
        num_frames=args.num_frames,
        img_size=tuple(args.img_size),
        device=args.device,
        only_final_frame=not args.all_frames,
    )
    if args.gpt_score:
        compute_gpt_score_dataset(
            real_dir=args.real_dir,
            gen_dir=args.gen_dir,
            num_frames=args.gpt_num_frames,
            resize=tuple(args.gpt_resize),
        )
