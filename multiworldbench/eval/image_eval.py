import os
import argparse
import shutil
import tempfile
import numpy as np
from PIL import Image
from tqdm import tqdm
import base64
import json
import requests

from typing import Optional, Tuple, Union, List

import torch
import torchvision.transforms as T

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import lpips
from pytorch_fid.fid_score import calculate_fid_given_paths


# ------------------------------------------------
# Utils
# ------------------------------------------------
def load_image(path, resize=None):
    img = Image.open(path).convert("RGB")
    if resize is not None:
        img = img.resize(resize, Image.BICUBIC)
    return img


def pil_to_tensor(img):
    return T.ToTensor()(img)  # [0,1]


def prepare_fid_dir(src_dir, dst_dir, resize):
    os.makedirs(dst_dir, exist_ok=True)
    for fname in sorted(os.listdir(src_dir)):
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)

        img = Image.open(src_path).convert("RGB")
        img = img.resize(resize, Image.BICUBIC)
        img.save(dst_path)


# ------------------------------------------------
# Main evaluation
# ------------------------------------------------
def evaluate(
    ref_dir,
    gen_dir,
    device="cuda",
    resize=None,
    fid_resize=(256, 256),
):
    # 仅获取ref_dir下的文件并排序，作为基准列表
    ref_files = sorted(os.listdir(ref_dir))
    # 过滤有效图片文件（可选：可添加后缀过滤，如.jpg/.png等，保持与参考逻辑一致）
    image_suffixes = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    ref_files = sorted([
        f for f in ref_files
        if f.lower().endswith(image_suffixes)
    ])

    print(f"Found {len(ref_files)} reference images in {ref_dir}")

    # LPIPS model
    lpips_model = lpips.LPIPS(net="alex").to(device)
    lpips_model.eval()

    # 初始化评分列表，仅存储有效匹配文件的分数
    lpips_scores = []
    ssim_scores = []
    psnr_scores = []
    # 存储用于FID计算的有效文件名（避免缺失文件影响FID）
    valid_filenames = []

    # ---------- Pairwise metrics ----------
    for fname in tqdm(ref_files, desc="Computing LPIPS / SSIM / PSNR"):
        ref_path = os.path.join(ref_dir, fname)
        gen_path = os.path.join(gen_dir, fname)

        # 检查gen_dir中是否存在同名文件，不存在则告警并跳过
        if not os.path.exists(gen_path):
            print(f"[WARN] Missing generated image: {fname} (skipped)")
            continue

        # 加载图片
        ref_img = load_image(ref_path, resize)
        gen_img = load_image(gen_path, resize)

        # SSIM / PSNR (numpy, [0,255])
        ref_np = np.array(ref_img)
        gen_np = np.array(gen_img)

        ssim_val = ssim(
            ref_np,
            gen_np,
            channel_axis=-1,
            data_range=255
        )
        psnr_val = psnr(
            ref_np,
            gen_np,
            data_range=255
        )

        # LPIPS ([−1,1])
        ref_tensor = pil_to_tensor(ref_img).unsqueeze(0).to(device) * 2 - 1
        gen_tensor = pil_to_tensor(gen_img).unsqueeze(0).to(device) * 2 - 1

        with torch.no_grad():
            lpips_val = lpips_model(ref_tensor, gen_tensor).item()

        # 仅将有效分数加入列表
        ssim_scores.append(ssim_val)
        psnr_scores.append(psnr_val)
        lpips_scores.append(lpips_val)
        # 记录有效文件名，用于后续FID计算
        valid_filenames.append(fname)

    # 检查是否有有效匹配的文件
    if len(valid_filenames) == 0:
        print("[ERROR] No valid image pairs found between ref_dir and gen_dir!")
        return {
            "LPIPS (↓)": float('nan'),
            "SSIM (↑)": float('nan'),
            "PSNR (↑)": float('nan'),
            "FID (↓)": float('nan'),
        }

    print(f"Successfully processed {len(valid_filenames)} valid image pairs")

    # ---------- FID (with resize,仅使用有效匹配的文件) ----------
    print(f"Preparing resized images for FID: {fid_resize}")

    fid_ref_dir = tempfile.mkdtemp(prefix="fid_ref_")
    fid_gen_dir = tempfile.mkdtemp(prefix="fid_gen_")

    # 仅复制有效匹配的文件到临时FID目录
    for fname in valid_filenames:
        # 复制参考图片
        src_ref = os.path.join(ref_dir, fname)
        dst_ref = os.path.join(fid_ref_dir, fname)
        img = Image.open(src_ref).convert("RGB")
        img = img.resize(fid_resize, Image.Resampling.LANCZOS)
        img.save(dst_ref)

        # 复制生成图片
        src_gen = os.path.join(gen_dir, fname)
        dst_gen = os.path.join(fid_gen_dir, fname)
        img = Image.open(src_gen).convert("RGB")
        img = img.resize(fid_resize, Image.Resampling.LANCZOS)
        img.save(dst_gen)

    print("Computing FID...")
    fid_score_val = calculate_fid_given_paths(
        paths=[fid_ref_dir, fid_gen_dir],
        batch_size=50,
        device=device,
        dims=2048,
    )

    # cleanup临时目录
    shutil.rmtree(fid_ref_dir)
    shutil.rmtree(fid_gen_dir)

    # ---------- Results (计算有效文件的平均值) ----------
    results = {
        "LPIPS (↓)": float(np.mean(lpips_scores)),
        "SSIM (↑)": float(np.mean(ssim_scores)),
        "PSNR (↑)": float(np.mean(psnr_scores)),
        "FID (↓)": float(fid_score_val),
        "Valid Image Pairs": len(valid_filenames),  # 新增：记录有效匹配文件数
        "Total Reference Images": len(ref_files)     # 新增：记录总参考文件数
    }

    # 打印详细汇总信息
    print("\n" + "="*50)
    print("Evaluation Summary")
    print("="*50)
    print(f"Total reference images: {len(ref_files)}")
    print(f"Valid matched images: {len(valid_filenames)}")
    print(f"Missing generated images: {len(ref_files) - len(valid_filenames)}")
    print(f"LPIPS (lower is better): {results['LPIPS (↓)']:.4f}")
    print(f"SSIM (higher is better): {results['SSIM (↑)']:.4f}")
    print(f"PSNR (higher is better): {results['PSNR (↑)']:.2f}")
    print(f"FID (lower is better): {results['FID (↓)']:.2f}")
    print("="*50)

    return results



GPT_SCORE_PROMPT = """
You are an expert evaluator for image generation quality, especially for action-conditional and world-model-based image generation (such images are used to predict the next moment's state of the real world).

You will be given 2 images in total:
- The FIRST image is the REAL reference image (ground truth): it represents the true next moment's world state, including the action outcome, object states, and environmental context at that moment.
- The SECOND image is the GENERATED image (model output): it is the model's attempt to predict the next moment's world state corresponding to the real reference image, with the same timestamp context.

Your task is to evaluate how well the GENERATED image matches the REAL reference image by considering the image as a whole (focus on the content presented in the image itself).

Please focus on the following aspects (in order of importance):
1. Action completion and goal achievement:
   - Does the generated image restore the SAME action outcome as the real reference image?
   - Does it reach the same final goal or outcome shown in the real reference image?
   - If the action outcome is incomplete, incorrect, or deviates from the real reference image, the score should be LOW, even if the visuals look realistic.
2. Object and state consistency:
   - Are the objects, subjects (identities), and their states in the generated image consistent with those in the real reference image?
   - Are there any unreasonable missing objects, morphological contradictions, or state deviations?
3. Physical plausibility and scene coherence:
   - Are the object interactions and state presentations in the generated image physically plausible (in line with common sense)?
   - Is the scene context of the generated image coherent with the next moment's state in the real reference image?
4. Visual realism:
   - Are the texture, lighting, and detail of the generated image visually realistic?
5. Integrity and accuracy of next moment's world state prediction:
   - Judge whether the generated image has perfectly predicted and restored the next moment's world state presented by the real reference image (ground truth), including the coherence of the scene state, the integrity of key object states, and the consistency of environmental context changes between the predicted content and the true next moment state.

Important rules:
- Evaluate the generated image as a whole, not partial details; do not infer content that is not supported by the image itself.
- Do NOT reward generated images that look realistic but have inconsistent action outcomes or goals with the real reference image.
- Prioritize action completion and goal achievement, followed by object consistency, physical plausibility, and finally visual realism and world state prediction integrity.

Give a score from 0 to 10 (higher is better), with specific score ranges corresponding to the following standards:
- 10: The generated image matches the real reference image very well, perfectly restores the action outcome and goal, and has intact prediction of the next moment's world state.
- 7–9: The action outcome and goal are mostly correct with minor deviations; object states and physical logic are coherent, and the next moment's world state is well predicted.
- 4–6: The action outcome is partially correct or incomplete; there are slight contradictions in object states, and the prediction of the next moment's world state has partial omissions.
- 1–3: The action outcome is largely incorrect or inconsistent; object states are chaotic, physical logic is violated, and the next moment's world state cannot be effectively predicted.
- 0: The action outcome is completely wrong or absent; the generated image is irrelevant to the real reference image, and the next moment's world state prediction fails completely.

Output ONLY a JSON object in the following format (do not add any extra content outside the JSON):
{
  "score": float,
  "reason": string
}
"""
import json
import re

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

def parse_json_from_llm(text: str):
    """
    尝试从 LLM 输出中稳健解析 JSON
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 兜底：提取 {...}
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group())
        else:
            raise ValueError(f"Cannot parse JSON from LLM output:\n{text}")

def gpt_score_single(
    gt_image_path: str,
    gen_image_path: str,
    model: str = "gpt-5.1-2025-11-13"
):
    """
    对单对 (gt, gen) 图像进行 GPT 评分
    """
    result = llm(
        prompt=GPT_SCORE_PROMPT,
        model=model,
        image_path=[gt_image_path, gen_image_path],
        temperature=0.0,   # 评分建议 temperature=0
        max_tokens=512
    )

    parsed = parse_json_from_llm(result)

    score = float(parsed["score"])
    print(f"GPT Score: {score}")
    reason = parsed.get("reason", "")

    return {
        "gt": gt_image_path,
        "gen": gen_image_path,
        "score": score,
        "reason": reason
    }
def compute_gpt_score_for_dirs(
    gt_dir: str,
    gen_dir: str,
    suffixes=(".png", ".jpg", ".jpeg"),
):
    """
    对两个目录下的图像进行 GPT 评分
    文件名必须一致
    """
    gt_files = sorted([
        f for f in os.listdir(gt_dir)
        if f.lower().endswith(suffixes)
    ])

    results = []

    for fname in tqdm(gt_files, desc="GPT Scoring"):
        gt_path = os.path.join(gt_dir, fname)
        gen_path = os.path.join(gen_dir, fname)

        if not os.path.exists(gen_path):
            print(f"[WARN] Missing generated image: {fname}")
            continue

        try:
            res = gpt_score_single(gt_path, gen_path)
            results.append(res)
        except Exception as e:
            print(f"[ERROR] {fname}: {e}")

    scores = [r["score"] for r in results]
    return {
        "num_samples": len(scores),
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
    }


# ------------------------------------------------
# Entry
# ------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_dir", type=str, required=True)
    parser.add_argument("--gen_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=[256, 256],
        help="Resize images for LPIPS / SSIM / PSNR, e.g. --resize 256 256"
    )

    parser.add_argument(
        "--fid_resize",
        type=int,
        nargs=2,
        default=[256, 256],
        help="Resize images for FID, e.g. --fid_resize 256 256"
    )

    args = parser.parse_args()

    resize = tuple(args.resize) if args.resize is not None else None
    fid_resize = tuple(args.fid_resize)

    results = evaluate(
        ref_dir=args.ref_dir,
        gen_dir=args.gen_dir,
        device=args.device,
        resize=resize,
        fid_resize=fid_resize,
    )

    print("\n========== Evaluation Results ==========")
    for k, v in results.items():
        print(f"{k}: {v:.6f}")

    # compute_gpt_score_for_dirs(gt_dir=args.ref_dir, gen_dir=args.gen_dir)
    gpt_results = compute_gpt_score_for_dirs(gt_dir=args.ref_dir, gen_dir=args.gen_dir)
    print("\n========== GPT Score Results ==========")
    for k, v in gpt_results.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")