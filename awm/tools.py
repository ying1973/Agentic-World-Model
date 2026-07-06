from curses import savetty
from typing import Callable, Optional, Union, List, Dict
from dataclasses import dataclass
import requests
import json
import base64
import re
import logging
from io import BytesIO
import subprocess
import os
import time
import tempfile
from datetime import datetime, timedelta
import mimetypes


from awm.media_cache import global_media_cache


logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

_desc_path = os.path.join(os.path.dirname(__file__), "tool_descriptions_new.json")
try:
    with open(_desc_path, "r", encoding="utf-8") as _f:
        TOOL_DESCRIPTIONS = json.load(_f)
except Exception:
    TOOL_DESCRIPTIONS = {}


_desc_path_new_tools = os.path.join(os.path.dirname(__file__), "tool_descriptions_new_tools.json")
try:
    with open(_desc_path_new_tools, "r", encoding="utf-8") as _f:
        TOOL_DESCRIPTIONS_NEW_TOOLS = json.load(_f)
except Exception:
    TOOL_DESCRIPTIONS_NEW_TOOLS = {}


token = os.environ.get("AWM_API_KEY", "")
token_gf = os.environ.get("AWM_GF_API_KEY", token)
API_BASE_URL = os.environ.get("AWM_API_BASE_URL", "").rstrip("/")
TASK_API_BASE_URL = os.environ.get("AWM_TASK_API_BASE_URL", API_BASE_URL).rstrip("/")

CUDA_VISIBLE_DEVICE_general = os.environ.get("AWM_CUDA_VISIBLE_DEVICES", "0")
OUTPUT_ROOT = os.environ.get("AWM_OUTPUT_ROOT", "./output")
PROJECT_ROOT = os.environ.get("AWM_PROJECT_ROOT", os.getcwd())
MODEL_ROOT = os.environ.get("AWM_MODEL_ROOT", "./models")

# Global model and processor (initialized once)
_qwen_model = None
_qwen_processor = None
_model_path = os.environ.get(
    "AWM_QWEN_VL_MODEL_PATH",
    os.path.join(MODEL_ROOT, "Qwen", "Qwen3-VL-4B-Instruct"),
)



def _make_filename(prompt: str, ext: str) -> str:
    """Create a safe filename based on the first 30 chars of prompt + timestamp + ext."""
    if prompt is None:
        prompt = ""
    safe = re.sub(r'[^A-Za-z0-9_.-]', '_', prompt)
    base = safe[:50]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{base}_{timestamp}{ext}"

def _output_dir(*parts: str) -> str:
    return os.path.join(OUTPUT_ROOT, *parts)

def _require_token(value: str, env_name: str = "AWM_API_KEY") -> str:
    if not value:
        raise RuntimeError(f"Missing API token. Please set {env_name} in your environment.")
    return value

def _require_endpoint(value: str, env_name: str) -> str:
    if not value:
        raise RuntimeError(f"Missing API endpoint. Please set {env_name} in your environment.")
    return value

def _api_url(path: str) -> str:
    return _require_endpoint(API_BASE_URL, "AWM_API_BASE_URL") + path

def _task_api_url(path: str) -> str:
    return _require_endpoint(TASK_API_BASE_URL, "AWM_TASK_API_BASE_URL") + path

def encode_base64(image_path: str) -> str:
    """
    Encode an image file (given by path) to a base64 string.
    """
    with open(image_path, "rb") as _f:
        data = _f.read()
    return base64.b64encode(data).decode()

def resolve_image_id(image_id: str) -> str:
    """
    工具调用前使用：将 image_id → 临时 image_path
    """
    if isinstance(image_id, str) and image_id.startswith("media_"):
        return global_media_cache.materialize_to_temp_file(image_id)
    return image_id


@dataclass
class Tool:
    """工具类"""
    name: str
    func: Callable
    description: str


def sora2(prompt: str, image_id: str) -> str:
    """
    sora2 视频生成工具（通过远端 API 调用）。

    说明:
      - 本函数接收 `image_id`（不是直接的文件路径）。
      - 当 `image_id` 以 `media_` 开头时，会通过 `resolve_image_id` 从全局媒体缓存生成临时文件路径；
        否则视为本地文件路径并直接使用。

    Args:
        prompt (str): 视频生成的文本描述。
        image_id (str): 图片的唯一 id 或本地图片路径；函数内部会解析为实际图片路径。

    Returns:
        str: 生成的视频文件的本地路径（成功）或空字符串（失败）。
    """
    logger.critical("Tool 'sora2' called / 工具 'sora2' 被调用")

    image_path = resolve_image_id(image_id)
    url = _api_url("/v1/chat/completions")
    image_base64 = encode_base64(image_path)
    payload = json.dumps({
        "model": "sora_video2",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url","image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ]
            }
        ],
        "size": "1280x704"
    })
    headers = {
        'Authorization': f'Bearer {_require_token(token)}',
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    response_text = response.text
    print(f"sora2 response: {response_text}")
    video_url_match = re.search(r'https?://[^\s,"]+\.mp4', response_text)

    if video_url_match:
        video_url = video_url_match.group(0)
        print(f"找到视频URL: {video_url}")
        try:
            video_response = requests.get(video_url, stream=True)
            if video_response.status_code == 200:
                video_content = video_response.content

                video_filename_wo_ext = _make_filename(prompt, "")
                save_dir = _output_dir("videos", "sora2", video_filename_wo_ext)
                os.makedirs(save_dir, exist_ok=True)
                
                video_filename = _make_filename(prompt, ".mp4")
                video_save_path = os.path.join(save_dir, video_filename)

                with open(video_save_path, 'wb') as f:
                    f.write(video_content)
                print(f"视频文件已保存到: {video_save_path}")

                return video_save_path
        except Exception as e:
            print(f"下载视频时出错: {str(e)}")
    else:
        print("未在响应中找到视频URL")
        return "Error: Sora2 video generation failed"


def wan2_5(prompt, image_id: str) -> str:
    """
    wan2_5 视频生成工具（通过远端 API 调用，包含任务轮询与下载逻辑）。
    """
    logger.critical("Tool 'wan2.5' called / 工具 'wan2.5' 被调用")

    POLL_INTERVAL = 3
    MAX_TIMEOUT = 300

    image_path = resolve_image_id(image_id)

    image_base64 = encode_base64(image_path)

    CREATE_TASK_URL = _task_api_url("/task/bailian/image2video")
    headers = {
        'Authorization': f'Bearer {_require_token(token)}',
        'Content-Type': 'application/json'
    }
    payload = json.dumps({
        "model": "wan2.5-i2v-preview",
        "input": {
            "prompt": prompt,
            "img_url": f"data:image/png;base64,{image_base64}"
        },
        "parameters": {
            "resolution": "480P",
            "prompt_extend": True,
            "duration": 3,
            "audio": True
        }
    })

    response = requests.post(CREATE_TASK_URL, headers=headers, data=payload)
    task_result = response.json()
    # 提取task_id
    task_id = task_result.get("output", {}).get("task_id")
    if not task_id:
        print(f"创建任务失败，返回数据未包含task_id：{task_result}")
        return "Error: wan2_5 video generation failed"
    print(f"任务创建成功，task_id：{task_id}")

    query_task_url = _task_api_url(f"/task/{task_id}")
    headers = {'Authorization': f'Bearer {_require_token(token)}'}
    start_time = datetime.now()  # 记录轮询开始时间，用于判断超时

    while True:
        # 检查是否超时
        if datetime.now() - start_time > timedelta(seconds=MAX_TIMEOUT):
            print(f"任务查询超时，超过{MAX_TIMEOUT}秒，task_id：{task_id}")
            return "Error: wan2_5 video generation failed"

        try:
            response = requests.get(query_task_url, headers=headers, timeout=30)
            task_status_result = response.json()
            task_status = task_status_result.get("output", {}).get("task_status")
            video_url = task_status_result.get("output", {}).get("video_url")

            # print(f"当前任务状态：{task_status}，查询时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            if task_status == "SUCCEEDED":
                if not video_url:
                    print(f"任务成功但未返回视频URL：{task_status_result}")
                    return "Error: wan2_5 video generation failed"
                print(f"任务执行成功，获取视频URL：{video_url}")
                break
            elif task_status == "FAILED":
                print(f"任务执行失败，返回数据：{task_status_result}")
                return "Error: wan2_5 video generation failed"
            elif task_status in ["PENDING", "RUNNING"]:
                time.sleep(POLL_INTERVAL)
            else:
                print(f"未知任务状态：{task_status}，终止查询...")
                return "Error: wan2_5 video generation failed"
        except Exception as e:
            print(f"任务查询异常：{str(e)}")
            return "Error: wan2_5 video generation failed"

    try:
        response = requests.get(video_url, stream=True, timeout=60)
        response.raise_for_status()
        # 获取视频文件大小（可选）
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0

        video_filename_wo_ext = _make_filename(prompt, "")
        save_dir = _output_dir("videos", "wan2_5", video_filename_wo_ext)
        os.makedirs(save_dir, exist_ok=True)
        
        video_filename = _make_filename(prompt, ".mp4")
        video_save_path = os.path.join(save_dir, video_filename)

        with open(video_save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024):  # 每次读取1MB
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    # 打印下载进度（可选）
                    progress = (downloaded_size / total_size) * 100 if total_size > 0 else 0
                    print(f"视频下载中：{downloaded_size/1024/1024:.2f}MB / {total_size/1024/1024:.2f}MB ({progress:.1f}%)", end='\r')

        print(f"\n视频下载完成，已保存至：{video_save_path}")
        return video_save_path

    except Exception as e:
        print(f"视频下载失败：{str(e)}")
        return "Error: wan2_5 video generation failed"


def wan2_2(prompt: str, image_id: str) -> str:
    """
    wan2_2 本地视频生成工具（启动本地脚本生成视频）。

    说明:
      - 接受 `image_id`，并在内部通过 `resolve_image_id` 将其解析为实际图片文件路径。
      - 该函数会调用本地的 `generate.py` 脚本并将输出保存为 mp4 文件。

    Args:
        prompt (str): 视频生成的文本描述。
        image_id (str): 图片的唯一 id 或本地图片路径；函数内部会解析为实际图片路径。

    Returns:
        str: 生成的视频文件的本地路径（成功）或空字符串（失败）。
    """
    logger.critical("Tool 'wan2.2' called / 工具 'wan2.2' 被调用")
    image_path = resolve_image_id(image_id)

    work_dir = os.environ.get("AWM_WAN22_REPO", os.path.join(PROJECT_ROOT, "third_party", "Wan2_2"))

    output_filename_wo_ext = _make_filename(prompt, "")
    output_dir = _output_dir("videos", "wan2_2", output_filename_wo_ext)
    os.makedirs(output_dir, exist_ok=True)

    output_filename = _make_filename(prompt, ".mp4")
    output_file = os.path.join(output_dir, output_filename)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICE_general

    # 构建命令行参数
    command = [
        "python", "generate.py",
        "--task", "ti2v-5B",
        "--size", "1280*704",
        "--ckpt_dir", os.environ.get("AWM_WAN22_CKPT", os.path.join(MODEL_ROOT, "Wan-AI", "Wan2.2-TI2V-5B")),
        "--image", image_path,
        "--prompt", prompt,
        "--offload_model", "True",
        "--convert_model_dtype",
        "--t5_cpu",
        "--frame_num", "121",
        "--save_file", output_file,
    ]
    
    try:
        # 执行命令
        print("开始执行wan2_2视频生成...")
        result = subprocess.run(
            command,
            cwd=work_dir,
            env=env,
            text=True,
            check=True,
            shell=False,
            stdout=None,
            stderr=None
        )
        print(f"wan2_2视频生成成功: {result.stdout}")
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"wan2_2视频生成失败: {e.stderr}")
        return "Error: wan2_2 video generation failed"


def GEbase(prompt, image: str):
    """
    GEbase 视频生成工具（封装第三方/本地推理调用）。

    说明:
      - `image` 参数可以是全局媒体缓存的 `image_id` 或直接的本地图片路径；
      - 函数会根据传入的 `image` 调用底层推理主函数并将结果保存到默认输出目录。

    Args:
        prompt (str): 视频生成说明文本。
        image (str): 图片的唯一 id 或本地图片路径；函数内部应解析为实际图片路径（如果需要）。

    Returns:
        None: 该函数通过副作用将输出写入文件系统（如 `./output/videos/GEbase`）。
    """
    logger.critical("Tool 'GEbase' called / 工具 'GEbase' 被调用")

    image_path = resolve_image_id(image)

    work_dir = os.environ.get("AWM_GEBASE_REPO", os.path.join(PROJECT_ROOT, "third_party", "Genie-Envisioner"))
    config_path = "configs/ltx_model/video_model_infer_slow.yaml"
    ckp_path = os.environ.get("AWM_GEBASE_CKPT", os.path.join(MODEL_ROOT, "agibot_world", "Genie-Envisioner", "ge_base_slow_v0.1.safetensors"))
    save_dir = _output_dir("videos", "GEbase", _make_filename(prompt, ""))
    os.makedirs(save_dir, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICE_general

    command = [
        "torchrun",
        "--nnodes=1",
        "--nproc_per_node=1",
        "--node_rank=0",
        "main.py",
        "--runner_class_path", "runner/ge_inferencer.py",
        "--runner_class", "Inferencer",
        "--config_file", config_path,
        "--mode", "infer_single_image",
        "--checkpoint_path", ckp_path,
        "--output_path", save_dir,
        "--image_path", image_path,
        "--prompt", prompt,
    ]

    try:
        print("开始执行 GEbase 推理 (torchrun)...")
        result = subprocess.run(
            command,
            cwd=work_dir,
            env=env,
            text=True,
            check=True,
            shell=False,
            stdout=None,
            stderr=None
        )
        print(f"GEbase 视频生成成功: {result.stdout}")
        return save_dir
    except subprocess.CalledProcessError as e:
        print(f"GEbase 视频生成失败: {e.stderr}")
        return "Error: GEbase video generation failed"


def diamond(game:str, image_id:str, action:int) -> str:
    """
    DIAMOND 世界模型工具：根据当前环境截图与动作，生成下一时刻的观察图像。

    说明:
      - 接受 `image_id`（或本地路径），函数内部会使用 `resolve_image_id` 将其解析为实际图片路径。

    Args:
        game (str): 游戏名称，例如 `pong`、`breakout` 等 Atari100k 中的游戏名。
        image_id (str): 当前环境图片的唯一 id 或本地路径；函数内部会解析为实际图片路径。
        action (int): 要执行的动作 id。

    Returns:
        str: 生成的下一时刻环境状态图片的本地路径（成功），或错误指示字符串（失败）。
    """
    logger.critical("Tool 'diamond' called / 工具 'diamond' 被调用")
    
    image_path = resolve_image_id(image_id)
    image_path = os.path.abspath(image_path)

    work_dir = PROJECT_ROOT
    checkpoint = os.environ.get("AWM_DIAMOND_CKPT", os.path.join(MODEL_ROOT, "DIAMOND", "atari_100k", "models"))
    output_dir = _output_dir("images", "diamond")
    os.makedirs(output_dir, exist_ok=True)
    out_image = os.path.join(output_dir, f"next_obs_{os.path.basename(image_path).split('.')[0]}_{action}.png")

    command = [
        "python", 
        "./diamond/scripts/run_world_model.py",
        f"--checkpoint={checkpoint}",
        f"--game={game}",
        f"--input-image={image_path}",
        f"--action={action}",
        f"--out-image={out_image}"
    ]
    try:
        # 执行命令
        result = subprocess.run(
            command,
            cwd=work_dir,
            capture_output=True,
            text=True,
            check=True,
            shell=False,
            stdout=None,  # 子进程stdout直接输出到终端
            stderr=None   # 子进程stderr直接输出到终端
        )
        print(f"命令执行成功: {result.stdout}")
        return out_image
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {e.stderr}")
        return "Error: Diamond image generation failed"


def WebDreamer(task: str, image_id: str):
    logger.critical("Tool 'WebDreamer' called / 工具 'WebDreamer' 被调用")
    image_path = resolve_image_id(image_id)

    try:
        from WebDreamer.world_model import WebWorldModel, CustomAPIClient, encode_image
    except ImportError as exc:
        raise RuntimeError(
            "WebDreamer is an optional third-party dependency. "
            "Install it and make it importable before using the WebDreamer tool."
        ) from exc

    custom_client = CustomAPIClient(_require_token(token))
    world_model = WebWorldModel(custom_client)

    screenshot_b64 = encode_image(image_path)
    screenshot = "data:image/jpeg;base64," + screenshot_b64
    action_description = task
    task = "UNKNOWN"
    try:
        imagination = world_model.multiple_step_change_prediction(screenshot, task,
                                                                    action_description,
                                                                    format='change', k=0)
        save_dir = _output_dir("texts", "WebDreamer")
        os.makedirs(save_dir, exist_ok=True)
        # Use task-based filename (task may be more descriptive than a missing prompt)
        code_filename = _make_filename(task, ".json")
        code_save_path = os.path.join(save_dir, code_filename)
        with open(code_save_path, 'w') as f:
            json.dump(imagination, f, indent=4)
        print(f"WebDreamer: 响应已保存至 {code_save_path}")
        return imagination

    except Exception as e:
        print(f"WebDreamer: Text generation failed: {e}")
        return "WebDreamer text generation failed"


def qwen_coder(prompt: str, max_tokens: int = 1688, temperature: float = 0.5):
    """
    qwen_coder：调用远端接口生成代码或文本回复的工具封装。

    说明:
      - 该函数仅发送 `prompt` 到远端 API，并将原始响应保存到 `./output/texts`。

    Args:
        prompt (str): 要发送给模型的提示文本。
        max_tokens (int): 最大返回 token 数量。
        temperature (float): 采样温度，控制输出随机性。

    Returns:
        str: 模型生成的文本内容（字符串）。
    """
    url = _api_url("/v1/chat/completions")
    logger.critical("Tool 'qwen_coder' called / 工具 'qwen_coder' 被调用")

    payload = json.dumps({
        "model": "qwen3-coder-plus",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    })
    headers = {
        'Authorization': f'Bearer {_require_token(token)}',
        'Content-Type': 'application/json'
    }
    try:

        response = requests.request("POST", url, headers=headers, data=payload)

        save_dir = _output_dir("texts", "qwen_coder")
        os.makedirs(save_dir, exist_ok=True)
        code_filename = _make_filename(prompt, ".json")
        code_save_path = os.path.join(save_dir, code_filename)
        with open(code_save_path, 'w') as f:
            json.dump(response.json(), f, indent=4)
        print(f"qwen_coder: 响应已保存至 {code_save_path}")

        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Qwen-coder: Text generation failed: {e}")
        return "Qwen-coder text generation failed"


def gpt_image_1(prompt: str, image_id: str):
    """
    gpt_image_1：基于传入图片进行编辑/变换的远端图像工具封装。

    说明:
      - 接受 `image_id`，函数内部会解析为实际图片路径并将文件内容上传给远端图像编辑接口；
      - 编辑结果以 Base64 返回并保存到 `./output/images`。

    Args:
        prompt (str): 描述要对图片执行的编辑或变换的文本提示。
        image_id (str): 图片的唯一 id 或本地路径；函数内部会解析为实际路径并读取文件。

    Returns:
        str: 返回生成图像的本地路径（成功），或错误指示字符串（失败）。
    """
    logger.critical("Tool 'gpt_image_1' called / 工具 'gpt_image_1' 被调用")
    url = _api_url("/v1/images/edits")

    # read image bytes from path
    image_path = resolve_image_id(image_id)
    with open(image_path, 'rb') as _imgf:
        img_bytes = _imgf.read()
        
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        raise ValueError(f"Cannot determine MIME type for {image_path}")

    payload = {
        "model": "gpt-image-1",
        "prompt": prompt
    }
    files=[('image',('image_file.png', img_bytes, mime_type))]
    headers = {
        'Authorization': f'Bearer {_require_token(token)}'
    }

    response = requests.request("POST", url, headers=headers, data=payload, files=files)

    response_data = response.json()
    if "data" in response_data and len(response_data["data"]) > 0:
        b64_image_data = response_data["data"][0].get("b64_json")
        if b64_image_data:
            # 解码Base64数据
            image_bytes = base64.b64decode(b64_image_data)
            
            save_dir = _output_dir("images", "gpt_image_1")
            os.makedirs(save_dir, exist_ok=True)
            image_filename = _make_filename(prompt, ".png")
            save_path = os.path.join(save_dir, image_filename)
            with open(save_path, "wb") as f:
                f.write(image_bytes)
            
            print(f"gpt_image_1: 图片已保存至 {save_path}")
            return save_path
    return "Error: gpt_image_1 image generation failed"


def nano_banana(prompt: str, image_id: str):
    """
    nano_banana：另一个远端图像编辑/生成接口的封装。

    说明:
      - 接受 `image_id` 并将对应文件上传到远端服务，请求返回 Base64 格式的编辑结果并保存到本地。

    Args:
        prompt (str): 对图片的编辑或生成描述。
        image_id (str): 图片的唯一 id 或本地路径；函数内部会解析为实际图片路径并读取。

    Returns:
        str: 返回生成图像的本地路径（成功），或错误指示字符串（失败）。
    """
    logger.critical("Tool 'nano_banana' called / 工具 'nano_banana' 被调用")
    url = _api_url("/v1/images/edits")

    image_path = resolve_image_id(image_id)
    with open(image_path, 'rb') as _imgf:
        img_bytes = _imgf.read()

    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        raise ValueError(f"Cannot determine MIME type for {image_path}")

    payload = {
        "model": "nano-banana",
        "prompt": prompt,
        "response_format": "b64_json",
    }
    files=[('image',('image_file.png', img_bytes, mime_type))]
    headers = {
        'Authorization': f'Bearer {_require_token(token)}'
    }

    response = requests.request("POST", url, headers=headers, data=payload, files=files)

    response_data = response.json()
    if "data" in response_data and len(response_data["data"]) > 0:
        b64_image_data = response_data["data"][0].get("b64_json")
        if b64_image_data:
            image_bytes = base64.b64decode(b64_image_data)
            
            save_dir = _output_dir("images", "nano_banana")
            os.makedirs(save_dir, exist_ok=True)
            image_filename = _make_filename(prompt, ".png")
            save_path = os.path.join(save_dir, image_filename)
            with open(save_path, "wb") as f:
                f.write(image_bytes)
            print(f"nano_banana: 图片已保存至 {save_path}")

            return save_path
    return "Error: nano_banana image generation failed"


def qwen_image_edit(prompt: str, image_id: str):
    """
    qwen_image_edit：另一个远端图像编辑/生成接口的封装。

    说明:
      - 接受 `image_id` 并将对应文件上传到远端服务，请求返回 Base64 格式的编辑结果并保存到本地。

    Args:
        prompt (str): 对图片的编辑或生成描述。
        image_id (str): 图片的唯一 id 或本地路径；函数内部会解析为实际图片路径并读取。

    Returns:
        str: 返回生成图像的本地路径（成功），或错误指示字符串（失败）。
    """
    logger.critical("Tool 'qwen_image_edit' called / 工具 'qwen_image_edit' 被调用")
    url = _api_url("/v1/images/edits")
    image_path = resolve_image_id(image_id)

    with open(image_path, 'rb') as _imgf:
        img_bytes = _imgf.read()
    
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        raise ValueError(f"Cannot determine MIME type for {image_path}")

    payload = {
        "model": "qwen-image-edit",
        "prompt": prompt,
        "response_format": "b64_json"
    }
    files=[('image',('image_file.png', img_bytes, mime_type))]
    headers = {
        'Authorization': f'Bearer {_require_token(token)}'
    }

    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    response_data = response.json()
    if "data" in response_data and len(response_data["data"]) > 0:
        b64_image_data = response_data["data"][0].get("b64_json")
        if b64_image_data:
            image_bytes = base64.b64decode(b64_image_data)
            
            save_dir = _output_dir("images", "qwen_image_edit")
            os.makedirs(save_dir, exist_ok=True)
            image_filename = _make_filename(prompt, ".png")
            save_path = os.path.join(save_dir, image_filename)
            with open(save_path, "wb") as f:
                f.write(image_bytes)
            print(f"qwen_image_edit: 图片已保存至 {save_path}")

            return save_path
    return "Error: qwen_image_edit image generation failed"


def chatgpt5(prompt: str, image_id: Optional[str] = None, max_tokens: int = 1688, temperature: float = 0.5):
    """
    chatgpt5：多模态聊天工具封装，支持附带图片（通过 `image_id`）。

    说明:
      - `image_id` 可选；若提供，函数内部会通过 `resolve_image_id` 获取图片路径并编码为 Base64，随后以 `image_url` 的形式
        附加到发送给远端模型的消息中。
      - 响应会被保存到 `./output/texts`，同时返回模型生成的文本内容。

    Args:
        prompt (str): 要发送给模型的文本提示。
        image_id (Optional[str]): 可选的图片唯一 id 或本地路径；函数内部会解析并编码为 Base64 后发送给模型。
        max_tokens (int): 最大返回 token 数量。
        temperature (float): 采样温度。

    Returns:
        str | None: 模型返回的文本内容，或在出错时返回 None。
    """
    logger.critical("Tool 'chatgpt5.1' called / 工具 'chatgpt5.1' 被调用")
    url = _api_url("/v1/chat/completions")

    content = [{"type": "text","text": prompt}]
    if image_id is not None:
        image_path = resolve_image_id(image_id)
        image_base64 = encode_base64(image_path)
        content.append({"type": "image_url","image_url": {"url": f"data:image/png;base64,{image_base64}"}})
    
    payload = json.dumps({
        "model": "gpt-5.1",
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
        'Authorization': f'Bearer {_require_token(token_gf, "AWM_GF_API_KEY")}',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    try:
        result = response.json()["choices"][0]["message"]["content"]
        
        save_dir = _output_dir("texts", "chatgpt5")
        os.makedirs(save_dir, exist_ok=True)
        code_filename = _make_filename(prompt, ".json")
        code_save_path = os.path.join(save_dir, code_filename)
        with open(code_save_path, 'w') as f:
            json.dump(response.json(), f, indent=4)
        print(f"chatgpt5: 响应已保存至 {code_save_path}")
        
        return result
    except KeyError as e:
        print(f"API 响应格式异常：{e}，原始响应：{response.text}")
        return "Error: chatpgpt5 text generation failed"


def claude4(prompt: str, image_id: Optional[str] = None, max_tokens: int = 1688, temperature: float = 0.5):
    """
    claude4：调用指定远端模型（Claude 系列）的聊天/多模态接口封装。

    说明:
      - 支持可选的 `image_id`；若提供会被解析并以 Base64 的 `image_url` 形式发送给模型。
      - 响应会存储到 `./output/texts`，同时返回模型生成的文本内容。

    Args:
        prompt (str): 要发送给模型的提示文本。
        image_id (Optional[str]): 可选图片的唯一 id 或本地路径；函数内部会解析并编码后发送给模型。
        max_tokens (int): 最大返回 token 数量。
        temperature (float): 采样温度。

    Returns:
        str | None: 模型返回的文本内容，或在出错时返回 None。
    """
    logger.critical("Tool 'claude4' called / 工具 'claude4' 被调用")
    url = _api_url("/v1/chat/completions")
    
    content = [{"type": "text","text": prompt}]
    if image_id is not None:
        image_path = resolve_image_id(image_id)
        image_base64 = encode_base64(image_path)
        content.append({"type": "image_url","image_url": {"url": f"data:image/png;base64,{image_base64}"}})
    
    payload = json.dumps({
        "model": "claude-sonnet-4-20250514",
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
        'Authorization': f'Bearer {_require_token(token)}',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    try:
        result = response.json()["choices"][0]["message"]["content"]
        
        save_dir = _output_dir("texts", "claude4")
        os.makedirs(save_dir, exist_ok=True)
        code_filename = _make_filename(prompt, ".json")
        code_save_path = os.path.join(save_dir, code_filename)
        with open(code_save_path, 'w') as f:
            json.dump(response.json(), f, indent=4)
        print(f"claude4: 响应已保存至 {code_save_path}")
        
        return result
    except KeyError as e:
        print(f"API 响应格式异常：{e}，原始响应：{response.text}")
        return "Error: claude4 text generation failed"


def chatgpt5_mini(prompt: str, image_id: Optional[str] = None, max_tokens: int = 1688, temperature: float = 0.5):
    """
    chatgpt5_mini：多模态聊天工具封装，支持附带图片（通过 `image_id`）。

    说明:
      - `image_id` 可选；若提供，函数内部会通过 `resolve_image_id` 获取图片路径并编码为 Base64，随后以 `image_url` 的形式
        附加到发送给远端模型的消息中。
      - 响应会被保存到 `./output/texts`，同时返回模型生成的文本内容。

    Args:
        prompt (str): 要发送给模型的文本提示。
        image_id (Optional[str]): 可选的图片唯一 id 或本地路径；函数内部会解析并编码为 Base64 后发送给模型。
        max_tokens (int): 最大返回 token 数量。
        temperature (float): 采样温度。

    Returns:
        str | None: 模型返回的文本内容，或在出错时返回 None。
    """
    logger.critical("Tool 'chatgpt5-mini' called / 工具 'chatgpt5-mini' 被调用")
    url = _api_url("/v1/chat/completions")

    content = [{"type": "text","text": prompt}]
    if image_id is not None:
        image_path = resolve_image_id(image_id)
        image_base64 = encode_base64(image_path)
        content.append({"type": "image_url","image_url": {"url": f"data:image/png;base64,{image_base64}"}})
    
    payload = json.dumps({
        "model": "gpt-5-mini",
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
        'Authorization': f'Bearer {_require_token(token_gf, "AWM_GF_API_KEY")}',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    try:
        result = response.json()["choices"][0]["message"]["content"]
        
        save_dir = _output_dir("texts", "chatgpt5_mini")
        os.makedirs(save_dir, exist_ok=True)
        code_filename = _make_filename(prompt, ".json")
        code_save_path = os.path.join(save_dir, code_filename)
        with open(code_save_path, 'w') as f:
            json.dump(response.json(), f, indent=4)
        print(f"chatgpt5_mini: 响应已保存至 {code_save_path}")
        
        return result
    except KeyError as e:
        print(f"API 响应格式异常：{e}，原始响应：{response.text}")
        return "Error: chatgpt5_mini text generation failed"


def gemini3_flash(prompt: str, image_id: Optional[str] = None, max_tokens: int = 1688, temperature: float = 0.5):
    """
    gemini3_flash：多模态聊天工具封装，支持附带图片（通过 `image_id`）。

    说明:
      - `image_id` 可选；若提供，函数内部会通过 `resolve_image_id` 获取图片路径并编码为 Base64，随后以 `image_url` 的形式
        附加到发送给远端模型的消息中。
      - 响应会被保存到 `./output/texts`，同时返回模型生成的文本内容。

    Args:
        prompt (str): 要发送给模型的文本提示。
        image_id (Optional[str]): 可选的图片唯一 id 或本地路径；函数内部会解析并编码为 Base64 后发送给模型。
        max_tokens (int): 最大返回 token 数量。
        temperature (float): 采样温度。

    Returns:
        str | None: 模型返回的文本内容，或在出错时返回 None。
    """
    logger.critical("Tool 'gemini3_flash' called / 工具 'gemini3_flash' 被调用")
    url = _api_url("/v1/chat/completions")

    content = [{"type": "text","text": prompt}]
    if image_id is not None:
        image_path = resolve_image_id(image_id)
        image_base64 = encode_base64(image_path)
        content.append({"type": "image_url","image_url": {"url": f"data:image/png;base64,{image_base64}"}})
    
    payload = json.dumps({
        "model": "gemini-3-flash-preview",
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
        'Authorization': f'Bearer {_require_token(token)}',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    try:
        result = response.json()["choices"][0]["message"]["content"]
        
        save_dir = _output_dir("texts", "gemini3_flash")
        os.makedirs(save_dir, exist_ok=True)
        code_filename = _make_filename(prompt, ".json")
        code_save_path = os.path.join(save_dir, code_filename)
        with open(code_save_path, 'w') as f:
            json.dump(response.json(), f, indent=4)
        print(f"gemini3_flash: 响应已保存至 {code_save_path}")
        
        return result
    except KeyError as e:
        print(f"API 响应格式异常：{e}，原始响应：{response.text}")
        return "Error: gemini3_flash text generation failed"


def get_tools():
    """获取所有工具"""
    # helper to stringify the structured description (keeps dataclass 'description' as string)
    def _desc(name: str) -> str:
        try:
            return json.dumps(TOOL_DESCRIPTIONS.get(name, {}), ensure_ascii=False)
        except Exception:
            return str(TOOL_DESCRIPTIONS.get(name, {}))

    return [
        # Tool(name="sora2", func=sora2, description=_desc("sora2")),
        Tool(name="wan2.5", func=wan2_5, description=_desc("wan2.5")),
        # Tool(name="wan2.2", func=wan2_2, description=_desc("wan2.2")),
        # Tool(name="GEbase", func=GEbase, description=_desc("GEbase")),
        # Tool(name="diamond", func=diamond, description=_desc("diamond")),
        Tool(name="WebDreamer", func=WebDreamer, description=_desc("WebDreamer")),
        Tool(name="qwen_coder", func=qwen_coder, description=_desc("qwen_coder")),
        Tool(name="gpt_image_1", func=gpt_image_1, description=_desc("gpt_image_1")),
        Tool(name="nano_banana", func=nano_banana, description=_desc("nano_banana")),
        Tool(name="qwen_image_edit", func=qwen_image_edit, description=_desc("qwen_image_edit")),
        Tool(name="chatgpt5", func=chatgpt5, description=_desc("chatgpt5")),
        Tool(name="claude4", func=claude4, description=_desc("claude4")),
        Tool(name="chatgpt5_mini", func=chatgpt5_mini, description=_desc("chatgpt5_mini")),
        Tool(name="gemini3_flash", func=gemini3_flash, description=_desc("gemini3_flash")),
    ]


def synthesis_chatgpt(
    prompt: str,
    images: Optional[List[dict]] = None,
    max_tokens: int = 1688,
    temperature: float = 0.5
):
    url = _api_url("/v1/chat/completions")

    content = [{"type": "text", "text": prompt}]
    if images:
        for item in images:
            if item["type"] == "image":
                content.append({"type": "text", "text": f"TOOL OUTPUT: [IMAGE_ID]: {item['id']}"} )
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{item['data']}"}
                })

            elif item["type"] == "video_frame":
                content.append({"type": "text", "text":
                    f"TOOL OUTPUT: [VIDEO_ID]: {item['video_id']}\n"
                    f"This is a PREVIEW FRAME of video '{item['video_id']}'.\n"
                    "Use ONLY video_id to refer to this video."
                })
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{item['data']}"}
                })

            elif item["type"] == "user_input_image":
                content.append({"type": "text", "text": f"USER INPUT: [IMAGE_ID]: {item['id']}"} )
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{item['data']}"}
                })

    payload = json.dumps({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature
    })

    headers = {
        'Authorization': f'Bearer {_require_token(token)}',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json()["choices"][0]["message"]["content"]


def synthesis_qwen3vl(
    prompt: str,
    images: Optional[List[dict]] = None,
    max_tokens: int = 1688,
    temperature: float = 0.5,
    api_base: str = None,
    model_name: str = "qwen3vl_lora_sft_183",
):
    api_base = api_base or os.environ.get("AWM_LOCAL_VLM_API_BASE", "").rstrip("/")
    url = _require_endpoint(api_base, "AWM_LOCAL_VLM_API_BASE") + "/chat/completions"

    content = [{"type": "text", "text": prompt}]
    if images:
        for item in images:
            if item["type"] == "image":
                content.append({"type": "text", "text": f"TOOL OUTPUT: [IMAGE_ID]: {item['id']}"} )
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{item['data']}"}
                })

            elif item["type"] == "video_frame":
                content.append({"type": "text", "text":
                    f"TOOL OUTPUT: [VIDEO_ID]: {item['video_id']}\n"
                    f"This is a PREVIEW FRAME of video '{item['video_id']}'.\n"
                    "Use ONLY video_id to refer to this video."
                })
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{item['data']}"}
                })

            elif item["type"] == "user_input_image":
                content.append({"type": "text", "text": f"USER INPUT: [IMAGE_ID]: {item['id']}"} )
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{item['data']}"}
                })

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    headers = {"Content-Type": "application/json"}
    local_api_key = os.environ.get("AWM_LOCAL_VLM_API_KEY")
    if local_api_key:
        headers["Authorization"] = f"Bearer {local_api_key}"

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


def jimeng(prompt, image_id: str) ->str:
    url = _task_api_url("/task/jimeng/image2video")
    
    POLL_INTERVAL = 3
    MAX_TIMEOUT = 300
    image_path = resolve_image_id(image_id)
    image_base64 = encode_base64(image_path)

    payload = json.dumps({
        "binary_data_base64": [image_base64],
        "prompt": prompt,
    })

    headers = {
        'Authorization': f'Bearer {_require_token(token)}',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    # print("这是视频生成api返回值",response.text)
    task_result = response.json()
    
    if task_result.get("code") != 10000:
        print(f"创建任务失败，返回码异常：{task_result.get('code')}，消息：{task_result.get('message')}")
        return "Error: jimeng video generation failed"
    
    task_id = task_result.get("data", {}).get("task_id")
    if not task_id:
        print(f"创建任务成功但未返回task_id：{task_result}")
        return "Error: jimeng video generation failed"
    print(f"任务创建成功，task_id：{task_id}")

    query_task_url = _task_api_url(f"/task/{task_id}")
    headers = {'Authorization': f'Bearer {_require_token(token)}'}
    start_time = datetime.now()

    while True:
        if datetime.now() - start_time > timedelta(seconds=MAX_TIMEOUT):
            print(f"任务查询超时，超过{MAX_TIMEOUT}秒，task_id：{task_id}")
            return "Error: jimeng video generation failed"

        try:
            response = requests.get(query_task_url, headers=headers, timeout=30)
            # print("这是视频查询返回值", response.text)
            task_status_result = response.json()

            if task_status_result.get("code") != 10000:
                print(f"任务查询失败，返回码异常：{task_status_result.get('code')}，消息：{task_status_result.get('message')}")
                time.sleep(POLL_INTERVAL)
                continue

            task_status = task_status_result.get("data", {}).get("status")
            video_url = task_status_result.get("data", {}).get("video_url")

            if task_status == "done":
                if not video_url:
                    print(f"任务成功但未返回视频URL：{task_status_result}")
                    return "Error: jimeng video generation failed"
                print(f"任务执行成功，获取视频URL：{video_url}")
                break
            elif task_status == "failed":
                print(f"任务执行失败，返回数据：{task_status_result}")
                return "Error: jimeng video generation failed"
            elif task_status in ["running", "pending", "in_queue"]: 
                time.sleep(POLL_INTERVAL)
            else:
                # print(f"未知任务状态：{task_status}, 继续查询...")
                time.sleep(POLL_INTERVAL)
        except Exception as e:
            print(f"任务查询异常：{str(e)}")
            return "Error: jimeng video generation failed"

    try:
        response = requests.get(video_url, stream=True, timeout=60)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0

        video_filename_wo_ext = _make_filename(prompt, "")
        save_dir = _output_dir("videos", "jimeng", video_filename_wo_ext)
        os.makedirs(save_dir, exist_ok=True)
        
        video_filename = _make_filename(prompt, ".mp4")
        video_save_path = os.path.join(save_dir, video_filename)

        with open(video_save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024):  # 每次读取1MB
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    # 打印下载进度（可选）
                    progress = (downloaded_size / total_size) * 100 if total_size > 0 else 0
                    print(f"视频下载中：{downloaded_size/1024/1024:.2f}MB / {total_size/1024/1024:.2f}MB ({progress:.1f}%)", end='\r')

        print(f"\n视频下载完成，已保存至：{video_save_path}")
        return video_save_path

    except Exception as e:
        print(f"视频下载失败：{str(e)}")
        return "Error: jimeng video generation failed"


def seedance(prompt, image_id: str) -> str:
    logger.critical("Tool 'seedance' called / 工具 'seedance' 被调用")

    POLL_INTERVAL = 3  # 每3秒查询一次，避免频繁请求给接口压力
    MAX_TIMEOUT = 300  # 最大超时5分钟，可根据视频生成时长调整
    image_path = resolve_image_id(image_id)

    full_prompt = f"{prompt} --rs 480p --dur 5"

    image_base64 = encode_base64(image_path)

    CREATE_TASK_URL = _task_api_url("/task/volces/seedance")
    payload = json.dumps({
        "model": "doubao-seedance-1-5-pro-251215",
        "content": [
            {
                "type": "text", 
                "text": full_prompt
            },
            {
                "type": "image_url", 
                "image_url": {"url": f"data:image/png;base64,{image_base64}"}
            }
        ],
        "generate_audio": False
    })
    headers = {
        'Authorization': f'Bearer {_require_token(token)}',
        'Content-Type': 'application/json'
    }

    response = requests.post(CREATE_TASK_URL, headers=headers, data=payload)
    task_result = response.json()
    task_id = task_result.get("id")
    if not task_id:
        print(f"创建任务失败，返回数据未包含id：{task_result}")
        return "Error: seedance video generation failed"
    print(f"任务创建成功，id：{task_id}")

    query_task_url = _task_api_url(f"/task/{task_id}")
    headers = {'Authorization': f'Bearer {_require_token(token)}'}
    start_time = datetime.now()  # 记录轮询开始时间，用于判断超时

    while True:
        # 检查是否超时
        if datetime.now() - start_time > timedelta(seconds=MAX_TIMEOUT):
            print(f"任务查询超时，超过{MAX_TIMEOUT}秒，task_id：{task_id}")
            return "Error: seedance video generation failed"

        try:
            response = requests.get(query_task_url, headers=headers, timeout=30)
            # print("这是视频查询结果", response.text)
            task_status_result = response.json()
            
            task_status = task_status_result.get("status")
            video_url = task_status_result.get("content", {}).get("video_url")
            if task_status == "succeeded":
                if not video_url:
                    print(f"任务成功但未返回视频URL：{task_status_result}")
                    return "Error: seedance video generation failed"
                print(f"任务执行成功，获取视频URL：{video_url}")
                break
            elif task_status == "failed":
                print(f"任务执行失败，返回数据：{task_status_result}")
                return "Error: seedance video generation failed"
            elif task_status in ["pending", "running"]:
                time.sleep(POLL_INTERVAL)
            else:
                # print(f"未知任务状态：{task_status}，继续查询...")
                time.sleep(POLL_INTERVAL)
        except Exception as e:
            print(f"任务查询异常：{str(e)}")
            return "Error: seedance video generation failed"

    try:
        response = requests.get(video_url, stream=True, timeout=60)
        response.raise_for_status()
        # 获取视频文件大小（可选）
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0

        video_filename_wo_ext = _make_filename(prompt, "")
        save_dir = _output_dir("videos", "seedance", video_filename_wo_ext)
        os.makedirs(save_dir, exist_ok=True)
        
        video_filename = _make_filename(prompt, ".mp4")
        video_save_path = os.path.join(save_dir, video_filename)

        with open(video_save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024):  # 每次读取1MB
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    # 打印下载进度（可选）
                    progress = (downloaded_size / total_size) * 100 if total_size > 0 else 0
                    print(f"视频下载中：{downloaded_size/1024/1024:.2f}MB / {total_size/1024/1024:.2f}MB ({progress:.1f}%)", end='\r')

        print(f"\n视频下载完成，已保存至：{video_save_path}")
        return video_save_path

    except Exception as e:
        print(f"视频下载失败：{str(e)}")
        return "Error: seedance video generation failed"


def seedream(prompt: str, image_id: str):
    logger.critical("Tool 'seedream' called / 工具 'seedream' 被调用")
    url = _api_url("/v1/images/edits")

    image_path = resolve_image_id(image_id)

    with open(image_path, 'rb') as _imgf:
        img_bytes = _imgf.read()

    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        raise ValueError(f"Cannot determine MIME type for {image_path}")

    payload = {
        "model": "doubao-seedream-4-0-250828",
        "prompt": prompt,
        "response_format": "b64_json",
    }
    files=[('image',('image_file.png', img_bytes, mime_type))]
    headers = {
        'Authorization': f'Bearer {_require_token(token)}'
    }

    response = requests.request("POST", url, headers=headers, data=payload, files=files)

    response_data = response.json()
    if "data" in response_data and len(response_data["data"]) > 0:
        b64_image_data = response_data["data"][0].get("b64_json")
        if b64_image_data:
            image_bytes = base64.b64decode(b64_image_data)
            
            save_dir = _output_dir("images", "seedream")
            os.makedirs(save_dir, exist_ok=True)
            image_filename = _make_filename(prompt, ".png")
            save_path = os.path.join(save_dir, image_filename)
            with open(save_path, "wb") as f:
                f.write(image_bytes)
            print(f"seedream: 图片已保存至 {save_path}")

            return save_path
    return "Error: seedream image generation failed"


def flux(prompt: str, image_id: str):
    logger.critical("Tool 'flux' called / 工具 'flux' 被调用")
    url = _api_url("/v1/images/edits")

    image_path = resolve_image_id(image_id)

    with open(image_path, 'rb') as _imgf:
        img_bytes = _imgf.read()

    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        raise ValueError(f"Cannot determine MIME type for {image_path}")

    payload = {
        "model": "flux-kontext-pro",
        "prompt": prompt,
        "response_format": "b64_json",
    }
    files=[('image',('image_file.png', img_bytes, mime_type))]
    headers = {
        'Authorization': f'Bearer {_require_token(token)}'
    }

    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    # print(response.text)
    response_data = response.json()
    if "data" in response_data and len(response_data["data"]) > 0:
        b64_image_data = response_data["data"][0].get("b64_json")
        if b64_image_data:
            image_bytes = base64.b64decode(b64_image_data)
            
            save_dir = _output_dir("images", "flux")
            os.makedirs(save_dir, exist_ok=True)
            image_filename = _make_filename(prompt, ".png")
            save_path = os.path.join(save_dir, image_filename)
            with open(save_path, "wb") as f:
                f.write(image_bytes)
            print(f"flux: 图片已保存至 {save_path}")

            return save_path
    return "Error: flux image generation failed"


def gemini2_5(prompt: str, image_id: Optional[str] = None, max_tokens: int = 1688, temperature: float = 0.5):
    logger.critical("Tool 'gemini2_5' called / 工具 'gemini2_5' 被调用")
    url = _api_url("/v1/chat/completions")

    content = [{"type": "text","text": prompt}]
    if image_id is not None:
        image_path = resolve_image_id(image_id)
        image_base64 = encode_base64(image_path)
        content.append({"type": "image_url","image_url": {"url": f"data:image/png;base64,{image_base64}"}})
    
    payload = json.dumps({
        "model": "gemini-2.5-pro",
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
        'Authorization': f'Bearer {_require_token(token)}',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    try:
        result = response.json()["choices"][0]["message"]["content"]
        
        save_dir = _output_dir("texts", "gemini2_5")
        os.makedirs(save_dir, exist_ok=True)
        code_filename = _make_filename(prompt, ".json")
        code_save_path = os.path.join(save_dir, code_filename)
        with open(code_save_path, 'w') as f:
            json.dump(response.json(), f, indent=4)
        print(f"seed: 响应已保存至 {code_save_path}")
        
        return result
    except KeyError as e:
        print(f"API 响应格式异常：{e}，原始响应：{response.text}")
        return "Error: seed text generation failed"


def qwen_chat(prompt: str, image_id: Optional[str] = None, max_tokens: int = 1688, temperature: float = 0.5):
    logger.critical("Tool 'qwen-chat' called / 工具 'qwen-chat' 被调用")
    url = _api_url("/v1/chat/completions")

    content = [{"type": "text","text": prompt}]
    if image_id is not None:
        image_path = resolve_image_id(image_id)
        image_base64 = encode_base64(image_path)
        content.append({"type": "image_url","image_url": {"url": f"data:image/png;base64,{image_base64}"}})
    
    payload = json.dumps({
        "model": "qwen3-vl-235b-a22b-instruct",
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
        'Authorization': f'Bearer {_require_token(token)}',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    try:
        result = response.json()["choices"][0]["message"]["content"]
        
        save_dir = _output_dir("texts", "qwen-chat")
        os.makedirs(save_dir, exist_ok=True)
        code_filename = _make_filename(prompt, ".json")
        code_save_path = os.path.join(save_dir, code_filename)
        with open(code_save_path, 'w') as f:
            json.dump(response.json(), f, indent=4)
        print(f"qwen-chat: 响应已保存至 {code_save_path}")
        
        return result
    except KeyError as e:
        print(f"API 响应格式异常：{e}，原始响应：{response.text}")
        return "Error: qwen-chat text generation failed"

def get_tools_new():
    """获取所有工具"""
    # helper to stringify the structured description (keeps dataclass 'description' as string)
    def _desc(name: str) -> str:
        try:
            return json.dumps(TOOL_DESCRIPTIONS_NEW_TOOLS.get(name, {}), ensure_ascii=False)
        except Exception:
            return str(TOOL_DESCRIPTIONS_NEW_TOOLS.get(name, {}))

    return [
        Tool(name="jimeng", func=jimeng, description=_desc("jimeng")),
        Tool(name="seedance", func=seedance, description=_desc("seedance")),
        Tool(name="seedream", func=seedream, description=_desc("seedream")),
        Tool(name="flux", func=flux, description=_desc("flux")),
        Tool(name="gemini2_5", func=gemini2_5, description=_desc("gemini2_5")),
        Tool(name="qwen_chat", func=qwen_chat, description=_desc("qwen_chat")),
    ]
