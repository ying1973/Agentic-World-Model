import csv
import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


OPEN_SOURCE_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = None
OUT_ROOT = OPEN_SOURCE_ROOT / "data" / "MultiWorldBench"

SENSITIVE_PATH_PATTERNS = [
    re.compile("/" + "home" + r"/[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*"),
    re.compile("/" + "mnt" + r"/[A-Za-z]/Users/[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*"),
]


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(str(src))
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def write_json(path: Path, data) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows, fieldnames) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sanitize_text(value: str) -> str:
    for pattern in SENSITIVE_PATH_PATTERNS:
        value = pattern.sub("<LOCAL_PROJECT_PATH>", value)
    return value


def sanitize_taco_item(item):
    if isinstance(item, dict):
        cleaned = {}
        for key, value in item.items():
            if key == "solutions" and isinstance(value, str):
                cleaned[key] = sanitize_text(value)
            else:
                cleaned[key] = sanitize_taco_item(value)
        return cleaned
    if isinstance(item, list):
        return [sanitize_taco_item(value) for value in item]
    if isinstance(item, str):
        return sanitize_text(item)
    return item


def rel(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def flatten_mobile_path(path_value: str) -> str:
    parent = Path(path_value).parent.name
    name = Path(path_value).name
    return f"{parent}-{name}" if parent else name


def flatten_android_path(path_value: str) -> str:
    parts = Path(path_value).parts
    parent = parts[0] if parts else ""
    name = Path(path_value).name
    return f"{parent}-{name}" if parent else name


def parse_coin_filename(filename: str):
    stem = Path(filename).stem
    parts = stem.split("-")
    if len(parts) < 3:
        return parts[0], "", ""
    return parts[0], parts[1], "-".join(parts[2:])


def parse_droid_filename(filename: str):
    stem = Path(filename).stem
    parts = stem.split("-")
    if len(parts) < 3:
        return "", ""
    return parts[1], "-".join(parts[2:])


def parse_atari_filename(filename: str):
    match = re.match(r"^(.+)-([^-]+)-prev-([^-]+)-([^-]+)\.png$", filename)
    if not match:
        raise ValueError(f"Unexpected Atari filename: {filename}")
    return match.groups()


def extract_first_frame(video_path: Path, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(out_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {video_path}: {result.stderr[-500:]}")


def prepare_coin() -> int:
    src_root = DATASETS_ROOT / "COIN"
    dst_root = OUT_ROOT / "general" / "coin"
    videos_dst = dst_root / "ground_truth_videos"
    frames_dst = dst_root / "first_frame"

    with (src_root / "bench_task.json").open(encoding="utf-8") as f:
        raw_samples = json.load(f)

    samples = []
    for idx, item in enumerate(raw_samples):
        video_src = Path(item["video"])
        image_src = Path(item["image"])
        video_dst = videos_dst / video_src.name
        image_dst = frames_dst / image_src.name
        copy_file(video_src, video_dst)
        copy_file(image_src, image_dst)
        video_id, task, action = parse_coin_filename(video_src.name)
        samples.append(
            {
                "id": idx,
                "video_id": item.get("video_id", video_id),
                "task": task,
                "action": action,
                "prompt": item.get("prompt", ""),
                "input_image": rel(image_dst, dst_root),
                "ground_truth_video": rel(video_dst, dst_root),
            }
        )

    write_json(dst_root / "samples.json", samples)
    return len(samples)


def prepare_droid() -> int:
    src_dir = DATASETS_ROOT / "droid" / "benchdata"
    dst_root = OUT_ROOT / "embodied" / "droid"
    videos_dst = dst_root / "ground_truth_videos"
    frames_dst = dst_root / "first_frame"

    samples = []
    for idx, video_src in enumerate(sorted(src_dir.glob("*.mp4"))):
        video_dst = videos_dst / video_src.name
        frame_dst = frames_dst / f"{video_src.name}.png"
        copy_file(video_src, video_dst)
        extract_first_frame(video_src, frame_dst)
        video_id, action = parse_droid_filename(video_src.name)
        samples.append(
            {
                "id": idx,
                "video_id": video_id,
                "action": action,
                "input_image": rel(frame_dst, dst_root),
                "ground_truth_video": rel(video_dst, dst_root),
            }
        )

    write_json(dst_root / "samples.json", samples)
    return len(samples)


def prepare_atari() -> int:
    src_root = DATASETS_ROOT / "atari_ALL" / "benchdata"
    dst_root = OUT_ROOT / "game" / "atari"
    prev_dst = dst_root / "prev"
    curr_dst = dst_root / "curr"

    samples = []
    for idx, prev_src in enumerate(sorted((src_root / "prev").glob("*.png"))):
        game, sample_id, action_name, action_id = parse_atari_filename(prev_src.name)
        curr_name = prev_src.name.replace("-prev-", "-curr-")
        curr_src = src_root / "curr" / curr_name
        prev_out = prev_dst / prev_src.name
        curr_out = curr_dst / curr_name
        copy_file(prev_src, prev_out)
        copy_file(curr_src, curr_out)
        samples.append(
            {
                "id": idx,
                "game": game,
                "sample_id": sample_id,
                "action_name": action_name,
                "action_id": action_id,
                "input_image": rel(prev_out, dst_root),
                "ground_truth_image": rel(curr_out, dst_root),
            }
        )

    write_json(dst_root / "samples.json", samples)
    return len(samples)


def prepare_mobileworld() -> int:
    src_root = DATASETS_ROOT / "MobileWorld"
    dst_root = OUT_ROOT / "gui" / "mobileworld"
    input_dst = dst_root / "input_images"
    gt_dst = dst_root / "ground_truth_images"
    rows = read_csv(src_root / "benchmark" / "new_gen.csv")

    out_rows = []
    samples = []
    for idx, row in enumerate(rows):
        input_name = flatten_mobile_path(row["input_image"])
        output_name = flatten_mobile_path(row["output_image"])
        input_src = src_root / "mobileworldbench" / "gen_images" / row["input_image"]
        gt_src = src_root / "benchmark" / "output_image" / output_name
        input_out = input_dst / input_name
        gt_out = gt_dst / output_name
        copy_file(input_src, input_out)
        copy_file(gt_src, gt_out)

        normalized = {
            "id": idx,
            "input_image": rel(input_out, dst_root),
            "ground_truth_image": rel(gt_out, dst_root),
            "action": row.get("action", ""),
            "changes": row.get("changes", ""),
            "split": row.get("split", ""),
            "folder": row.get("folder", ""),
            "source_input_image": row["input_image"],
            "source_output_image": row["output_image"],
        }
        samples.append(normalized)
        out_rows.append(normalized)

    fieldnames = [
        "id",
        "input_image",
        "ground_truth_image",
        "action",
        "changes",
        "split",
        "folder",
        "source_input_image",
        "source_output_image",
    ]
    write_csv(dst_root / "samples.csv", out_rows, fieldnames)
    write_json(dst_root / "samples.json", samples)
    return len(samples)


def prepare_android_control() -> int:
    src_root = DATASETS_ROOT / "android_control"
    dst_root = OUT_ROOT / "gui" / "android_control"
    input_dst = dst_root / "input_images"
    gt_dst = dst_root / "ground_truth_images"
    rows = read_csv(src_root / "benchdata.csv")

    out_rows = []
    samples = []
    for idx, row in enumerate(rows):
        input_name = flatten_android_path(row["input_image"])
        output_name = flatten_android_path(row["output_image"])
        input_src = src_root / "benchdata" / "selected_episodes" / row["input_image"]
        gt_src = src_root / "benchdata" / "output_images" / output_name
        input_out = input_dst / input_name
        gt_out = gt_dst / output_name
        copy_file(input_src, input_out)
        copy_file(gt_src, gt_out)

        normalized = {
            "id": idx,
            "input_image": rel(input_out, dst_root),
            "ground_truth_image": rel(gt_out, dst_root),
            "action": row.get("action", ""),
            "source_input_image": row["input_image"],
            "source_output_image": row["output_image"],
        }
        samples.append(normalized)
        out_rows.append(normalized)

    fieldnames = [
        "id",
        "input_image",
        "ground_truth_image",
        "action",
        "source_input_image",
        "source_output_image",
    ]
    write_csv(dst_root / "samples.csv", out_rows, fieldnames)
    write_json(dst_root / "samples.json", samples)
    return len(samples)


def prepare_webdreamer() -> int:
    src_root = DATASETS_ROOT / "WebDreamer_data" / "benchdata"
    dst_root = OUT_ROOT / "gui" / "webdreamer"
    images_dst = dst_root / "images"

    with (src_root / "samples.json").open(encoding="utf-8") as f:
        rows = json.load(f)

    samples = []
    for idx, row in enumerate(rows):
        image_src = src_root / row["image"]
        image_dst = images_dst / Path(row["image"]).name
        copy_file(image_src, image_dst)
        samples.append(
            {
                "id": row.get("id", f"sample_{idx}"),
                "prompt": row.get("prompt", ""),
                "action": row.get("action", ""),
                "ground_truth_response": row.get("response", ""),
                "input_image": rel(image_dst, dst_root),
            }
        )

    write_json(dst_root / "samples.json", samples)
    return len(samples)


def prepare_taco() -> int:
    src = DATASETS_ROOT / "TACO" / "new_test_200.json"
    dst_root = OUT_ROOT / "code" / "taco"
    dst = dst_root / "new_test_200.json"
    with src.open(encoding="utf-8") as f:
        data = json.load(f)
    data = sanitize_taco_item(data)
    write_json(dst, data)
    return len(data)


def prepare_mbpp() -> int:
    src = DATASETS_ROOT / "MBPP" / "benchdata_100.jsonl"
    dst_root = OUT_ROOT / "code" / "mbpp"
    dst = dst_root / "benchdata_100.jsonl"
    copy_file(src, dst)
    with dst.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def prepare_bfcl() -> int:
    src_root = DATASETS_ROOT / "BFCL"
    dst_root = OUT_ROOT / "tool" / "bfcl"
    question_dst = dst_root / "benchdata.json"
    answer_dst = dst_root / "answer_benchdata.json"
    copy_file(src_root / "benchdata.json", question_dst)
    copy_file(src_root / "answer" / "answer_benchdata.json", answer_dst)

    questions = []
    answers = {}
    with question_dst.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    with answer_dst.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                answers[item["id"]] = item.get("ground_truth", [])

    merged = []
    for item in questions:
        merged.append(
            {
                "id": item["id"],
                "question": item.get("question", []),
                "function": item.get("function", []),
                "ground_truth": answers.get(item["id"], []),
            }
        )
    write_json(dst_root / "samples.json", merged)
    return len(merged)


def main() -> None:
    global DATASETS_ROOT, OUT_ROOT

    parser = argparse.ArgumentParser(description="Prepare the open-source MultiWorldBench evaluation data subset.")
    parser.add_argument(
        "--datasets-root",
        default=os.environ.get("MWB_SOURCE_DATASETS_ROOT", ""),
        help="Root directory of the original local datasets. Can also be set with MWB_SOURCE_DATASETS_ROOT.",
    )
    parser.add_argument(
        "--out-root",
        default=os.environ.get("MWB_EVAL_DATA_OUT", str(OUT_ROOT)),
        help="Output directory for the open-source MultiWorldBench evaluation data.",
    )
    args = parser.parse_args()

    if not args.datasets_root:
        raise RuntimeError("Missing source dataset root. Pass --datasets-root or set MWB_SOURCE_DATASETS_ROOT.")

    DATASETS_ROOT = Path(args.datasets_root).expanduser().resolve()
    OUT_ROOT = Path(args.out_root).expanduser().resolve()

    reset_dir(OUT_ROOT)
    counts = {
        "coin": prepare_coin(),
        "droid": prepare_droid(),
        "atari": prepare_atari(),
        "mobileworld": prepare_mobileworld(),
        "android_control": prepare_android_control(),
        "webdreamer": prepare_webdreamer(),
        "taco": prepare_taco(),
        "mbpp": prepare_mbpp(),
        "bfcl": prepare_bfcl(),
    }
    write_json(OUT_ROOT / "manifest.json", counts)
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
