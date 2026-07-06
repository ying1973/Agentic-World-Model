# MultiWorldBench Evaluation Data

This directory contains only the MultiWorldBench evaluation split used by the
open-source AWM scripts and evaluators. Training data, generated predictions,
agent trajectories, model outputs, and third-party repositories are excluded.

## Layout

```text
data/MultiWorldBench/
  general/coin/
    samples.json
    first_frame/
    ground_truth_videos/
  embodied/droid/
    samples.json
    first_frame/
    ground_truth_videos/
  game/atari/
    samples.json
    prev/
    curr/
  gui/mobileworld/
    samples.json
    samples.csv
    input_images/
    ground_truth_images/
  code/taco/
    new_test_200.json
  tool/bfcl/
    benchdata.json
    answer_benchdata.json
    samples.json
  manifest.json
```

## Subsets

- COIN (`general/coin`): 143 video-prediction samples. `first_frame/` contains
  the input state; `ground_truth_videos/` contains the reference future videos.
- DROID (`embodied/droid`): 150 robotics video-prediction samples. First frames
  were extracted from the benchmark videos for reproducible AWM inference.
- Atari (`game/atari`): 240 game-world image-prediction samples. `prev/` is the
  input frame; `curr/` is the reference next frame.
- MobileWorld (`gui/mobileworld`): 150 mobile GUI image-prediction samples from
  `benchmark/new_gen.csv`. `input_images/` contains current screenshots and
  `ground_truth_images/` contains next screenshots.
- TACO (`code/taco`): 200 code-generation evaluation problems with test cases.
- BFCL (`tool/bfcl`): 200 function-calling evaluation samples with questions,
  available functions, and ground-truth calls.

## Evaluator Inputs

Common evaluator arguments:

```bash
export PYTHONPATH=/path/to/opensource:$PYTHONPATH
```

Video metrics:

```bash
python -m multiworldbench.eval.video_eval \
  --real_dir data/MultiWorldBench/general/coin/ground_truth_videos \
  --gen_dir /path/to/generated/videos
```

Image metrics:

```bash
python -m multiworldbench.eval.image_eval \
  --ref_dir data/MultiWorldBench/gui/mobileworld/ground_truth_images \
  --gen_dir /path/to/generated/images
```

Atari image metrics:

```bash
python -m multiworldbench.eval.atari_eval \
  --ref_dir data/MultiWorldBench/game/atari/curr \
  --gen_dir /path/to/generated/atari/images
```

Code and tool evaluators:

```bash
python -m multiworldbench.eval.taco_code_eval \
  --dataset_path data/MultiWorldBench/code/taco/new_test_200.json \
  --generated_path /path/to/generated_solutions.json

python -m multiworldbench.eval.tool_eval \
  --result_path /path/to/bfcl_predictions.json \
  --answer_path data/MultiWorldBench/tool/bfcl/answer_benchdata.json
```
