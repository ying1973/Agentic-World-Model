# Agentic World Model & MultiWorldBench

Official implementation of **Agentic World Model (AWM)** and **MultiWorldBench**.

Agentic World Model is an agent-based framework that dynamically routes and composes specialized world models to solve heterogeneous world modeling tasks. MultiWorldBench is a unified benchmark for evaluating world modeling across diverse domains, including video prediction, embodied robotics, games, GUI interaction, code generation, and tool use.

---

## Highlights

- 🌍 **MultiWorldBench**, a benchmark covering six representative world modeling domains.
- 🤖 **Agentic World Model (AWM)**, an agent framework that dynamically selects and composes world-model tools.
- 🧩 Unified evaluation pipeline for image, video, code, and tool-use tasks.
- 📊 Public benchmark data and evaluation scripts for all released subsets.

---

## Repository Structure

```text
.
├── awm/                  # Agentic World Model
├── data/                 # MultiWorldBench datasets
├── multiworldbench/      # Evaluation toolkit
├── scripts/              # Inference scripts
├── requirements.txt
└── README.md
```

The repository is organized into four main components:

- **awm/** implements the routing agent and world-model tool interfaces.
- **data/** contains the released MultiWorldBench benchmark.
- **multiworldbench/** provides evaluation scripts for different task types.
- **scripts/** includes example inference scripts for each benchmark subset.

---

## MultiWorldBench

The released benchmark currently contains six representative subsets.

| Subset | Domain | Samples |
| --- | --- | ---: |
| COIN | General-world video prediction | 143 |
| DROID | Embodied robotics | 150 |
| Atari | Game world | 240 |
| MobileWorld | GUI world | 150 |
| TACO | Code generation | 200 |
| BFCL | Tool use | 200 |

All benchmark data is available under

```text
data/MultiWorldBench/
```

---

## Installation

Clone the repository and install the dependencies.

```bash
conda create -n awm python=3.12
conda activate awm

pip install -r requirements.txt
```

If you use a custom CUDA version, install the appropriate PyTorch package before installing the remaining dependencies.

Then set the project root:

```bash
export PYTHONPATH=/path/to/opensource:$PYTHONPATH
```

---

## Configuration

Before running AWM, configure your API credentials.

```bash
export AWM_API_KEY=your_api_key
export AWM_API_BASE_URL=https://your-api-endpoint
```

If you plan to use LLM-based semantic evaluation, also configure

```bash
export MWB_EVAL_API_KEY=...
export MWB_EVAL_API_URL=...
```

---

## Running AWM

Example: run AWM on the TACO benchmark.

```bash
python scripts/awm_infer/infer_taco.py \
    --json_path data/MultiWorldBench/code/taco/new_test_200.json \
    --out_dir output/taco \
    --max_samples 5
```

Inference scripts for other benchmarks are located in

```text
scripts/awm_infer/
```

---

## Evaluation

After inference, evaluate the generated results using the corresponding evaluator.

For example,

```bash
python -m multiworldbench.eval.video_eval \
    --real_dir data/MultiWorldBench/general/coin/ground_truth_videos \
    --gen_dir output/generated_videos \
    --device cuda
```

Evaluation scripts are provided for

- Video prediction
- Image prediction
- Code generation
- Tool use

See `multiworldbench/eval/` for details.
