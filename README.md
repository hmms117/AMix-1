# AMix-1: A Pathway to Test-Time Scalable Protein Foundation Model
[![deploy](https://img.shields.io/badge/Project-Homepage-blue)](https://gensi-thuair.github.io/AMix-1/)
[![arXiv](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2507.08920)
[![deploy](https://img.shields.io/badge/Hugging%20Face-AMix_1_1.7B-FFEB3B)](https://huggingface.co/GenSI/AMix-1-1.7B)

## Introduction
We introduce **AMix-1**, a powerful protein foundation model built on Bayesian Flow Networks and empowered by a systematic training methodology, encompassing **pretraining scaling laws**, **emergent capability analysis**, **in-context learning mechanism**, and **test-time scaling algorithm**.

<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/intro.png" style="width: 85%" />
</div>

## 🔧 Installation Guide

```bash
# Clone the repository
git clone https://github.com/GenSI-THUAIR/AMix-1.git
cd AMix-1

# Create and activate a Python 3.10 environment (recommended: conda)
conda create -n amix python=3.10 -y
conda activate amix

# Alternatively, using venv
# python3.10 -m venv venv
# source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```


## Inference

### Download

Download config.yaml and model checkpoint through Huggingface:
```bash
# login through your HuggingFace key
huggingface-cli login
# run code in src
cd src
# download the whole AMix-1-1.7B directory
huggingface-cli download GenSI/AMix-1-1.7B --local-dir ./AMix-1-1.7B --local-dir-use-symlinks False
# adjust directory structure
mv AMix-1-1.7B/.hydra AMix-1-1.7B/ckpt ./
# back to AMix-1
cd ..
```

Before inference, the directory structure should be:
```
AMix-1/
├── imgs
├── src
│   ├── .hydra
│   │   └── config.yaml
│   ├── ckpt
│   │   └── AMix-1-1.7b.ckpt
│   ├── model.py
│   ├── inference.py
│   └── ...
│
├── README.md
├── inference.sh
└── ...
```

### Important Parameters
Parameters for `inference.sh`

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--input_seq` | Input MSA, a single sequence or multiple sequences with equal length | `"ASAAA"` or `"AAASA,SAASA,ASASA"` |
| `--output_dir` | Sequences generated per round | `"./output"` |
| `--num_seq` | Number of sequences to generate | `10` |
| `--time` | Noise factor for generation (0.0 to 1.0), 1.0 means no noise | `0.8` |
| `--ckpt_path` | Checkpoint path for the model | `"./ckpt/AMix-1-1.7b.ckpt"` |

### Run

```angular2html
sh inference.sh --input_seq "AAASASA" --output_dir "./output" --num_seq 10 --time 0.8 --ckpt_path "./ckpt/AMix-1-1.7b.ckpt"
```

## Test-time Scaling: EvoAMix-1

<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/TTS.png" style="width: 100%" />
</div>

EvoAMix-1 is our test-time scaling algorithm that iteratively evolves protein sequences through generation, evaluation, and selection cycles. This approach enables continuous improvement of protein designs by leveraging multiple evaluation metrics and sophisticated filtering strategies.

### 🚀 Quick Start

#### Launch Script: `run_tts.sh`

We provide a convenient bash script `run_tts.sh` to launch the EvoAMix-1 pipeline with proper parameter configuration:

```bash
./run_tts.sh --rounds 5 --num-seqs 10 --top-k 3 --eval-task-weights "pLDDT:1.0,pTM:0.5"
```

#### Key Configuration Parameters

Before running, you **must** configure these essential paths in `run_tts.sh`:

```bash
# TODO: Set your model checkpoint file path
CKPT_FILE="./ckpt/AMix-1-1.7b.ckpt"

# TODO: Set your initial data directory path  
INIT_DATA="/path/to/your/data/CASP14_orphan"

# TODO: Set your experiment output base directory
EXP_BASE_DIR="/path/to/your/experiments"
```

#### Important Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--rounds` | Number of evolution iterations | `5` |
| `--num-seqs` | Sequences generated per round | `10` |
| `--top-k` | Top sequences selected for next round | `3` |
| `--eval-task-weights` | Evaluation metrics and weights | `"pLDDT:1.0,rosetta_energy:-1.0"` |
| `--eval-filter` | Filter criteria for sequences | `"pLDDT>80,progen_nll<10"` |
| `--infer-step` | Inference steps for generation | `10` |
| `--esm-fold-gpus` | Number of GPUs for ESM-Fold | `1` |

#### Evaluation Task Weights Format

The evaluation weights use the format `"metric1:weight1,metric2:weight2"`:
- **Positive weights**: Higher scores are better (e.g., `pLDDT:1.0`)
- **Negative weights**: Lower scores are better (e.g., `rosetta_energy:-1.0`)

#### Evaluation Filters Format

Filters use the format `"metric1>value1,metric2<value2"` and support operators: `>`, `<`, `>=`, `<=`, `==`, `!=`

### 🔧 Custom Evaluation Metrics

#### Adding New Plug-and-Play Verifiers

You can easily extend EvoAMix-1 with custom evaluation metrics by implementing new verifiers in `src/evaluation/metrics.py`. Each verifier should follow this pattern:

```python
def evaluate_your_metric(eval_file, args=None):
    """
    Evaluate sequences using your custom metric
    
    Args:
        eval_file: Path to FASTA file containing sequences
        args: Optional command line arguments
    
    Returns:
        Dictionary mapping sequence IDs to scores
    """
    logger.info("=== Starting your metric evaluation ===")
    
    abs_eval_file = os.path.abspath(eval_file)
    sequences = parse_fasta(abs_eval_file)
    
    metric_scores = {}
    
    for seq_record in sequences:
        seq_id = seq_record.id
        sequence = str(seq_record.seq)
        
        # Implement your evaluation logic here
        score = your_evaluation_function(sequence)
        
        # Store score in annotations
        seq_record.description += f" your_metric={score:.4f}"
        if 'annotations' not in seq_record.__dict__:
            seq_record.annotations = {}
        seq_record.annotations['your_metric'] = score
        
        metric_scores[seq_id] = score
    
    # Write updated sequences back to FASTA
    write_fasta(sequences, abs_eval_file)
    
    logger.info("=== Your metric evaluation completed ===")
    return metric_scores
```

#### Requirements for Custom Verifiers

1. **Function signature**: Must accept `eval_file` and optional `args` parameters
2. **Score annotation**: Store scores in FASTA description as `metric_name=value`
3. **Sequence annotations**: Store scores in `seq_record.annotations[metric_name]`
4. **File update**: Write updated sequences back to the original FASTA file
5. **Registration**: Add your function to the `METRICS` dictionary:

```python
METRICS = {
    # ... existing metrics
    "your_metric": evaluate_your_metric,
}
```

#### Available Built-in Verifiers

We provide several built-in evaluation metrics:

- **`pLDDT`**: Protein local structure quality (via ESM-Fold)
- **`pTM`**: Protein template modeling score (via ESM-Fold)
- **`TM_score`**: Structural similarity using TM-align
- **`rosetta_energy`**: Rosetta energy scoring
- **`novelty`**: Sequence novelty compared to original
- **`diversity`**: Intra-sample sequence diversity
- **`repeatness`**: Amino acid repeat analysis
- **`CLEAN_EC`**: Enzyme function prediction
- **`identical`**: Sequence identity to original

### ⚠️ Environment Management for Custom Verifiers

**Important**: Different verifiers may have conflicting dependencies or specific environment requirements. To avoid conflicts, we strongly recommend:

#### 1. Create Isolated Environments

Create separate conda/virtual environments for verifiers with complex dependencies:

```bash
# Example: Create environment for a deep learning verifier
conda create -n verifier_env python=3.10
conda activate verifier_env
pip install your_verifier_requirements

# Example: Create environment for structure-based verifiers
conda create -n structure_env python=3.9
conda activate structure_env
pip install pymol biopython
```

#### 2. Use Subprocess Calls

Implement verifiers that require special environments using subprocess calls:

```python
def evaluate_external_verifier(eval_file, args=None):
    """Example of calling external verifier in isolated environment"""
    logger.info("=== Starting external verifier evaluation ===")
    
    # Call external script in specific environment
    cmd = [
        "conda", "run", "-n", "verifier_env", 
        "python", "/path/to/external_verifier.py",
        "--input", eval_file,
        "--output", eval_file  # Update in-place
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("External verifier completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"External verifier failed: {e}")
        raise
    
    # Parse results from updated FASTA file
    sequences = parse_fasta(eval_file)
    scores = {}
    for seq in sequences:
        # Extract score from description or annotations
        scores[seq.id] = extract_score_from_seq(seq)
    
    return scores
```

### 📊 Output and Results

EvoAMix-1 generates comprehensive outputs in your experiment directory:

```
exp_dir/
├── tts.log                    # Main execution log
├── error.log                  # Error and warning log
├── data.json                  # Complete evolution history
├── round_1/
│   ├── filtered_sequences_round_1.json
│   └── tmp_round_1_all_samples.fasta
├── round_2/
│   └── ...
└── plots/                     # Evolution curve visualizations
    ├── sample_0_metrics_plot.png
    ├── all_samples_weighted_score_plot.png
    └── average_metrics_plot.png
```

The evolution process tracks all metrics across rounds, enabling detailed analysis of sequence improvement over time.

## Citation

```bibtex
@article{lv2025amix1,
  title={AMix-1: A Pathway to Test-Time Scalable Protein Foundation Model},
  author={Changze Lv*, Jiang Zhou*, Siyu Long*, Lihao Wang, Jiangtao Feng, Dongyu Xue, Yu Pei, Hao Wang, Zherui Zhang, Yuchen Cai, Zhiqiang Gao, Ziyuan Ma, Jiakai Hu, Chaochen Gao, Jingjing Gong, Yuxuan Song, Shuyi Zhang, Xiaoqing Zheng, Deyi Xiong, Lei Bai, Ya-Qin Zhang, Wei-Ying Ma, Bowen Zhou, Hao Zhou},
  journal={arXiv preprint arXiv:2507.08920},
  year={2025}
}
```
