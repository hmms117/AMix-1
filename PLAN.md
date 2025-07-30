# EvoAMix-1 / evo-GPT-X Design Plan

This document captures the **high-level architecture** we agreed on for running multiple verifier models (ESMFold, MiniFold, expression/stability predictors, etc.) in a resource-efficient way on a small GPU cluster managed by Slurm.

---
## 0 · Prompt specification

### 0.1 SYSTEM message (model set-up)
```text
You are an expert protein engineer and machine-learning scientist.
Your goal is to help me optimise protein sequences for multiple partially-conflicting
objectives under specific assay conditions.

◼  Base LM              : “evo-GPT-X” (AA-token-level LLM, 2 – 8 B params)
◼  Fine-tuning paradigm : Offline RL with Quantile-Reward Policy Optimisation (QRPO)
                           or Residual-Policy Optimisation (ResidPO).  No on-policy
                           PPO / RFXL sampling.
◼  Safety               : NEVER propose sequences that violate BSL-1 rules.
```

### 0.2 USER prompt template
```text
### TASK CONTEXT
• Protein family      : {FAMILY_TAG}        # e.g. GH13
• Assay pH            : {PH_VALUE}          # float
• Assay temperature °C: {TEMP_VALUE}        # int
• Target(s)           : expression | activity | stability
                        (stability reported as either
                         – “half-life improvement vs ref”  OR
                         – “residual activity”)
• Mode                : {rank | design | mutate}

### DATA BUNDLE
{JSON-OR-CSV-BLOCK}

### WHAT TO DO
1. ***Rank***   each supplied sequence ➜ return a rank / scalar per target (0-1).
2. ***Design*** N novel sequences maximising a weighted objective
   score = w_expr·norm(expr) + w_act·norm(activity) + w_stab·norm(stability)
3. ***Mutate*** supplied parents with ≤k edits, respecting MSA conservation.

### OFFLINE RL SET-UP (background)
• Use the historic assay table (~1000 assays × 3-200 samples) as (prompt,reward).
• QRPO specifics: β ∈ {0.1, 0.03, 0.003}; n_ref = 3.
• Dataset need only (prompt, completion, reward).

### OUTPUT FORMAT (YAML)
mode      : score|design|mutate
context   :
  family      : GH13
  pH          : 7.5
  temperature : 37
results:
  - seq_id           : ...
    sequence         : "MKWVTFISLLFL..."
    expression_score : 0.72
    activity_score   : 0.81
    stability_score  : 0.65
    overall_score    : 0.73
    lineage          : parent-123 | de-novo
    notes            : "single P143A mutation improved half-life 1.4×"
```

---
## 1 · End-to-end round flow
```
(generator)   round_3.fasta ───▶ driver.py
                                      │
                                      ├─ submit HTTP batch ➜ expression-srv
                                      ├─ submit HTTP batch ➜ stability-srv
                                      ├─ submit HTTP batch ➜ minifold-srv
                                      └─ submit HTTP batch ➜ esmfold-srv
                                                │
                                    (each service appends scores to FASTA/JSON)
                                      │
                    all .scores_ready ✔ │
                                      ▼
                            filter_top_k()   →   next round
```

* `round_k.fasta` (and an adjacent `round_k.json`) is the **single source-of-truth** for scores.  Every verifier appends its metric under `annotations` keyed by sequence ID.  If the key already exists ➜ skip processing (idempotent caching).

---
## 2 · Verifier micro-service pattern (FastAPI)

```text
1×GPU  ==  1×FastAPI server  ==  1 heavy model kept resident in VRAM
```

Example: **MiniFold pLDDT**
```python
# verify_minifold.py
from fastapi import FastAPI
from pydantic import BaseModel
import torch, uvicorn, hashlib
from minifold import MiniFoldModel  # placeholder import

app = FastAPI()
DEVICE = "cuda"
MODEL  = MiniFoldModel.from_pretrained("facebook/minifold").to(DEVICE).eval()
CACHE  = {}

class Query(BaseModel):
    seqs: list[str]

@torch.no_grad()
@app.post("/score")
def score(q: Query):
    out = []
    for s in q.seqs:
        h = hashlib.sha256(s.encode()).hexdigest()
        if h not in CACHE:
            plddt = MODEL.score([s], batch_size=1, device=DEVICE)[0]
            CACHE[h] = plddt
        out.append(CACHE[h])
    return {"pLDDT": out}
```

Slurm wrapper (single GPU, 7-day wall-clock):
```bash
#!/usr/bin/env bash
#SBATCH -J minifold_srv
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00
#SBATCH --cpus-per-task=8
#SBATCH -o logs/minifold_%j.out

module load gcc cuda
uvicorn verify_minifold:app --host 0.0.0.0 --port 8000
```

> **Note** – Similar services are created for `esm2_activity`, `esm2_expression`, `esmfold`, etc.  Light verifiers (≤2 GiB) can be grouped into one service.

---
## 3 · Client wrapper inside `evaluation/metrics.py`

```python
import requests, json
from utils import parse_fasta, write_fasta

SERVICE_HOST = "http://minifold.node:8000"

def evaluate_plddt_service(eval_file, args=None):
    seqs = parse_fasta(eval_file)
    batch = [s['seq'] for s in seqs if 'pLDDT' not in s['annotations']]
    if not batch:
        return {}   # all cached
    r = requests.post(f"{SERVICE_HOST}/score", json={"seqs": batch}, timeout=900)
    scores = r.json()["pLDDT"]
    j = 0
    for s in seqs:
        if 'pLDDT' in s['annotations']:
            continue
        s['annotations']['pLDDT'] = str(scores[j])
        j += 1
    write_fasta(seqs, eval_file)
    return {rec['id']: float(rec['annotations']['pLDDT']) for rec in seqs}
```
Register in the metric map:
```python
METRICS['pLDDT'] = evaluate_plddt_service
```

---
## 4 · Driver responsibilities (`driver.py`)
1. Launch generator → `round_k.fasta`.
2. For each verifier *service* send one HTTP batch request (can be asyncio gather).
3. Wait until every target metric is present in the FASTA/JSON.
4. Call `filter_top_k()` and write `round_{k+1}.fasta`.
5. Repeat until convergence.

---
## 5 · Resource profile & sharding
| Verifier           | GPU-hours / 1 k seqs | Suggested batch | Service | Notes |
|--------------------|----------------------|-----------------|---------|-------|
| ESMFold            |     4.0             | 1280 AA         | yes     | heavy |
| MiniFold           |     0.8             | 2048 AA         | yes     | moderate |
| ESM2-650 M value   |     0.1             | 64 seq          | yes/merge | expression / activity |
| Light heuristics   |   <0.01             | 512 seq         | merged  | novelty, repeatness |

If the cluster has ≤4 GPUs run heavy services on dedicated GPUs and merge all light verifiers into one catch-all service.

---
## 6 · Fault tolerance
* **Caching:** SHA-256(sequence) → score dict inside each service (RAM + optional Redis).
* **Idempotence:** verifiers skip sequences already annotated.
* **Graceful shutdown:** trap `SIGTERM`, finish current batch, pickle unfinished requests to `/tmp/replay.json`.
* **Slurm restart:** use `--requeue` or an external cron to restart services weekly.

---
## 7 · Future extensions
* Swap FastAPI for **Ray Serve** when you outgrow single-GPU services.
* Containerise each verifier with Docker + `slurm-docker-runner` to guarantee dependency isolation.
* Expose `/model_info` returning git commit and checksum → saved in `round_k.json` for provenance.

---
**Last updated:** <!-- TODO: date automatically filled by CI -->