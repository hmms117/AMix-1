# AMix-1: A Pathway to Test-Time Scalable Protein Foundation Model
[![deploy](https://img.shields.io/badge/Project-Homepage-blue)](https://gensi-thuair.github.io/AMix-1/)
[![arXiv](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2507.08920)
[![deploy](https://img.shields.io/badge/Hugging%20Face-AMix_1_1.7B-FFEB3B)](https://huggingface.co/GenSI/AMix-1-1.7B)

## Introduction
We introduce **AMix-1**, a powerful protein foundation model built on Bayesian Flow Networks and empowered by a systematic training methodology, encompassing **pretraining scaling laws**, **emergent capability analysis**, **in-context learning mechanism**, and **test-time scaling algorithm**.

<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/intro.png" style="width: 85%" />
</div>

## Installation

@Zhoujiang

## Inference

Download config.yaml and model checkpoint

For a single sequence:
```angular2html
python inference.py --input_seqs "AAASASA" --num_seq 1 --novelty 0.2
```

For multiple sequence alignment (MSA):
```angular2html
python inference.py --input_seqs "AAASASA" --num_seq 10 --novelty 0.2
```

## Test-time Scaling

@Zhoujiang

## Citation

```bibtex
@article{lv2025amix1,
  title={AMix-1: A Pathway to Test-Time Scalable Protein Foundation Model},
  author={Changze Lv*, Jiang Zhou*, Siyu Long*, Lihao Wang, Jiangtao Feng, Dongyu Xue, Yu Pei, Hao Wang, Zherui Zhang, Yuchen Cai, Zhiqiang Gao, Ziyuan Ma, Jiakai Hu, Chaochen Gao, Jingjing Gong, Yuxuan Song, Shuyi Zhang, Xiaoqing Zheng, Deyi Xiong, Lei Bai, Ya-Qin Zhang, Wei-Ying Ma, Bowen Zhou, Hao Zhou},
  journal={arXiv preprint arXiv:2507.08920},
  year={2025}
}
```
