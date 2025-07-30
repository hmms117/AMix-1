'''
Some code modified partially from ESM implementation in Huggingface and DPLM (https://github.com/bytedance/dplm).
---------------------------
Copyright (c) 2025 Institute for AI Industry Research (AIR), Tsinghua University, and AI For Science Group, Shanghai Artificial Intelligence Laboratory
SPDX-License-Identifier: Apache-2.0
'''

from pathlib import Path
import hydra
from omegaconf import OmegaConf
import sys
import torch
from transformers import EsmTokenizer
import torch.nn.functional as F
import argparse
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import re
from torch.nn.utils.rnn import pad_sequence
from typing import Iterable
from torch.utils.data import Sampler, BatchSampler
from tqdm import tqdm
import numpy as np
from typing import Iterable
import math
import logging
import random

tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t30_150M_UR50D')

def set_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class SortishSampler(Sampler):
    """
    Returns indices such that inputs with similar lengths are close together.
    Modified from DPLM (https://github.com/bytedance/dplm)
    """

    def __init__(
        self, sequence_lengths: Iterable, bucket_size: int, num_replicas: int = 1, rank: int = 0
    ):
        if dist.is_initialized():
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
        self.data = np.argsort(sequence_lengths)
        self.num_replicas = num_replicas
        self.num_samples = int(math.ceil(len(self.data) * 1.0 / self.num_replicas))
        self.bucket_size = bucket_size
        n_buckets = int(np.ceil(len(self.data) / self.bucket_size))
        self.data = [
            self.data[i * bucket_size : i * bucket_size + bucket_size] for i in range(n_buckets)
        ]
        self.rank = rank
        self.epoch = 0
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        np.random.seed(self.epoch)
        for bucket in self.data:
            np.random.shuffle(bucket)
        np.random.shuffle(self.data)
        indices = [item for sublist in self.data for item in sublist]
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size
        # subsample
        start = self.rank * self.num_samples
        end = start + self.num_samples
        indices = indices[start:end]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

class ApproxBatchSampler(BatchSampler):
    """
    Modified from DPLM (https://github.com/bytedance/dplm)
    
    Parameters:
    -----------
    sampler : Pytorch Sampler
            Choose base sampler class to use for bucketing

    max_tokens : int
            Maximum number of tokens per batch

    max_batch: int
            Maximum batch size

    sample_lengths : array-like
            List of lengths of sequences in the order of the dataset
    """

    def __init__(
        self,
        sampler,
        max_tokens,
        max_batch,
        sample_lengths,
        max_square_tokens=np.inf,
        drop_last=False,
        batch_size=None,
        max_len=512,
        debug_mode = False,
    ):
        super().__init__(sampler, max_batch, drop_last)
        self.longest_token = 0
        self.max_tokens = max_tokens
        self.max_batch = max_batch
        self.sampler = sampler
        self.sample_lengths = sample_lengths
        self.max_square_tokens = max_square_tokens
        self.max_len = max_len
        self.debug_mode = debug_mode
        self.batches = self._build_batches()
        
        
    def _build_batches(self):
        batches = []
        length = 0
        ell_sq = 0
        batch = []
        for i, idx in enumerate(self.sampler):
            this_length = min(self.max_len, self.sample_lengths[idx])
            linear = (len(batch) + 1) * max(length, this_length)
            quadratic = (len(batch) + 1) * max(ell_sq, this_length**2)
            if linear <= self.max_tokens and quadratic < self.max_square_tokens:
                batch.append(idx)
                length = max(length, this_length)
                ell_sq = max(ell_sq, this_length**2)
                if len(batch) == self.max_batch:
                    batches.append(batch)
                    batch = []
                    length = 0
            else:
                if len(batch) == 0:
                    print('Current batch is empty! idx is ', idx)
                    continue
                batches.append(batch)
                batch = [idx]
                length = this_length
                ell_sq = this_length**2
            if self.debug_mode and len(batches) > 150:
                break
        if len(batch) > 0:
            batches.append(batch)
            
        if self.sampler.num_replicas > 1:
            num_samples = torch.tensor(len(batches)).cuda()
            print(f'==============Local Rank {self.sampler.rank} Num Samples {num_samples}==============')
            dist.all_reduce(num_samples, op=dist.ReduceOp.MAX)
            print(f'==============All Reduce Num Samples {num_samples}==============')
            num_samples = num_samples.item()

            if len(batches) < num_samples:
                # padding_size = num_samples - len(batches)
                a = num_samples // len(batches)
                b = num_samples % len(batches)
                new_batches = batches * a
                new_batches += batches[:b]
                assert len(new_batches) == num_samples
                batches = new_batches
            print(f'==============After Reduce, Rank{self.sampler.rank}, Num Samples {num_samples}==============')
        return batches
            
    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch

def discreteBayesianFlow(t, x, beta1, beta_time_order=2, mask = None, seed=None):
    """
    Args:
        t: [B, N]
        x: [B, N, K], already one-hot
        beta1: [B, N]
    """
    # last_index = mask.sum(dim=-1).long() -1
    # if mask is not None:
    if seed is not None:
        set_seed(seed)
    
    K = x.size(-1)
    beta = beta1 * (t**beta_time_order)  # (B, N)
    beta = beta.unsqueeze(-1)  # (B, N, 1)
    mean = beta * (K * x - 1)  # (B, N, K)
    std = (beta * K).sqrt()  # (B, N, 1)
    eps = torch.randn_like(mean)  # (B, N, K)
    y = mean + std * eps  # (B, N, K)
    theta = F.softmax(y, dim=-1)  # (B, N, K)
    if mask is not None:
        theta = theta * mask[...,None] + (1 - mask[...,None]) * x
    return theta

def extract_profile(input_seq):
    results = []
    seqs = [re.sub(r"[a-z]", "", seq) for seq in input_seq]

    profile = tokenizer.batch_encode_plus(seqs,add_special_tokens=True, padding="longest", return_tensors='pt')
    profile = F.one_hot(profile['input_ids'], num_classes=len(tokenizer)).sum(dim=0)
    
    profile = (profile/profile.sum(dim=-1)[...,None])
    results.append({"profile": profile})
    return results

class ProfileDataset(Dataset):
    def __init__(
        self, input_seq, num_seq = 10,
    ):
        self.results = extract_profile(input_seq) * num_seq
        self.meta_lens = [x['profile'].shape[0] for x in self.results]
    def __len__(self):
        return len(self.results)
    def get_metadata_lens(self):
        return self.meta_lens
    def __getitem__(self, idx):
        return self.results[idx]
    def collate(self, batch_list):
        # lens = [x['profile'].shape[0] for x in batch_list]
        lengths = torch.tensor(
            [x['profile'].shape[0] for x in batch_list],
            requires_grad=False,
        )
        max_L = max(lengths)
        padding_mask = torch.arange(max_L).expand(
            len(lengths), max_L
        ) < lengths.unsqueeze(1)

        profiles = pad_sequence([x['profile'] for x in batch_list],
                    batch_first=True,
                    padding_value=0,
                )
        batch = {
            'inputs_embeds': profiles,
            'attention_mask': padding_mask.bool(),
        }
        return batch

def get_dataloader(input_seq, num_seq = 10):
    dataset = ProfileDataset(input_seq, num_seq)
    lens = dataset.get_metadata_lens()
    sampler = SortishSampler(lens, 1000, num_replicas=1, rank=0)
    batch_sampler = ApproxBatchSampler(sampler, 40000, 800, lens, max_len=1024)
    dl = DataLoader(
        dataset=dataset, 
        batch_sampler=batch_sampler, 
        num_workers=4, 
        pin_memory=True, 
        collate_fn=dataset.collate
    )
    return dl

def generation(model, batch, start_t = 0.2, infer_step = 100, seed=None):
    if seed is not None:
        set_seed(seed)
    inputs_embeds = batch['inputs_embeds'].to(model.device).to(model.dtype)
    attention_mask = batch['attention_mask'].to(model.device)
    mask = attention_mask.clone()
    idx_mask = mask.sum(dim=-1)- 1
    mask = mask.to(model.dtype)
    mask[torch.arange(mask.shape[0]),idx_mask] = 0
    mask[:,0] = 0
    probs = inputs_embeds
    for idx, _t in enumerate(tqdm(np.linspace(start = start_t, stop=1.0, num=infer_step + 1)[:-1])):
        t = (torch.ones_like(attention_mask) * _t).to(inputs_embeds)
        # if idx > 0:
        inputs_embeds = discreteBayesianFlow(t, probs, beta1=model.bfn.cfg.beta1, beta_time_order=model.bfn.cfg.beta_time_order, mask = mask, seed=seed)
        pred_logits = model.bfn.forward(t, inputs_embeds, attention_mask)
        probs = torch.nn.functional.softmax(pred_logits, dim=-1) * mask[..., None] + inputs_embeds * (1 - mask[..., None])
    
    probs = probs[...,1:-1,:]
    probs[...,:4]= 0
    probs[...,24:] = 0
    output_results = [
        "".join(seq.split(" "))
        for seq in tokenizer.batch_decode(torch.argmax(probs, dim=-1), skip_special_tokens=True)
    ]
    return output_results

def calc_similarity(seq, input_seqs):
    def single_similarity(a, b):
        if len(a) != len(b):
            return 0.0
        return sum(x == y for x, y in zip(a, b)) / len(a)
    return max(single_similarity(seq, s) for s in input_seqs)

def load_and_generate(
    input_seq,
    num_seq=10,
    ckpt_path='./ckpt/AMix-1-1.7b.ckpt',
    time=0.2,
    seed=None
):
    if seed is not None:
        set_seed(seed)
    root_path = Path(ckpt_path).parents[1]
    sys.path.append(str(root_path))
    cfg_path = Path(root_path, ".hydra", "config.yaml")
    ckpt_cfg = OmegaConf.load(cfg_path)

    ckpt_cfg.model.bfn.net.config._attn_implementation = 'sdpa'
    ckpt_cfg.model.bfn.cfg.num_diffusion_timesteps = 100

    model = hydra.utils.instantiate(ckpt_cfg.model)
    model.load_state_dict(torch.load(ckpt_path, map_location='cuda')['state_dict'])
    model = model.cuda()

    print(model)

    dataloader = get_dataloader(input_seq, num_seq)

    total_output = []
    with torch.no_grad():
        for batch in dataloader:
            output = generation(model, batch, start_t= time, infer_step=ckpt_cfg.model.bfn.cfg.num_diffusion_timesteps,seed=seed)
            total_output.append(output)

    return total_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Protein sequence generation demo")
    parser.add_argument('--input_seq', type=str, required=True, help='Input MSA, e.g. "AAASA,SAASA,ASASA"')
    parser.add_argument('--output_dir', type=str, default="./output", help='Output directory for generated sequences')
    parser.add_argument('--num_seq', type=int, default=10, help='Number of sequences to generate')
    parser.add_argument('--time', type=float, default=0.2, help='Noise factor for generation (0.0 to 1.0), 1.0 means no noise')
    parser.add_argument('--ckpt_path', type=str, default='./ckpt/AMix-1-1.7b.ckpt', help='Checkpoint path for the model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    error_log_file = output_path / 'error.log'

    logging.basicConfig(
        filename=str(error_log_file),
        filemode='w',
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.ERROR
    )

    input_seq = args.input_seq.replace(" ", "").split(",")
    print(f"Input MSA: {input_seq}")

    seq_lens = [len(seq) for seq in input_seq]
    if len(set(seq_lens)) != 1:
        raise ValueError("Sequences must have the same length!")
    if not (0.0 <= args.time <= 1.0):
        raise ValueError("Time must be between 0.0 and 1.0!")

    outputs = load_and_generate(
        input_seq=input_seq,
        num_seq=args.num_seq,
        ckpt_path=args.ckpt_path,
        time=args.time,
        seed=args.seed
    )
    print(f"Output: {outputs[0]}")
