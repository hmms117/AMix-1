# Copyright (c) 2025 Institute for AI Industry Research (AIR), Tsinghua University, and AI For Science Group, Shanghai Artificial Intelligence Laboratory
# SPDX-License-Identifier: Apache-2.0
import re
import json
import random

def load_data_from_json(json_path):
    """Load data from JSON file in the new structure."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def get_mutated_seqs(seqs, num_per_seq, mutate_ratio=0.1):
    mutated_seqs = []
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'  # standard amino acids

    for seq in seqs:
        for _ in range(num_per_seq):
            seq_list = list(seq)
            num_mutations = max(1, int(len(seq) * mutate_ratio))
            mutation_positions = random.sample(range(len(seq)), num_mutations)

            for pos in mutation_positions:
                original = seq_list[pos]
                choices = [aa for aa in amino_acids if aa != original]
                seq_list[pos] = random.choice(choices)

            mutated_seq = ''.join(seq_list)
            mutated_seqs.append(mutated_seq)

    return mutated_seqs

def mutation(json_file, round_idx, new_round_idx, generation_params, mutation_type="random"):
    # Load JSON data
    data = load_data_from_json(json_file)
    # 使用字典推导式直接按样本分组序列，减少循环次数
    sequences_by_sample = {}
    sample_to_lengths = {}  # 用于记录每个样本的序列长度
    
    for sample_idx, sample in enumerate(data):
        round_str = str(round_idx)
        if round_str in sample["rounds"]:
            if round_idx == 0:
                # 第一轮只使用原始蛋白序列（第一条）进行突变
                cleaned_seqs = [re.sub(r"[a-z]", "", item["seq"]) for item in sample["rounds"][round_str][:1]]
            else:
                # 此后所有的蛋白序列参与突变
                cleaned_seqs = [re.sub(r"[a-z]", "", item["seq"]) for item in sample["rounds"][round_str][:]]
            
            if cleaned_seqs:  # 只处理非空列表
                sequences_by_sample[sample_idx] = cleaned_seqs
                # 假设同一样本中所有序列长度相同，只需存储一次
                sample_to_lengths[sample_idx] = len(cleaned_seqs[0]) + 1  # +1 for BOS
    
    for sample_idx, sample_sequences in sequences_by_sample.items():
        mutated_seqs=get_mutated_seqs(sample_sequences, generation_params['num_seqs'] // len(sample_sequences), generation_params['mutation_ratio'])
        sample = data[sample_idx]
        new_round_str = str(new_round_idx)
        if new_round_str not in sample["rounds"]:
            sample["rounds"][new_round_str] = []
        else:
            sample["rounds"][new_round_str] = []
        for mutated_seq in mutated_seqs:
            sample["rounds"][new_round_str].append({
                "seq": mutated_seq,
                "score": {}
            })
    
    # 直接保存到原始JSON文件
    with open(json_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Updated JSON saved to {json_file}")
