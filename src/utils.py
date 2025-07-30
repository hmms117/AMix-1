# Copyright (c) 2025 Institute for AI Industry Research (AIR), Tsinghua University, and AI For Science Group, Shanghai Artificial Intelligence Laboratory
# SPDX-License-Identifier: Apache-2.0
import os
import argparse
import json
import glob

def parse_fasta(file_path):
    sequences = []
    with open(file_path, 'r') as f:
        seq_id = None
        seq = []
        annotations = {}
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_id:
                    sequences.append({
                        'id': seq_id,
                        'seq': ''.join(seq),
                        'annotations': annotations
                    })
                header = line[1:].split()
                seq_id = header[0] if header else "unnamed"
                
                # 安全解析注释，处理没有 = 符号的情况
                annotations = {}
                for item in header[1:]:
                    if '=' in item:
                        k, v = item.split('=', 1)  # 只分割第一个 = 符号
                        annotations[k] = v
                    else:
                        # 对于没有 = 的项，使用该项作为键，值为 True
                        annotations[item] = True
                
                seq = []
            else:
                seq.append(line)
        if seq_id:
            sequences.append({
                'id': seq_id,
                'seq': ''.join(seq),
                'annotations': annotations
            })
    return sequences

def write_fasta(sequences, file_path):
    with open(file_path, 'w') as f:
        for seq in sequences:
            # 构建注释字符串
            annotations_list = []
            for k, v in seq['annotations'].items():
                if v is True:  # 对于布尔值 True，只输出键
                    annotations_list.append(f"{k}")
                else:  # 对于其他值，输出 key=value
                    annotations_list.append(f"{k}={v}")
            
            annotations = ' '.join(annotations_list)
            if annotations:
                f.write(f">{seq['id']} {annotations}\n")
            else:
                f.write(f">{seq['id']}\n")
            f.write(f"{seq['seq']}\n")



def create_initial_json(init_dir, output_json, eval_task="pLDDT", higher_better=True):
    """从初始 A3M 文件创建 JSON 数据文件
    
    Args:
        init_dir: 包含初始 A3M 文件的目录
        output_json: 输出的 JSON 文件路径
        eval_task: 评测任务名称
        higher_better: 评测指标是否越高越好
    """
    # 获取所有 A3M 文件
    a3m_files = glob.glob(os.path.join(init_dir, "*.a3m"))
    if not a3m_files:
        a3m_files = glob.glob(os.path.join(init_dir, "*.fasta"))
        print(f"No A3M files found in {init_dir}, using fasta files instead")
        if not a3m_files:
            raise ValueError(f"No A3M or fasta files found in {init_dir}")
    
    # 创建 JSON 数据结构
    json_data = []
    
    # 处理每个 A3M 文件
    for a3m_file in a3m_files:
        # 从文件名获取样本 ID（去掉扩展名）
        sample_id = os.path.splitext(os.path.basename(a3m_file))[0]
        
        # 读取序列
        sequences = parse_fasta(a3m_file)
        
        # 创建样本数据
        sample_data = {
            "sample_meta": {
                "id": sample_id,
                "eval_task": eval_task,
                "higher_better": higher_better
            },
            "rounds": {
                "0": []  # 初始轮次
            }
        }
        
        # 添加序列到初始轮次
        for seq in sequences:
            sample_data["rounds"]["0"].append({
                "seq": seq["seq"],
                "score": {}  # 初始为空，后续由评测填充
            })
        
        json_data.append(sample_data)
    
    # 保存 JSON 文件
    with open(output_json, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Created initial JSON data with {len(json_data)} samples")
    print(f"Output saved to: {output_json}")