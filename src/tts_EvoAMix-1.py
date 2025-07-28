import os
import subprocess
import argparse
import json
import matplotlib.pyplot as plt 
import numpy as np 
import logging
from utils import parse_fasta, write_fasta, create_initial_json

from random_mutation import mutation

from evaluation.metrics import verifier
from rollout import run_generation

# Initialize logger
def setup_logger(exp_dir):
    tts_log_file = os.path.join(exp_dir, 'tts.log')
    error_log_file = os.path.join(exp_dir, 'error.log')
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Remove any existing handlers:
    for h in list(root.handlers):
        root.removeHandler(h)

    # Now add tts handlers:
    fh = logging.FileHandler(tts_log_file)
    sh = logging.StreamHandler()
    for h in (fh, sh):
        h.setLevel(logging.INFO)
        h.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        root.addHandler(h)

    # And the error handler:
    eh = logging.FileHandler(error_log_file)
    eh.setLevel(logging.WARNING)
    eh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root.addHandler(eh)


    return logger


logger = None  # Global logger variable

def run_generation_json(json_file, round_idx, new_round_idx, generation_params):
    """Perform batch generation using JSON data structure"""
    run_generation(
        ckpt_path=generation_params['ckpt_file'],
        json_data=json_file,
        round_idx=round_idx,
        num_seqs=generation_params['num_seqs'],
        time=generation_params['init_t'],
        infer_step=generation_params['infer_step']
    )


def run_batch_evaluation(json_file, round_idx, eval_tasks, exp_dir, esm_fold_gpus=1, args=None):
    """Perform batch evaluation on sequences from specified round for all samples
    
    Args:
        json_file: Path to JSON data file
        round_idx: Round index to evaluate
        eval_tasks: List of evaluation tasks or single evaluation task
        exp_dir: Experiment directory path for creating temporary files
        esm_fold_gpus: Number of ESM-Fold workers for parallel computation
    
    Returns:
        Updated JSON data
    """
    # Create round_{round_idx} subdirectory
    round_dir = os.path.join(exp_dir, f'round_{round_idx}')
    os.makedirs(round_dir, exist_ok=True)
    
    # Create temporary FASTA file containing sequences from all samples
    tmp_fasta = os.path.join(round_dir, f'tmp_round_{round_idx}_all_samples.fasta')
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Write all sample sequences to temporary FASTA file
    with open(tmp_fasta, 'w') as f:
        for sample_idx, sample in enumerate(data):
            sample_id = sample["sample_meta"]["id"]
            round_data = sample["rounds"][str(round_idx)]
            for seq_idx, entry in enumerate(round_data):
                f.write(f">{sample_id}_seq{seq_idx} L={len(entry['seq'])}\n{entry['seq']}\n")
    
    # Convert single evaluation task to list
    if isinstance(eval_tasks, str):
        eval_tasks = [eval_tasks]
    
    # Efficiently run multiple evaluation tasks
    logger.info(f"Running evaluation tasks: {', '.join(eval_tasks)}")
    verifier(tmp_fasta, eval_tasks, esm_fold_gpus, args)
    
    # Update JSON data from evaluation results
    evaluated_seqs = parse_fasta(tmp_fasta)
    for sample_idx, sample in enumerate(data):
        sample_id = sample["sample_meta"]["id"]
        round_data = sample["rounds"][str(round_idx)]
        for seq_idx, entry in enumerate(round_data):
            eval_seq = next((seq for seq in evaluated_seqs 
                          if seq['id'] == f"{sample_id}_seq{seq_idx}"), None)
            if eval_seq:
                # Ensure score dictionary exists
                if 'score' not in entry:
                    entry['score'] = {}
                
                # Update scores for each evaluation task
                for eval_task in eval_seq['annotations']:
                    if eval_task in eval_seq['annotations']:
                        if eval_task != "EC":
                            entry['score'][eval_task] = float(eval_seq['annotations'].get(eval_task, 0))
                        else:
                            entry['score'][eval_task] = eval_seq['annotations'].get(eval_task, 0)
                    else:
                        logger.warning(f"Warning: Evaluation result for {eval_task} not found for sample {sample_id} sequence {seq_idx}")
                        # Use default scores to avoid workflow interruption
                        if eval_task == "pLDDT":
                            logger.warning(f"Setting default pLDDT score (50.0) for testing...")
                            entry['score'][eval_task] = 50.0
                        elif eval_task == "pTM":
                            logger.warning(f"Setting default pTM score (0.5) for testing...")
                            entry['score'][eval_task] = 0.5
                        elif eval_task == "rosetta_energy":
                            logger.warning(f"Setting default rosetta_energy score (0.0) for testing...")
                            entry['score'][eval_task] = 0.0
                        elif eval_task == "progen_nll":
                            logger.warning(f"Setting default progen_nll score (10.0) for testing...")
                            entry['score'][eval_task] = 10.0
                        else:
                            logger.warning(f"Setting default {eval_task} score (0.0) for testing...")
                            entry['score'][eval_task] = 0.0
            else:
                logger.warning(f"Warning: Evaluation results not found for sample {sample_id} sequence {seq_idx}")
    
    # Save updated JSON
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    return data


def filter_top_k_samples_json(json_file, sample_idx, filter_window, top_k, round_num, eval_filters=None, eval_weights=None, sort_by="avg_rank"):
    """Filter top-k samples from JSON data and update as input for next round
    
    Args:
        json_file: Path to JSON data file
        sample_idx: Sample index
        filter_window: Filter window size
        top_k: Number of top samples to select
        round_num: Current round number
        eval_filters: Evaluation filter conditions dictionary
        eval_weights: Evaluation weights dictionary
        sort_by: avg_rank or weighted_score
    
    Returns:
        New round index and filtered sequence list
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Get sample
    if isinstance(sample_idx, int) or (isinstance(sample_idx, str) and sample_idx.isdigit()):
        sample_idx = int(sample_idx)
        sample = data[sample_idx]
    else:
        sample = next((s for s in data if s["sample_meta"]["id"] == sample_idx), None)
        if sample is None:
            raise ValueError(f"Sample with ID {sample_idx} not found in data")
    
    # Collect sequences from filter_window rounds
    all_seqs = []
    for i in range(round_num, max(round_num - filter_window, -1), -1):
        round_str = str(i)
        if round_str in sample["rounds"]:
            all_seqs.extend(sample["rounds"][round_str])
    
    # Remove duplicates (based on sequence itself)
    # unique_seqs = list({entry['seq']: entry for entry in all_seqs}.values())
    
    # Apply filters if specified
    # filtered_seqs = unique_seqs
    filtered_seqs = all_seqs
    if eval_filters:
        filtered_seqs = []
        for seq in all_seqs:
            passes_filters = True
            for metric, filter_info in eval_filters.items():
                if metric not in seq['score']:
                    logger.warning(f"Metric {metric} not found in sequence scores, skipping this filter condition")
                    continue
                
                score = seq['score'][metric]
                operator = filter_info['operator']
                threshold = filter_info['threshold']
                
                if operator == '>':
                    if not score > threshold:
                        passes_filters = False
                        break
                elif operator == '>=':
                    if not score >= threshold:
                        passes_filters = False
                        break
                elif operator == '<':
                    if not score < threshold:
                        passes_filters = False
                        break
                elif operator == '<=':
                    if not score <= threshold:
                        passes_filters = False
                        break
                elif operator == '==':
                    if not score == threshold:
                        passes_filters = False
                        break
                elif operator == '!=':
                    if not score != threshold:
                        passes_filters = False
                        break
            
            if passes_filters:
                filtered_seqs.append(seq)
    
    # Calculate weighted scores
    for seq in filtered_seqs:
        seq['weighted_score'] = 0
        for metric, weight in eval_weights.items():
            if metric in seq['score']:
                # If weight is negative, lower values are better; if positive, higher values are better
                value = seq['score'][metric]
                # Normalize weights
                if weight < 0:
                    # Negative weight: lower values are better, so multiplying by negative weight makes larger weighted scores better
                    seq['weighted_score'] += value * weight
                else:
                    # Positive weight: higher values are better
                    seq['weighted_score'] += value * weight
    
    # Calculate ranking for each score
    for seq in filtered_seqs:
        seq['avg_rank'] = 0
    for metric, weight in eval_weights.items():
        if weight == 0:
            continue  # Skip if weight is 0
        # Extract valid scores
        metric_values = [(seq_idx, seq['score'][metric]) for seq_idx, seq in enumerate(filtered_seqs) if metric in seq['score']]
        if not metric_values:
            continue

        # Sort by metric values, determine ascending or descending order (negative weight means lower is better)
        reverse = weight > 0
        metric_values.sort(key=lambda x: x[1], reverse=reverse)

        # Assign ranks (starting from 1)
        for rank, (seq_idx, _) in enumerate(metric_values, start=1):
            filtered_seqs[seq_idx]['avg_rank'] += rank / len(eval_weights)
    
    # Sort by weighted score (always higher is better)
    if sort_by == "weighted_score":
        sorted_seqs = sorted(filtered_seqs, key=lambda x: x.get('weighted_score', 0), reverse=True)
    elif sort_by == "avg_rank":
        sorted_seqs = sorted(filtered_seqs, key=lambda x: x.get('avg_rank', 0), reverse=False)
    else:
        raise ValueError(f"Invalid sort_by value: {sort_by}. Expected 'weighted_score' or 'avg_rank'.")
    
    # Take top-k as next round input
    top_seqs = sorted_seqs[:top_k]
    
    # Create new round
    new_round_idx = round_num + 1
    sample["rounds"][str(new_round_idx)] = top_seqs
    
    # Save updated JSON
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    return new_round_idx, top_seqs


def save_filtered_sequences_to_json(json_file, round_num, exp_dir):
    """Save filtered sequences from each round to separate JSON files
    
    Args:
        json_file: Original JSON data file path
        round_num: Current round number
        exp_dir: Experiment directory
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Create round directory
    round_dir = os.path.join(exp_dir, f'round_{round_num}')
    os.makedirs(round_dir, exist_ok=True)
    
    # Extract sequences from current round
    filtered_data = []
    for sample in data:
        sample_id = sample["sample_meta"]["id"]
        if str(round_num) in sample["rounds"]:
            sample_data = {
                "sample_id": sample_id,
                "sequences": sample["rounds"][str(round_num)]
            }
            filtered_data.append(sample_data)
    
    # Save to new JSON file
    output_file = os.path.join(round_dir, f'filtered_sequences_round_{round_num}.json')
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    logger.info(f"Filtered sequences from round {round_num} saved to {output_file}")


# New functions imported from plot_result_multi_filtered.py
def load_all_filtered_data(exp_dir, rounds):
    """
    Load filtered JSON data from all rounds and return a dictionary
    Keys are round numbers, values are corresponding round data lists
    Note: This example assumes data exists from round_1 to round_{rounds},
          round_0 is not included in plotting.
    """
    all_data = {}
    for r in range(1, rounds + 1):
        file_path = os.path.join(exp_dir, f"round_{r}", f"filtered_sequences_round_{r}.json")
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist")
            continue
        with open(file_path, 'r') as f:
            try:
                all_data[r] = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON file {file_path}: {e}")
    return all_data

# Replace with new plotting function from plot_result_multi_filtered.py, removing dependency on parse_eval_weights
def plot_evaluation_results_filtered(args, sample_id, filtered_data_all_rounds, plot_dir, eval_weights):
    """
    For a single sample, extract evaluation metrics and weighted scores from filtered data,
    plot curves showing mean and standard deviation changes across rounds, and save plots to plot_dir.
    """
    # Set academic style
    import matplotlib as mpl
    mpl.rcParams['figure.figsize'] = (10, 6)
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['xtick.labelsize'] = 11
    mpl.rcParams['ytick.labelsize'] = 11
    mpl.rcParams['legend.fontsize'] = 11
    mpl.rcParams['axes.titlesize'] = 14
    mpl.rcParams['lines.linewidth'] = 2.0
    mpl.rcParams['lines.markersize'] = 6
    
    rounds = args.rounds
    metrics = eval_weights + ['weighted_score']
    main_color = "#3a86ff"  # EvoAMix-1 blue

    for metric in metrics:
        round_nums = []
        means = []
        stds = []
        for r in range(1, rounds + 1):
            data = filtered_data_all_rounds.get(r, [])
            # Find data for specified sample_id
            sample_entry = next((s for s in data if s.get("sample_id") == sample_id), None)
            if sample_entry is None:
                continue
            scores = []
            for entry in sample_entry.get("sequences", []):
                if metric == 'weighted_score':
                    score = entry.get('weighted_score', 0)
                else:
                    score = entry.get('score', {}).get(metric, 0)
                scores.append(float(score))
            if scores:
                round_nums.append(r)
                means.append(np.mean(scores))
                stds.append(np.std(scores))
        
        if not round_nums:
            print(f"Insufficient data for sample {sample_id} metric {metric}, skipping plot")
            continue

        plt.figure(figsize=(10, 6))
        # Main line with EvoAMix-1 blue
        plt.plot(round_nums, means, color=main_color, marker='o', linewidth=2.5, 
                markersize=7, label=f'{metric} Mean')
        # Shaded area for std dev
        plt.fill_between(round_nums,
                         np.array(means) - np.array(stds),
                         np.array(means) + np.array(stds),
                         alpha=0.15, color=main_color, label=f'{metric} Std Dev')
        
        plt.xlabel('Round', fontweight='bold')
        plt.ylabel(metric, fontweight='bold')
        plt.title(f'{metric} vs Round for Sample {sample_id}', fontweight='bold')
        plt.legend(frameon=False)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        out_file = os.path.join(plot_dir, f"{sample_id}_{metric}_plot.png")
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {out_file}")

def plot_all_samples_curves_filtered(args, filtered_data_all_rounds, plot_dir, eval_weights):
    """
    For all samples, plot curves for each evaluation metric based on filtered data,
    one curve per sample, and save plots to plot_dir.
    """
    # Set academic style
    import matplotlib as mpl
    mpl.rcParams['figure.figsize'] = (12, 8)
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['xtick.labelsize'] = 11
    mpl.rcParams['ytick.labelsize'] = 11
    mpl.rcParams['legend.fontsize'] = 10
    mpl.rcParams['axes.titlesize'] = 14
    mpl.rcParams['lines.linewidth'] = 2.0
    mpl.rcParams['lines.markersize'] = 5
    
    metrics = eval_weights
    samples_curves = {}  # { sample_id: { metric: { round: score } } }
    rounds = args.rounds

    # Define colorful palette for better distinction between samples
    color_palette = [
        "#3a86ff",  # Blue
        "#ff6b6b",  # Red
        "#4ecdc4",  # Teal
        "#45b7d1",  # Light Blue
        "#f9ca24",  # Yellow
        "#6c5ce7",  # Purple
        "#a0e7e5",  # Light Teal
        "#feca57",  # Orange
        "#ff9ff3",  # Pink
        "#54a0ff",  # Sky Blue
        "#5f27cd",  # Dark Purple
        "#00d2d3",  # Cyan
        "#ff9f43",  # Light Orange
        "#c44569",  # Dark Pink
        "#40407a",  # Dark Blue
        "#2ed573",  # Green
        "#ffa502",  # Amber
        "#3742fa",  # Indigo
        "#2f3542",  # Dark Gray
        "#57606f"   # Gray
    ]

    for r in range(1, rounds + 1):
        data = filtered_data_all_rounds.get(r, [])
        for sample in data:
            sample_id = sample.get("sample_id")
            if sample_id is None:
                continue
            if sample_id not in samples_curves:
                samples_curves[sample_id] = {metric: {} for metric in metrics}
            for metric in metrics:
                scores = [float(entry.get('score', {}).get(metric, 0))
                          for entry in sample.get("sequences", [])]
                if scores:
                    samples_curves[sample_id][metric][r] = np.mean(scores)

    for metric in metrics:
        plt.figure(figsize=(12, 8))
        color_idx = 0
        for sample_id, metric_dict in samples_curves.items():
            rounds_list = sorted(metric_dict[metric].keys())
            if not rounds_list:
                continue
            scores = [metric_dict[metric][r] for r in rounds_list]
            current_color = color_palette[color_idx % len(color_palette)]
            plt.plot(rounds_list, scores, color=current_color, marker='o', 
                    linewidth=2, markersize=5, label=f'Sample {sample_id}')
            color_idx += 1
            
        plt.xlabel('Round', fontweight='bold')
        plt.ylabel(metric, fontweight='bold')
        plt.title(f'{metric} vs Round for All Samples', fontweight='bold')
        plt.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        out_file = os.path.join(plot_dir, f'all_samples_{metric}_plot.png')
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{metric} curves for all samples saved to {out_file}")

    # Weighted score plotting: weighted score curves for all samples
    weighted_curves = {}
    for r in range(1, rounds + 1):
        data = filtered_data_all_rounds.get(r, [])
        for sample in data:
            sample_id = sample.get("sample_id")
            if sample_id is None:
                continue
            if sample_id not in weighted_curves:
                weighted_curves[sample_id] = {}
            scores = [float(entry.get('weighted_score', 0))
                      for entry in sample.get("sequences", [])]
            if scores:
                weighted_curves[sample_id][r] = np.mean(scores)

    plt.figure(figsize=(12, 8))
    color_idx = 0
    for sample_id, score_dict in weighted_curves.items():
        rounds_list = sorted(score_dict.keys())
        if not rounds_list:
            continue
        scores = [score_dict[r] for r in rounds_list]
        current_color = color_palette[color_idx % len(color_palette)]
        plt.plot(rounds_list, scores, color=current_color, marker='o',
                linewidth=2, markersize=5, label=f'Sample {sample_id}')
        color_idx += 1
        
    plt.xlabel('Round', fontweight='bold')
    plt.ylabel('Weighted Score', fontweight='bold')
    plt.title('Weighted Score vs Round for All Samples', fontweight='bold')
    plt.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    out_file = os.path.join(plot_dir, f'all_samples_weighted_score_plot.png')
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Weighted score curves for all samples saved to {out_file}")

def plot_samples_average_filtered(args, filtered_data_all_rounds, plot_dir, eval_weights):
    """
    Calculate average and standard deviation of evaluation metrics and weighted scores across all samples for each round,
    plot overall average curves, and save plots to plot_dir.
    """
    # Set academic style
    import matplotlib as mpl
    mpl.rcParams['figure.figsize'] = (12, 8)
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['xtick.labelsize'] = 11
    mpl.rcParams['ytick.labelsize'] = 11
    mpl.rcParams['legend.fontsize'] = 11
    mpl.rcParams['axes.titlesize'] = 14
    mpl.rcParams['lines.linewidth'] = 2.5
    mpl.rcParams['lines.markersize'] = 7
    
    metrics = eval_weights + ['weighted_score']
    rounds = args.rounds
    main_color = "#3a86ff"  # EvoAMix-1 blue

    for metric in metrics:
        all_rounds_vals = []
        rounds_list = []
        for r in range(1, rounds + 1):
            data = filtered_data_all_rounds.get(r, [])
            round_vals = []
            for sample in data:
                if metric == 'weighted_score':
                    scores = [float(entry.get('weighted_score', 0))
                              for entry in sample.get("sequences", [])]
                else:
                    scores = [float(entry.get('score', {}).get(metric, 0))
                              for entry in sample.get("sequences", [])]
                if scores:
                    round_vals.append(np.mean(scores))
            if round_vals:
                rounds_list.append(r)
                all_rounds_vals.append(round_vals)
                
        if not rounds_list:
            print(f"Insufficient data for metric {metric}, skipping overall average plot")
            continue
            
        means = [np.mean(vals) for vals in all_rounds_vals]
        stds = [np.std(vals) for vals in all_rounds_vals]
        
        plt.figure(figsize=(12, 8))
        # Main line with thicker width
        plt.plot(rounds_list, means, color=main_color, marker='o', linewidth=3, 
                markersize=8, label=f'Average {metric} Across All Samples')
        # Shaded area for std dev
        plt.fill_between(rounds_list, 
                        np.array(means) - np.array(stds), 
                        np.array(means) + np.array(stds),
                        alpha=0.2, color=main_color, label='Standard Deviation')
        
        plt.xlabel('Round', fontweight='bold')
        plt.ylabel(metric, fontweight='bold')
        plt.title(f'Average {metric} vs Round Across All Samples', fontweight='bold')
        plt.legend(frameon=False)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        out_file = os.path.join(plot_dir, f'average_{metric}_plot.png')
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Overall average {metric} curve saved to {out_file}")

def evaluate_final_round_samples(final_round_idx, exp_dir, esm_fold_gpus=1, args=None):
    """Perform additional pLDDT evaluation on final round filtered samples
    
    Args:
        final_round_idx: Final round index
        exp_dir: Experiment directory path
        esm_fold_gpus: Number of ESM-Fold workers for parallel computation
        args: Command line arguments
    
    Returns:
        Updated JSON data
    """
    logger.info(f"Starting additional pLDDT evaluation for round {final_round_idx} filtered samples...")
    
    final_round_dir = os.path.join(exp_dir, f'round_{final_round_idx}')
    json_file = os.path.join(final_round_dir, f"filtered_sequences_round_{final_round_idx}.json")
    
    # Create temporary FASTA file containing all filtered sequences from final round
    tmp_fasta = os.path.join(final_round_dir, f'final_round_{final_round_idx}_filtered_samples.fasta')
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Write all filtered sequences from final round to temporary FASTA file
    sequence_count = 0
    with open(tmp_fasta, 'w') as f:
        for sample_idx, sample in enumerate(data):
            sample_id = sample["sample_id"]
            round_data = sample["sequences"]
            for seq_idx, entry in enumerate(round_data):
                f.write(f">{sample_id}_final_seq{seq_idx} L={len(entry['seq'])}\n{entry['seq']}\n")
                sequence_count += 1
    
    logger.info(f"Written {sequence_count} final round filtered sequences to temporary FASTA file: {tmp_fasta}")
    
    # Run pLDDT evaluation
    eval_tasks = ["pLDDT"]
    logger.info(f"Running pLDDT evaluation for final round samples...")
    verifier(tmp_fasta, eval_tasks, esm_fold_gpus, args)
    
    # Update JSON data from evaluation results
    evaluated_seqs = parse_fasta(tmp_fasta)
    updated_count = 0
    

    for sample_idx, sample in enumerate(data):
        sample_id = sample["sample_id"]
        round_data = sample["sequences"]
        for seq_idx, entry in enumerate(round_data):
            eval_seq = next((seq for seq in evaluated_seqs 
                            if seq['id'] == f"{sample_id}_final_seq{seq_idx}"), None)
            if eval_seq:
                # Ensure score dictionary exists
                if 'score' not in entry:
                    entry['score'] = {}
                
                # Update pLDDT score, overwrite if already exists
                if "pLDDT" in eval_seq['annotations']:
                    entry['score']["pLDDT"] = float(eval_seq['annotations'].get("pLDDT", 0))
                    updated_count += 1
                    logger.info(f"Updated pLDDT score for sample {sample_id} sequence {seq_idx}: {entry['score']['pLDDT']}")
                else:
                    logger.warning(f"Warning: pLDDT evaluation result not found for sample {sample_id} sequence {seq_idx}")
            else:
                logger.warning(f"Warning: Evaluation results not found for sample {sample_id} sequence {seq_idx}")
    
    logger.info(f"Successfully updated pLDDT scores for {updated_count} sequences")
    
    # Save updated JSON
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Final round filtered samples pLDDT evaluation completed, results saved to {json_file}")
    
    return data

def main():
    parser = argparse.ArgumentParser(description="Evolution-based TTS")
    parser.add_argument('--filter-window', type=int, required=True, help='Sampling space window for top k selection')
    parser.add_argument('--init-data', type=str, required=True, help='Initial data directory containing A3M files')
    parser.add_argument('--rounds', type=int, required=True, help='Number of iteration rounds')
    parser.add_argument('--top-k', type=int, required=True, help='Select top-k samples based on metrics each time')
    parser.add_argument('--exp-dir', type=str, required=True, help='Experiment directory')
    parser.add_argument('--eval-filter', type=str, required=False, help='Evaluation task filters, multiple filters separated by commas')
    parser.add_argument('--eval-task-weights', type=str, required=True, help='Evaluation task weights, multiple weights separated by commas')
    parser.add_argument('--esm-fold-gpus', type=int, default=1, help='Number of ESM-Fold GPUs for parallel computation')
    parser.add_argument('--reaction', type=str, required=False, help='Reaction equation')
    parser.add_argument('--sort-by', type=str, required=False, default="weighted_score", help='Specify multi-objective sorting scheme')
    parser.add_argument('--target-temp', type=float, required=False, default=0.0, help='Target temperature')
    parser.add_argument('--target-ph', type=float, required=False, default=0.0, help='Target pH')
    # Generation parameters
    parser.add_argument('--ckpt-file', type=str, required=True, help='Model checkpoint file')
    parser.add_argument('--infer-step', type=int, required=True, help='Inference steps')
    parser.add_argument('--beta1', type=float, required=True, help='Beta1 parameter')
    parser.add_argument('--beta-time-order', type=float, required=True, help='Beta time order parameter')
    parser.add_argument('--init-t', type=int, required=True, help='Initial t value')
    parser.add_argument('--mbcltbf', type=int, required=True, help='MBCLTBF parameter')
    parser.add_argument('--num-seqs', type=int, required=True, help='Number of sequences')
    parser.add_argument('--batch-size', type=int, required=True, help='Batch size')
    parser.add_argument('--infer-type', type=str, required=True, help='Inference type')
    parser.add_argument('--target-ec', type=str, required=False, help='Target EC number')
    parser.add_argument('--mutation-ratio', type=float, required=False, default=0.1, help='Mutation ratio')

    args = parser.parse_args()
    
    # Create experiment directory
    os.makedirs(args.exp_dir, exist_ok=True)

    # Initialize logger
    global logger
    logger = setup_logger(args.exp_dir)
    
    # Parse Evaluation Task Weights
    # eval-task-weights are comma-separated metrics and weights, e.g., pLDDT:1,progen_nll:-1
    # Positive weights mean higher is better, negative weights mean lower is better
    # {"progen_nll": -1.0, "pLDDT": 1.0}
    eval_weights = {}
    if args.eval_task_weights:
        weight_strings = args.eval_task_weights.split(',')
        for weight_str in weight_strings:
            if ':' in weight_str:
                metric, weight = weight_str.split(':')
                eval_weights[metric.strip()] = float(weight.strip())
            else:
                logger.warning(f"Unable to parse weight condition: {weight_str}, will be ignored")
        
        logger.info(f"Parsed weights: {eval_weights}")
    
    if not eval_weights:
        raise ValueError("Must provide at least one evaluation metric weight (--eval-task-weights)")
    
    # Parse Evaluation Filters
    # eval-filter are comma-separated metrics and thresholds, e.g., pLDDT>0.5,pTM<0.5
    # {"pLDDT": {"operator": ">", "threshold": 90}}
    eval_filter = {}
    if args.eval_filter:
        filter_strings = args.eval_filter.split(',')
        for filter_str in filter_strings:
            if '>=' in filter_str:
                metric, threshold = filter_str.split('>=')
                eval_filter[metric.strip()] = {"operator": ">=", "threshold": float(threshold.strip())}
            elif '>' in filter_str:
                metric, threshold = filter_str.split('>')
                eval_filter[metric.strip()] = {"operator": ">", "threshold": float(threshold.strip())}
            elif '<=' in filter_str:
                metric, threshold = filter_str.split('<=')
                eval_filter[metric.strip()] = {"operator": "<=", "threshold": float(threshold.strip())}
            elif '<' in filter_str:
                metric, threshold = filter_str.split('<')
                eval_filter[metric.strip()] = {"operator": "<", "threshold": float(threshold.strip())}            
            elif '==' in filter_str:
                metric, threshold = filter_str.split('==')
                eval_filter[metric.strip()] = {"operator": "==", "threshold": float(threshold.strip())}
            elif '!=' in filter_str:
                metric, threshold = filter_str.split('!=')
                eval_filter[metric.strip()] = {"operator": "!=", "threshold": float(threshold.strip())}
            else:
                logger.warning(f"Unable to parse filter condition: {filter_str}, will be ignored")
        
        logger.info(f"Parsed filter conditions: {eval_filter}")

    # Create JSON data from initial A3M files
    json_file = os.path.join(args.exp_dir, 'data.json')
    create_initial_json(args.init_data, json_file, eval_task="weighted_score")

    # Form list of all evaluation metrics to be tested
    # ["progen_nll", "pLDDT", "pTM"]
    eval_tasks = []
    
    # Add metrics from filter conditions
    for metric in eval_filter.keys():
        if metric not in eval_tasks:
            eval_tasks.append(metric)
    
    # Add metrics from weights
    for metric in eval_weights.keys():
        if metric not in eval_tasks:
            eval_tasks.append(metric)
    
    logger.info(f"All evaluation metrics: {eval_tasks}")

    # Save configuration file
    with open(os.path.join(args.exp_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    generation_params = {
        'ckpt_file': args.ckpt_file,
        'infer_step': args.infer_step,
        'beta1': args.beta1,
        'beta_time_order': args.beta_time_order,
        'init_t': args.init_t,
        'mbcltbf': args.mbcltbf,
        'num_seqs': args.num_seqs,
        'batch_size': args.batch_size,
        'infer_type': args.infer_type,
        'mutation_ratio': args.mutation_ratio
    }
    
    # Start iterative generation-evaluation process
    for round_num in range(args.rounds):
        logger.info(f"\n=== Starting round {round_num} generation and evaluation ===")
        
        # 1. Generate new sequences for all samples
        logger.info(f"Generating round {round_num} sequences for all samples...")
        run_generation_json(json_file, round_num, round_num, generation_params)
        
        # 2. Evaluate new sequences for all samples
        logger.info(f"Evaluating round {round_num} sequences for all samples...")
        # Pass eval_tasks list for multi-metric evaluation
        data = run_batch_evaluation(json_file, round_num, eval_tasks, args.exp_dir, args.esm_fold_gpus, args)
        
        # 3. Filter top-k sequences for each sample and save to separate JSON files
        logger.info(f"Filtering round {round_num} sequences for all samples...")
        for sample_idx in range(len(data)):
            # Pass eval_filter and eval_weights for multi-metric filtering and sorting
            next_round_idx, _ = filter_top_k_samples_json(
                json_file, 
                sample_idx, 
                args.filter_window, 
                args.top_k, 
                round_num,
                eval_filter,
                eval_weights,
                args.sort_by
            )
        
        # Save filtered sequences to separate JSON files
        save_filtered_sequences_to_json(json_file, round_num + 1, args.exp_dir)
    
    # 4. Update: Plot evaluation results using new plotting functions
    logger.info("\n=== Plotting evaluation results ===")
    
    # Create plot output directory
    plot_dir = os.path.join(args.exp_dir, "plot")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Load filtered data from all rounds
    filtered_data_all_rounds = load_all_filtered_data(args.exp_dir, args.rounds)
    
    # Plot curves for each sample
    logger.info("Plotting evaluation curves for each sample...")
    plot_all_samples_curves_filtered(args, filtered_data_all_rounds, plot_dir, eval_tasks)
    
    # Plot average curves for all samples
    logger.info("Plotting average evaluation curves for all samples...")
    plot_samples_average_filtered(args, filtered_data_all_rounds, plot_dir, eval_tasks)
    
    # Perform additional pLDDT evaluation on final round filtered samples
    final_round_idx = args.rounds  # Final round index
    logger.info(f"\n=== Performing additional pLDDT evaluation on final round (round {final_round_idx}) filtered samples ===")
    evaluate_final_round_samples(final_round_idx, args.exp_dir, args.esm_fold_gpus, args)
    
    # If detailed plotting for individual samples is needed, uncomment below and provide sample_id
    # Example: Plot detailed data for sample_1
    # logger.info("Plotting detailed evaluation curves for individual sample...")
    # plot_evaluation_results_filtered(args, sample_id="sample_1", filtered_data_all_rounds=filtered_data_all_rounds, plot_dir=plot_dir, eval_weights=eval_weights)

if __name__ == "__main__":
    main()