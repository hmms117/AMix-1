import os
import subprocess
import logging
import threading
import re
from math import ceil
from utils import parse_fasta, write_fasta
from Bio import pairwise2
from itertools import combinations

logger = logging.getLogger(__name__)

ESM_MODEL_PATH = ''   # Set your ESMFold model path here, or leave empty to automatically download
TMALIGN_EXEC_PATH = './evaluation/TMalign/TMalign'  # Set your TMalign executable path here


def compute_identity(seq1, seq2):
    """Calculate global alignment identity between two sequences"""
    alignments = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)
    best_alignment = alignments[0]
    matches = best_alignment[2]
    aln_len = best_alignment[4] - best_alignment[3]
    return matches / aln_len if aln_len > 0 else 0.0


def evaluate_novelty(eval_file, args=None):
    """
    Calculate the lowest identity between each designed sequence and its corresponding original sequence as a measure of novelty.
    Original sequences are found from init_data directory using original_seq_id to locate corresponding fasta files.
    """
    logger.info("=== Starting novelty evaluation ===")
    
    if args is None or not hasattr(args, "init_data"):
        ref_dir = "/mnt/workspace/zhoujiang/data/CAMEO_20/CAMEO_20"
        logger.warning("Missing init_data argument, using default directory: %s", ref_dir)
    else:   
        ref_dir = args.init_data

    abs_eval_file = os.path.abspath(eval_file)
    designed_seqs = parse_fasta(abs_eval_file)
    

    novelty_scores = {}

    for seq_record in designed_seqs:
        seq_id = seq_record['id']
        seq = seq_record['seq']

        # Extract original_seq_id
        if "_seq" in seq_id:
            original_seq_id = seq_id.split("_seq")[0]
        else:
            original_seq_id = seq_id

        ref_fasta_path = os.path.join(ref_dir, f"{original_seq_id}.fasta")
        ref_a3m_path = os.path.join(ref_dir, f"{original_seq_id}.a3m")
        
        if os.path.exists(ref_fasta_path):
            # Load FASTA format reference sequences
            ref_sequences = parse_fasta(ref_fasta_path)
        elif os.path.exists(ref_a3m_path):
            # If FASTA doesn't exist but A3M does, load A3M format reference sequences
            ref_sequences = parse_fasta(ref_a3m_path)
        else:
            logger.warning("Reference sequence file not found: %s or %s", ref_fasta_path, ref_a3m_path)
            novelty_scores[seq_id] = 0.0
            seq_record['annotations']["novelty"] = "0.0"
            continue

        # Calculate identity
        identities = [compute_identity(seq, ref_seq['seq']) for ref_seq in ref_sequences]
        min_identity = min(identities) if identities else 0.0
        novelty = 1.0 - min_identity

        novelty_scores[seq_id] = novelty
        seq_record['annotations']["novelty"] = str(novelty)

    write_fasta(designed_seqs, abs_eval_file)
    logger.info("=== Novelty evaluation completed ===")
    return novelty_scores


def evaluate_diversity(eval_file, args=None):
    """
    Calculate the average identity between all pairs of sequences within each sample as a measure of diversity.
    """
    logger.info("=== Starting diversity evaluation ===")
    
    abs_eval_file = os.path.abspath(eval_file)
    sequences = parse_fasta(abs_eval_file)

    # Group by sample_id -> [seqs]
    sample_groups = {}
    for seq in sequences:
        seq_id = seq['id']
        sample_id = seq_id.split("_seq")[0] if "_seq" in seq_id else seq_id
        sample_groups.setdefault(sample_id, []).append(seq)

    diversity_scores = {}
    for sample_id, seqs in sample_groups.items():
        if len(seqs) < 2:
            diversity_scores[sample_id] = 0.0
            continue
        
        pair_identities = [
            compute_identity(s1['seq'], s2['seq'])
            for s1, s2 in combinations(seqs, 2)
        ]
        mean_identity = sum(pair_identities) / len(pair_identities)
        diversity = 1.0 - mean_identity
        diversity_scores[sample_id] = diversity

        # Also write to each sequence (optional)
        for s in seqs:
            s['annotations']["diversity"] = str(diversity)

    write_fasta(sequences, abs_eval_file)
    logger.info("=== Diversity evaluation completed ===")
    return diversity_scores


def evaluate_progen_nll(eval_file, script_dir=None, args=None):
    # TODO: Implement ProGen negative log-likelihood evaluation
    # Store scores in seq_record['annotations']['progen_nll'] for each sequence
    pass

def evaluate_esmfold(eval_file, script_dir="./evaluation/", esm_fold_gpus=1, args=None):
    """
    Run ESM-Fold to predict structure from sequences, with support for multi-GPU processing.
    Output includes pLDDT and pTM.
    
    Args:
        eval_file: Path to FASTA file containing sequences
        script_dir: Directory containing evaluation scripts
        esm_fold_gpus: Number of GPUs to use for ESM-Fold
    
    Returns:
        Evaluation output
    """
    logger.info("=== Starting ESM-Fold evaluation (GPUs: %d) ===", esm_fold_gpus)
    
    if script_dir is None:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    abs_eval_file = os.path.abspath(eval_file)
    
    if esm_fold_gpus > 1:
        sequences = parse_fasta(abs_eval_file)
        total_seqs = len(sequences)
        seqs_per_worker = ceil(total_seqs / esm_fold_gpus)
        parent_dir = os.path.dirname(abs_eval_file)
        base_filename = os.path.basename(abs_eval_file).split('.')[0]
        split_files = []
        for i in range(esm_fold_gpus):
            start_idx = i * seqs_per_worker
            end_idx = min((i + 1) * seqs_per_worker, total_seqs)
            if start_idx >= total_seqs:
                break
            worker_seqs = sequences[start_idx:end_idx]
            split_file = os.path.join(parent_dir, f"{base_filename}_split_{i}.fasta")
            write_fasta(worker_seqs, split_file)
            split_files.append(split_file)
        logger.info("Split %d sequences into %d files for multi-GPU processing", total_seqs, len(split_files))
        
        def process_file(file_path, gpu_id):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            subprocess.run([
                "python", os.path.join(script_dir, "fold_eval_tts.py"),
                "-i", file_path,
                "-o", os.path.dirname(file_path),
                "-m", ESM_MODEL_PATH,
                "--max-tokens-per-batch", "2000"
            ], check=True, env=env)
        
        threads = []
        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(",")
        start_gpu_id = int(cuda_devices[0])
        for i, file_path in enumerate(split_files):
            gpu_id = start_gpu_id + i
            thread = threading.Thread(target=process_file, args=(file_path, gpu_id))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        logger.info("Merging results from %d split files", len(split_files))
        all_sequences = []
        for file_path in split_files:
            all_sequences.extend(parse_fasta(file_path))
        write_fasta(all_sequences, abs_eval_file)
        logger.info("=== ESM-Fold evaluation completed ===")
        return "ESM-Fold multi-GPU evaluation completed"
    else:
        subprocess.run([
            "python", os.path.join(script_dir, "fold_eval_tts.py"),
            "-i", abs_eval_file,
            "-o", os.path.dirname(abs_eval_file),
            "-m", ESM_MODEL_PATH,
            "--max-tokens-per-batch", "2000"
        ], check=True)
        logger.info("=== ESM-Fold evaluation completed ===")
        return "ESM-Fold evaluation completed"

def TMalign(file1, file2, exec_path=TMALIGN_EXEC_PATH):
    """
    Run TMalign to align two PDB structures and parse the result.

    Args:
        file1: Path to the first PDB file (aligned to file2)
        file2: Path to the second PDB file
        exec_path: Path to TMalign executable (default: TMALIGN_EXEC_PATH)

    Returns:
        A dictionary with the following keys:
            'aligned_length': Number of aligned residues (int)
            'RMSD': Root-mean-square deviation (float)
            'TM_score': TM-score value (float)

    Raises:
        RuntimeError: If TMalign execution fails
        ValueError: If alignment results cannot be parsed
    """
    cmd = [exec_path, file1, file2]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Error running TMalign: " + e.stderr)

    output = result.stdout

    # Parse aligned length and RMSD using regex
    parsed_result = {}
    match = re.search(r"Aligned length=\s*(\d+),\s*RMSD=\s*([\d.]+)", output)
    if match:
        parsed_result['aligned_length'] = int(match.group(1))
        parsed_result['RMSD'] = float(match.group(2))
    else:
        raise ValueError("Failed to parse aligned length and RMSD")

    # Parse the first TM-score value
    match_tm = re.search(r"TM-score=\s*([\d.]+)", output)
    if match_tm:
        parsed_result['TM_score'] = float(match_tm.group(1))
    else:
        raise ValueError("Failed to parse TM-score")

    return parsed_result


def evaluate_tm_score(eval_file, original_pdb_dir=None, args=None):
    """
    Evaluate sequences using TM-align by comparing original PDB structures 
    with ESM-Fold generated structures
    
    Args:
        eval_file: Path to FASTA file containing sequences
        original_pdb_dir: Directory containing original PDB files
    
    Returns:
        TM-align results as a dictionary mapping sequence IDs to TM scores
    """
    logger.info("=== Starting TM-score evaluation ===")
    
    if original_pdb_dir is None:
        logger.warning("No original PDB directory provided, cannot calculate TM-score")
        return {}
    
    abs_eval_file = os.path.abspath(eval_file)
    parent_dir = os.path.dirname(abs_eval_file)
    sequences = parse_fasta(abs_eval_file)
    
    # Dictionary to store TM scores for each sequence
    tm_scores = {}
    
    for seq_record in sequences:
        seq_id = seq_record['id']
        # Extract original sequence ID from the sequence ID in eval_file
        # Format is {original_seq_id}_seq{seq_idx}
        if "_seq" in seq_id:
            original_seq_id = seq_id.split("_seq")[0]
            seq_idx = seq_id.split("_seq")[1]
        else:
            original_seq_id = seq_id
            seq_idx = ""
        
        # Paths to original and generated PDB files
        original_pdb_path = os.path.join(original_pdb_dir, f"{original_seq_id}.pdb")
        generated_pdb_path = os.path.join(parent_dir, f"{seq_id}.pdb")
        
        # Check if both PDB files exist
        if not os.path.exists(original_pdb_path):
            logger.warning("Original PDB file not found: %s", original_pdb_path)
            continue
        if not os.path.exists(generated_pdb_path):
            logger.warning("Generated PDB file not found: %s", generated_pdb_path)
            continue
        
        # Calculate TM-score using TMalign
        result = TMalign(original_pdb_path, generated_pdb_path)
        tm_scores[seq_id] = result
        
        # Add score to sequence description in the FASTA file
        tm_value = result.get("TM_score", "N/A")
        rmsd_value = result.get("RMSD", "N/A")
        
        # Update annotations in the sequence record
        seq_record['annotations']["TM_score"] = str(tm_value)
        seq_record['annotations']["RMSD"] = str(rmsd_value)
    
    # Update the FASTA file with new descriptions containing TM scores
    write_fasta(sequences, abs_eval_file)
    
    logger.info("=== TM-score evaluation completed ===")
    return tm_scores

def evaluate_rosetta_energy(eval_file, script_dir=None, esm_fold_gpus=1, args=None):
    """
    Evaluate sequences using Rosetta Energy (requires ESM-Fold first)
    
    Args:
        eval_file: Path to FASTA file containing sequences
        script_dir: Directory containing evaluation scripts
        esm_fold_gpus: Number of GPUs to use for ESM-Fold
    
    Returns:
        Rosetta Energy evaluation results
    """
    logger.info("=== Starting Rosetta Energy evaluation (GPUs: %d) ===", esm_fold_gpus)
    
    if script_dir is None:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    abs_eval_file = os.path.abspath(eval_file)
    
    if esm_fold_gpus > 1:
        # Multi-GPU processing
        # Read FASTA file content
        sequences = parse_fasta(abs_eval_file)
        total_seqs = len(sequences)
        seqs_per_worker = ceil(total_seqs / esm_fold_gpus)
        
        # Get original file's directory
        parent_dir = os.path.dirname(abs_eval_file)
        base_filename = os.path.basename(abs_eval_file).split('.')[0]
        
        # Split sequences into multiple files
        split_files = []
        for i in range(esm_fold_gpus):
            start_idx = i * seqs_per_worker
            end_idx = min((i + 1) * seqs_per_worker, total_seqs)
            if start_idx >= total_seqs:
                break
            
            worker_seqs = sequences[start_idx:end_idx]
            split_file = os.path.join(parent_dir, f"{base_filename}_split_{i}.fasta")
            write_fasta(worker_seqs, split_file)
            split_files.append(split_file)
        
        logger.info("Split %d sequences into %d files for multi-GPU processing", total_seqs, len(split_files))
        
        # Define thread function
        def process_file(file_path, gpu_id):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            # Run ESM-Fold to generate PDB files
            subprocess.run([
                "python", os.path.join(script_dir, "fold_eval_tts.py"),
                "-i", file_path,
                "-o", os.path.dirname(file_path),
                "-m", ESM_MODEL_PATH,
                "--max-tokens-per-batch", "2000"
            ], check=True, env=env)
            
            # Use subprocess to calculate Rosetta Energy
            subprocess.run([
                "python", "-c", 
                f"from evaluation.rosetta_energy import eval_from_fasta_pdb; eval_from_fasta_pdb('{file_path}')"
            ], check=True)
        
        # Create and start threads
        threads = []
        # Get current environment's CUDA_VISIBLE_DEVICES
        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')
        start_gpu_id = int(cuda_devices[0])  # Start from first visible GPU
        
        for i, file_path in enumerate(split_files):
            # GPU ID increases from the first visible GPU
            gpu_id = start_gpu_id + i
            thread = threading.Thread(
                target=process_file, 
                args=(file_path, gpu_id)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Merge all split file results back to original file
        logger.info("Merging results from %d split files", len(split_files))
        all_sequences = []
        for file_path in split_files:
            all_sequences.extend(parse_fasta(file_path))
        
        write_fasta(all_sequences, abs_eval_file)
        
        # Optional: Clean up split files
        # for file_path in split_files:
        #     os.remove(file_path)
    else:
        # Single process mode, directly call ESM-Fold
        subprocess.run([
            "python", os.path.join(script_dir, "fold_eval_tts.py"),
            "-i", abs_eval_file,
            "-o", os.path.dirname(abs_eval_file),
            "-m", ESM_MODEL_PATH,
            "--max-tokens-per-batch", "2000"
        ], check=True)
        
        # Calculate Rosetta Energy using subprocess
        subprocess.run([
            "python", "-c", 
            f"from evaluation.rosetta_energy import eval_from_fasta_pdb; eval_from_fasta_pdb('{abs_eval_file}')"
        ], check=True)
    
    logger.info("=== Rosetta Energy evaluation completed ===")
    return "Rosetta Energy evaluation completed"

def evaluate_repeatness(eval_file, args=None):
    """
    Evaluate the repeatness of sequences, measuring consecutive repeated amino acids.
    For each repeated subsequence, if length > 4, add (length - 4) to repeatness score.
    
    Args:
        eval_file: Path to FASTA file containing sequences
    
    Returns:
        Dictionary mapping sequence IDs to repeatness scores
    """
    logger.info("=== Starting repeatness evaluation ===")
    
    abs_eval_file = os.path.abspath(eval_file)
    sequences = parse_fasta(abs_eval_file)
    
    # Dictionary to store repeatness scores
    repeatness_scores = {}
    
    for seq_record in sequences:
        seq_id = seq_record['id']
        seq = seq_record['seq']
        
        # Calculate repeatness score
        score = 0
        
        # Iterate through sequence to find runs of consecutive repeated characters
        i = 0
        while i < len(seq):
            char = seq[i]
            run_length = 1
            
            # Count consecutive occurrences of the same character
            j = i + 1
            while j < len(seq) and seq[j] == char:
                run_length += 1
                j += 1
            
            # If run length > 4, add to score
            if run_length > 4:
                score += (run_length - 4)
            
            # Move to the next run
            i = j
        
        # Store score in dictionary
        repeatness_scores[seq_id] = score
        
        # Add score to sequence annotations
        seq_record['annotations']["repeatness"] = str(score)
    
    # Update the FASTA file with new annotations
    write_fasta(sequences, abs_eval_file)
    
    logger.info("=== Repeatness evaluation completed ===")
    return repeatness_scores

def evaluate_clean(eval_file, tgt_ec, args=None):
    """
    Evaluate sequences using CLEAN model for distance or confidence to specific EC number
    
    Args:
        eval_file: Path to FASTA file containing sequences to evaluate
        tgt_ec: Target EC number (e.g., "1.1.1.1")
        
    Returns:
        No return value, results are added as CLEAN_EC annotations directly to the input FASTA file
    """
    logger.info("=== Starting CLEAN EC evaluation (target: %s) ===", tgt_ec)
    
    # Get CLEAN application directory path
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clean_app_dir = os.path.join(script_dir, "evaluation", "CLEAN", "app")
    
    # Check if FASTA file exists
    if not os.path.exists(eval_file):
        logger.error("FASTA file not found: %s", eval_file)
        return
    
    # Build command to call CLEAN_inference.py
    abs_eval_file = os.path.abspath(eval_file)
    
    # Set input FASTA file directory and filename
    fasta_folder = os.path.dirname(abs_eval_file)
    fasta_filename = os.path.basename(abs_eval_file)
    
    try:
        # Execute CLEAN_inference.py with necessary parameters, others use defaults
        # Set working directory to CLEAN application directory to ensure relative path references can be found
        subprocess.run([
            "python", os.path.join(clean_app_dir, "CLEAN_inference.py"),
            "--inference_fasta_folder", fasta_folder,
            "--inference_fasta", fasta_filename,
            "--gpu_id", os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
            "--target_ec", tgt_ec,
            "--use_confidence",  # Default to use confidence instead of distance
            "--output_annotated_fasta", abs_eval_file  # Directly overwrite original file
        ], check=True, cwd=clean_app_dir)  # Set working directory to CLEAN application directory
        
    except subprocess.CalledProcessError as e:
        logger.error("Error executing CLEAN_inference.py: %s", e)
    except Exception as e:
        logger.error("Unknown error during CLEAN EC evaluation: %s", e)
    
    clean_ec_score = {}
    sequences = parse_fasta(abs_eval_file)
    for seq_record in sequences:
        seq_id = seq_record['id']
        clean_ec_score[seq_id] = {
            "CLEAN_EC": float(seq_record["annotations"]["CLEAN_EC"])
        }
    logger.info("=== CLEAN EC evaluation completed ===")
    return clean_ec_score

def evaluate_clean_distance(eval_file, tgt_ec, args=None):
    """
    Evaluate sequences using CLEAN model for distance to specific EC number
    
    Args:
        eval_file: Path to FASTA file containing sequences to evaluate
        tgt_ec: Target EC number (e.g., "1.1.1.1")
        
    Returns:
        No return value, results are added as CLEAN_EC annotations directly to the input FASTA file
    """
    logger.info("=== Starting CLEAN Distance evaluation (target: %s) ===", tgt_ec)
    
    # Get CLEAN application directory path
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clean_app_dir = os.path.join(script_dir, "evaluation", "CLEAN", "app")
    
    # Check if FASTA file exists
    if not os.path.exists(eval_file):
        logger.error("FASTA file not found: %s", eval_file)
        return
    
    # Build command to call CLEAN_inference.py
    abs_eval_file = os.path.abspath(eval_file)
    
    # Set input FASTA file directory and filename
    fasta_folder = os.path.dirname(abs_eval_file)
    fasta_filename = os.path.basename(abs_eval_file)
    
    try:
        # Execute CLEAN_inference.py with necessary parameters, others use defaults
        # Set working directory to CLEAN application directory to ensure relative path references can be found
        subprocess.run([
            "python", os.path.join(clean_app_dir, "CLEAN_inference.py"),
            "--inference_fasta_folder", fasta_folder,
            "--inference_fasta", fasta_filename,
            "--gpu_id", os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
            "--target_ec", tgt_ec,
            "--no_use_confidence",  # Use distance
            "--output_annotated_fasta", abs_eval_file  # Directly overwrite original file
        ], check=True, cwd=clean_app_dir)  # Set working directory to CLEAN application directory
        
    except subprocess.CalledProcessError as e:
        logger.error("Error executing CLEAN_inference.py: %s", e)
    except Exception as e:
        logger.error("Unknown error during CLEAN EC evaluation: %s", e)
    
    logger.info("=== CLEAN Distance evaluation completed ===")
    return

def evaluate_identical(eval_file, args=None):
    """
    Calculate the highest identity between each designed sequence and its corresponding original sequence as an identical metric.
    Original sequences are found from init_data directory using original_seq_id to locate corresponding fasta files.
    """
    logger.info("=== Starting identical evaluation ===")
    
    if args is None or not hasattr(args, "init_data"):
        logger.warning("Missing init_data argument to calculate identical score, exiting...")
        exit()
    else:   
        ref_dir = args.init_data

    abs_eval_file = os.path.abspath(eval_file)
    designed_seqs = parse_fasta(abs_eval_file)
    
    identical_scores = {}

    for seq_record in designed_seqs:
        seq_id = seq_record['id']
        seq = seq_record['seq']

        # Extract original_seq_id
        if "_seq" in seq_id:
            original_seq_id = seq_id.split("_seq")[0]
        else:
            original_seq_id = seq_id

        ref_fasta_path = os.path.join(ref_dir, f"{original_seq_id}.fasta")
        ref_a3m_path = os.path.join(ref_dir, f"{original_seq_id}.a3m")
        
        if os.path.exists(ref_fasta_path):
            # Load FASTA format reference sequences
            ref_sequences = parse_fasta(ref_fasta_path)
        elif os.path.exists(ref_a3m_path):
            # If FASTA doesn't exist but A3M does, load A3M format reference sequences
            ref_sequences = parse_fasta(ref_a3m_path)
        else:
            logger.warning("Reference sequence file not found: %s or %s", ref_fasta_path, ref_a3m_path)
            identical_scores[seq_id] = 0.0
            seq_record['annotations']["identical"] = "0.0"
            continue

        # Calculate identity
        identities = [compute_identity(seq, ref_seq['seq']) for ref_seq in ref_sequences]
        max_identity = max(identities) if identities else 0.0

        identical_scores[seq_id] = max_identity
        seq_record['annotations']["identical"] = str(max_identity)

    write_fasta(designed_seqs, abs_eval_file)
    logger.info("=== Identical evaluation completed ===")
    return identical_scores

def evaluate_clipzyme(eval_file, args):
    # TODO: Implement ClipZyme evaluation
    # Store scores in seq_record['annotations']['clipzyme'] for each sequence
    pass

def evaluate_seq2phopt(eval_file, args=None):
    # TODO: Implement Seq2PhOpt evaluation for pH optimization
    # Store scores in seq_record['annotations']['seq2phopt'] for each sequence
    pass

def evaluate_seq2topt(eval_file, args=None):
    # TODO: Implement Seq2TOpt evaluation for temperature optimization
    # Store scores in seq_record['annotations']['seq2topt'] for each sequence
    pass


# Dictionary mapping evaluation task names to their corresponding functions
METRICS = {
    "progen_nll": evaluate_progen_nll,
    "TM_score": evaluate_tm_score,
    "rosetta_energy": evaluate_rosetta_energy,
    "repeatness": evaluate_repeatness,
    "novelty": evaluate_novelty,
    "diversity": evaluate_diversity,
    "CLEAN_EC": evaluate_clean,
    "identical": evaluate_identical,
    "clipzyme": evaluate_clipzyme,  # Add clipzyme to METRICS dictionary
    "CLEAN_Distance": evaluate_clean_distance,
    "seq2phopt": evaluate_seq2phopt,  # Add seq2phopt to METRICS dictionary
    "seq2topt": evaluate_seq2topt,  # Add seq2topt to METRICS dictionary
}


def get_metric_function(eval_task):
    """
    Get the evaluation function for a specific task
    
    Args:
        eval_task: Name of the evaluation task
    
    Returns:
        The corresponding evaluation function
    
    Raises:
        NotImplementedError: If the task is not supported
    """
    if eval_task in METRICS:
        return METRICS[eval_task]
    else:
        raise NotImplementedError(f"Unknown evaluation task: {eval_task}")

def verifier(eval_file, eval_tasks, esm_fold_gpus=1, args=None):
    """
    Run multiple evaluation tasks with optimized calls to avoid redundant computation
    
    Logic:
      1. If rosetta_energy is needed, run rosetta_energy directly (which includes ESM-Fold internally),
         no need to separately evaluate pLDDT/pTM.
      2. If rosetta_energy is not needed but pLDDT or pTM is needed, call ESM-Fold once to get both results.
      3. If TM_score or RMSD is needed, call TM-align once to get both results.
      4. For other tasks, call evaluation functions individually.
    
    Args:
        eval_file: Path to FASTA file
        eval_tasks: List of evaluation tasks or single evaluation task
        esm_fold_gpus: Number of GPUs for ESM-Fold
        args: Other parameters, including init_data pointing to original PDB file directory
        
    Returns:
        A dictionary containing results for each evaluation task
    """
    if isinstance(eval_tasks, str):
        eval_tasks = [eval_tasks]
    
    logger.info("Starting evaluation pipeline with tasks: %s", eval_tasks)
    
    completed_tasks = set()
    results = {}
    
    # If rosetta_energy is requested, run it directly
    if "rosetta_energy" in eval_tasks:
        results["rosetta_energy"] = evaluate_rosetta_energy(eval_file, esm_fold_gpus=esm_fold_gpus)
        completed_tasks.add("rosetta_energy")
        if "pLDDT" in eval_tasks:
            completed_tasks.add("pLDDT")
        if "pTM" in eval_tasks:
            completed_tasks.add("pTM")
    else:
        # If rosetta_energy is not requested but pLDDT or pTM is needed, call ESM-Fold
        if any(task in eval_tasks for task in ["pLDDT", "pTM", "TM_score", "RMSD", "clipzyme"]):
            results["esm_fold"] = evaluate_esmfold(eval_file, esm_fold_gpus=esm_fold_gpus)
            if "pLDDT" in eval_tasks:
                completed_tasks.add("pLDDT")
            if "pTM" in eval_tasks:
                completed_tasks.add("pTM")
    
    # Handle TM-align related tasks
    # Get original PDB files from args.init_data and compare with ESM-generated PDB files
    if any(task in eval_tasks for task in ["TM_score", "RMSD"]):
        original_pdb_dir = args.init_data if args and hasattr(args, 'init_data') else None
        results["TM_score"] = evaluate_tm_score(eval_file, original_pdb_dir)
        if "TM_score" in eval_tasks:
            completed_tasks.add("TM_score")
        if "RMSD" in eval_tasks:
            completed_tasks.add("RMSD")

    # Handle CLEAN related tasks
    if "CLEAN_EC" in eval_tasks:
        results["CLEAN_EC"] = evaluate_clean(eval_file, tgt_ec=args.target_ec)
        completed_tasks.add("CLEAN_EC")

    if "CLEAN_Distance" in eval_tasks:
        results["clean"] = evaluate_clean_distance(eval_file, tgt_ec=args.target_ec)
        completed_tasks.add("CLEAN_Distance")
    
    # Run other independent evaluation tasks
    for task in eval_tasks:
        if task not in completed_tasks:
            logger.info("Running %s evaluation", task)
            func = get_metric_function(task)
            results[task] = func(eval_file, args=args)
            completed_tasks.add(task)
    
    logger.info("Evaluation pipeline completed for tasks: %s", list(completed_tasks))
    return results
