#!/bin/bash

###### TTS-MULTI-DEV LAUNCHER ######

# =============================================================================
# USER CONFIGURATION - PLEASE MODIFY THESE PATHS AND SETTINGS
# =============================================================================

# TODO: Set your model checkpoint file path
CKPT_FILE="./ckpt/AMix-1-1.7b.ckpt"  # e.g., "./ckpt/AMix-1-1.7b.ckpt"

# TODO: Set your initial data directory path  
INIT_DATA="/tos-bjml-ai4s-serving/users/zhoujiang/data/CASP14_orphan"  # e.g., "/path/to/your/data/CASP14_orphan"

# TODO: Set your experiment output base directory
EXP_BASE_DIR="/tos-bjml-ai4s-serving/users/zhoujiang/exp_release/test"  # e.g., "/path/to/your/experiments"

# =============================================================================
# DEFAULT PARAMETERS - MODIFY AS NEEDED
# =============================================================================

# Evaluation task weights - Format: "metric1:weight1,metric2:weight2"
# Positive weights mean higher is better, negative weights mean lower is better
# Example: pLDDT:1.0,rosetta_energy:-1.0
# Supported evaluation tasks:
# - pLDDT: Protein local structure quality score (higher is better)
# - pTM: Protein template modeling score (higher is better)  
# - rosetta_energy: Rosetta energy score (lower is better)
EVAL_TASK_WEIGHTS="pLDDT:1.0"

# Evaluation filter - Format: "metric1>value1,metric2<value2"
# Supports: >, <, >=, <=, ==, !=
# Example: pLDDT>80,progen_nll<10
EVAL_FILTER=""

# Iteration and generation parameters
ROUNDS=1
NUM_SEQS=5
INFER_STEP=10
SORT_BY="weighted_score"

# Model parameters
FILTER_WINDOW=$ROUNDS
TOP_K=5
BETA1=1.6
BETA_TIME_ORDER=1.0
INIT_T=99
MBCLTBF=1

# Inference parameters
BATCH_SIZE=600
INFER_TYPE="profile"
ESM_FOLD_GPUS=1

# Target parameters
TARGET_EC="2.3.1.37"
REACTION="[CH3:1][N+:2]([CH3:3])([CH3:4])[CH2:5][CH:6]=[O:7].[O:9]=[O:10].[OH2:8]>>[CH3:1][N+:2]([CH3:3])([CH3:4])[CH2:5][C:6](=[O:7])[OH:8].[OH:9][OH:10]"
TARGET_TEMP=30.0
TARGET_PH=5.0

# =============================================================================
# HELP FUNCTION
# =============================================================================

show_help() {
    cat << EOF
TTS-MULTI-DEV Launcher Script

USAGE:
    $0 [OPTIONS]

REQUIRED CONFIGURATION (modify in script):
    CKPT_FILE      Path to model checkpoint file
    INIT_DATA      Path to initial data directory
    EXP_BASE_DIR   Base directory for experiments

OPTIONS:
    --exp-dir DIR              Experiment output directory
    --filter-window NUM        Filter window size (number of rounds to consider)
    --init-data PATH           Initial data file path
    --rounds NUM               Number of iteration rounds
    --top-k NUM                Number of sequences to select per round
    --eval-task-weights STR    Evaluation task weights (metric:weight,...)
    --eval-filter STR          Evaluation filter (metric>value,...)
    --ckpt-file PATH           Model checkpoint file
    --infer-step NUM           Inference steps
    --beta1 FLOAT              Beta1 parameter (noise addition rate)
    --beta-time-order FLOAT    Beta time order parameter
    --init-t NUM               Initial time step
    --mbcltbf NUM              MBCLTBF parameter
    --num-seqs NUM             Number of sequences to generate
    --batch-size NUM           Batch size for processing
    --infer-type STR           Inference type (supports 'profile')
    --esm-fold-gpus NUM        Number of ESM-Fold GPUs
    --reaction STR             Reaction equation
    --mutation-ratio FLOAT     Mutation ratio
    --sort-by STR              Sorting method
    --target-ec STR            Target EC number
    --target-temp FLOAT        Target temperature
    --target-ph FLOAT          Target pH
    --help                     Show this help message

EXAMPLES:
    $0 --rounds 5 --num-seqs 10 --top-k 3
    $0 --eval-task-weights "pLDDT:1.0,pTM:0.5" --eval-filter "pLDDT>80"

EOF
}

# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

validate_config() {
    local errors=0
    
    if [ -z "$CKPT_FILE" ]; then
        echo "ERROR: CKPT_FILE is not set. Please configure the model checkpoint file path."
        errors=$((errors + 1))
    elif [ ! -f "$CKPT_FILE" ]; then
        echo "ERROR: Checkpoint file not found: $CKPT_FILE"
        errors=$((errors + 1))
    fi
    
    if [ -z "$INIT_DATA" ]; then
        echo "ERROR: INIT_DATA is not set. Please configure the initial data directory path."
        errors=$((errors + 1))
    elif [ ! -d "$INIT_DATA" ]; then
        echo "ERROR: Initial data directory not found: $INIT_DATA"
        errors=$((errors + 1))
    fi
    
    if [ -z "$EXP_BASE_DIR" ]; then
        echo "ERROR: EXP_BASE_DIR is not set. Please configure the experiment base directory."
        errors=$((errors + 1))
    fi
    
    if [ $errors -gt 0 ]; then
        echo ""
        echo "Please modify the script to set the required configuration variables."
        echo "Run '$0 --help' for more information."
        exit 1
    fi
}

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

parse_arguments() {
    while [[ "$#" -gt 0 ]]; do
        case $1 in
            --help) show_help; exit 0 ;;
            --exp-dir) EXP_DIR="$2"; shift ;;
            --filter-window) FILTER_WINDOW="$2"; shift ;;
            --init-data) INIT_DATA="$2"; shift ;;
            --rounds) ROUNDS="$2"; shift ;;
            --top-k) TOP_K="$2"; shift ;;
            --eval-task-weights) EVAL_TASK_WEIGHTS="$2"; shift ;;
            --eval-filter) EVAL_FILTER="$2"; shift ;;
            --ckpt-file) CKPT_FILE="$2"; shift ;;
            --infer-step) INFER_STEP="$2"; shift ;;
            --beta1) BETA1="$2"; shift ;;
            --beta-time-order) BETA_TIME_ORDER="$2"; shift ;;
            --init-t) INIT_T="$2"; shift ;;
            --mbcltbf) MBCLTBF="$2"; shift ;;
            --num-seqs) NUM_SEQS="$2"; shift ;;
            --batch-size) BATCH_SIZE="$2"; shift ;;
            --infer-type) INFER_TYPE="$2"; shift ;;
            --esm-fold-gpus) ESM_FOLD_GPUS="$2"; shift ;;
            --reaction) REACTION="$2"; shift ;;
            --mutation-ratio) MUTATION_RATIO="$2"; shift ;;
            --sort-by) SORT_BY="$2"; shift ;;
            --target-ec) TARGET_EC="$2"; shift ;;
            --target-temp) TARGET_TEMP="$2"; shift ;;
            --target-ph) TARGET_PH="$2"; shift ;;
            *) echo "Unknown parameter: $1"; echo "Run '$0 --help' for usage."; exit 1 ;;
        esac
        shift
    done
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    # Parse command line arguments
    parse_arguments "$@"
    
    # Validate configuration
    validate_config
    
    # Set filter window to rounds if not specified
    FILTER_WINDOW=$ROUNDS
    
    # Generate experiment directory if not specified
    if [ -z "$EXP_DIR" ]; then
        FIRST_METRIC=$(echo $EVAL_TASK_WEIGHTS | cut -d ',' -f1 | cut -d ':' -f1)
        EXP_DIR="${EXP_BASE_DIR}/${INFER_STEP}_${NUM_SEQS}_${ROUNDS}_${FIRST_METRIC}"
    fi
    
    # Display configuration
    echo "=========================================="
    echo "TTS-MULTI-DEV Configuration"
    echo "=========================================="
    echo "Experiment directory: $EXP_DIR"
    echo "Filter window: $FILTER_WINDOW"
    echo "Initial data: $INIT_DATA"
    echo "Number of sequences: $NUM_SEQS"
    echo "Iteration rounds: $ROUNDS"
    echo "Top-K: $TOP_K"
    echo "Evaluation task weights: $EVAL_TASK_WEIGHTS"
    echo "Evaluation filter: $EVAL_FILTER"
    echo "Model checkpoint: $CKPT_FILE"
    echo "Inference steps: $INFER_STEP"
    echo "Batch size: $BATCH_SIZE"
    echo "ESM-Fold GPUs: $ESM_FOLD_GPUS"
    echo "Sort by: $SORT_BY"
    echo "=========================================="
    
    # Create experiment directory
    mkdir -p "$EXP_DIR"
    
    # Build command
    CMD="python3 ./tts_EvoAMix-1.py \
        --exp-dir \"$EXP_DIR\" \
        --filter-window \"$FILTER_WINDOW\" \
        --init-data \"$INIT_DATA\" \
        --rounds \"$ROUNDS\" \
        --top-k \"$TOP_K\" \
        --eval-task-weights \"$EVAL_TASK_WEIGHTS\" \
        --ckpt-file \"$CKPT_FILE\" \
        --infer-step \"$INFER_STEP\" \
        --beta1 \"$BETA1\" \
        --beta-time-order \"$BETA_TIME_ORDER\" \
        --init-t \"$INIT_T\" \
        --mbcltbf \"$MBCLTBF\" \
        --num-seqs \"$NUM_SEQS\" \
        --batch-size \"$BATCH_SIZE\" \
        --infer-type \"$INFER_TYPE\" \
        --esm-fold-gpus \"$ESM_FOLD_GPUS\" \
        --target-ec \"$TARGET_EC\" \
        --reaction \"$REACTION\" \
        --target-temp \"$TARGET_TEMP\" \
        --target-ph \"$TARGET_PH\" \
        --sort-by \"$SORT_BY\""
    
    # Add evaluation filter if provided
    if [ ! -z "$EVAL_FILTER" ]; then
        CMD="$CMD --eval-filter \"$EVAL_FILTER\""
    fi
    
    # Execute command
    echo "Executing command:"
    echo "$CMD"
    echo ""
    
    eval $CMD
    
    echo ""
    echo "Experiment completed. Results saved in: $EXP_DIR"
}

# Run main function with all arguments
main "$@"