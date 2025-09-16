#!/bin/bash

# Parallel version - runs multiple jobs simultaneously
# WARNING: This will use more system resources (CPU/GPU memory)

echo "Starting parallel batch processing of get_affordance.py..."
echo "Total combinations: 24 (8 scenes x 3 recordings)"

# Maximum number of parallel jobs (adjust based on your system capacity)
MAX_JOBS=8

# Function to run a single combination
run_combination() {
    local scene_idx=$1
    local recording_idx=$2
    
    echo "Starting scene_idx=$scene_idx, recording_idx=$recording_idx"
    
    python get_affordance.py \
        --school song \
        --scene_idx $scene_idx \
        --recording_idx $recording_idx
    
    if [ $? -eq 0 ]; then
        echo "✓ Completed scene_idx=$scene_idx, recording_idx=$recording_idx"
    else
        echo "✗ Failed scene_idx=$scene_idx, recording_idx=$recording_idx"
    fi
}

# Export the function so it's available to parallel processes
export -f run_combination

# Create array of all combinations
combinations=()
for scene_idx in {1..6}; do
    for recording_idx in {1..4}; do
        combinations+=("$scene_idx $recording_idx")
    done
done

# Run combinations in parallel using xargs
printf '%s\n' "${combinations[@]}" | xargs -n 2 -P $MAX_JOBS -I {} bash -c 'run_combination {}'

echo "Parallel batch processing completed!"