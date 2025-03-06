#!/bin/bash

# Initialize conda in the submission script
eval "$(/home/ensta/ensta-lachevre/miniconda3/bin/conda shell.bash hook)"

# Activate the environment

# Run the training with minimal logging
PYTHONPATH=$PYTHONPATH:. TQDM_DISABLE=1 python main.py train --batch_size 16 --device cuda --epochs 100 --learning_rate 1e-3 --noise_scale 0.43390241265296936 --window_size 5 --overlap 1 --use_wandb --run_name best_parameters --run_id 8oirz9p2 --checkpoints "aa"

# Capture the exit code
training_exit_code=$?

# If training completed successfully, mark it in the log
if [ $training_exit_code -eq 0 ]; then
    echo "Training completed successfully!"
fi

exit $training_exit_code
