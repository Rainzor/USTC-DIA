#!/bin/bash

# Common arguments for all training runs
COMMON_ARGS="--epochs 10 --batch_size 64 --lr_decay 0.5 --device gpu"

echo "================================================="
echo "Starting hyperparameter tuning experiments..."
echo "================================================="

# --- Test 0: Baseline ---

echo "--- Training Test 0: Baseline ---"
python main.py --conv 16 32 --fc 300 100 \
$COMMON_ARGS --output_dir output/baseline


# --- Test 1: Adjusting number of convolutional and fully-connected layers ---

# 1a: Deeper network with 3 conv layers and 2 FC layers
echo "--- Training Test 1a: Deeper CNN (3 conv, 2 fc) ---"
python main.py --conv 16 16 32 --fc 300 100 \
$COMMON_ARGS --output_dir output/arch_3conv_2fc

# 1b: Deeper network with 2 conv layers and 3 FC layers
echo "--- Training Test 1b: Deeper FC Net (2 conv, 3 fc) ---"
python main.py --conv 16 32 --fc 300 200 100 --lr 3e-3 \
$COMMON_ARGS --output_dir output/arch_2conv_3fc


# --- Test 2: Adjusting number of filters and neurons ---

# 2a: Wider network with more filters in conv layers
echo "--- Training Test 2a: Wider CNN (more filters) ---"
python main.py --conv 32 64 --fc 300 100 \
$COMMON_ARGS --output_dir output/filters_wider

# 2b: More neurons in fully-connected layers
echo "--- Training Test 2b: Wider FC Net (more neurons) ---"
python main.py --conv 16 32 --fc 600 200 \
$COMMON_ARGS --output_dir output/fc_neurons_more

# 2c: Fewer neurons in fully-connected layers
echo "--- Training Test 2c: Narrower FC Net (fewer neurons) ---"
python main.py --conv 16 32 --fc 150 50 \
$COMMON_ARGS --output_dir output/fc_neurons_less


# --- Test 3: Adjusting convolutional kernel size ---

# 3a: Smaller 3x3 kernel size
echo "--- Training Test 3a: Smaller kernel size (3x3) ---"
python main.py --conv 16 32 --fc 300 100 \
--filter_size 3 $COMMON_ARGS --output_dir output/kernel_3x3

# 3b: Larger 7x7 kernel size
echo "--- Training Test 3b: Larger kernel size (7x7) ---"
python main.py --conv 16 32 --fc 300 100 \
--filter_size 7 $COMMON_ARGS --output_dir output/kernel_7x7

echo "================================================="
echo "All experiments finished."
echo "Check the 'output' directory for results."
echo "=================================================" 