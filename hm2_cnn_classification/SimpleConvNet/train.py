# Setup cell.
from cs231n.classifiers.cnn import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.solver import Solver
import os

# Load the (preprocessed) CIFAR-10 data.
data = get_CIFAR10_data()
for k, v in list(data.items()):
    print(f"{k}: {v.shape}")

  

model = MultiLayerConvNet(weight_scale=0.001, nums_filters=[16,32], hidden_dims=[300,100], reg=0.001)


output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

solver = Solver(
    model,
    data,
    num_epochs=1,
    batch_size=50,
    update_rule='adam',
    optim_config={'learning_rate': 1e-3,},
    lr_decay=0.95,
    verbose=True,
    print_every=50,
    checkpoint_name=os.path.join(output_dir, "checkpoint")
)

import time
start_time = time.time()
solver.train()
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")   

# Print final training accuracy.
print(
    "Full data training accuracy:",
    solver.check_accuracy(data['X_train'], data['y_train'])
)

# Print final validation accuracy.
print(
    "Full data validation accuracy:",
    solver.check_accuracy(data['X_val'], data['y_val'])
)

# Print final test accuracy.
print(
    "Full data test accuracy:",
    solver.check_accuracy(data['X_test'], data['y_test'])
)