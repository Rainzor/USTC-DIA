# Setup cell.
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.cnn import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.solver import CupySolver
import os

checkpoint_name = "output/checkpoint_cupy_epoch_1.pkl"

# Load the (preprocessed) CIFAR-10 data.
data = get_CIFAR10_data()
for k, v in list(data.items()):
    print(f"{k}: {v.shape}")
  
solver = CupySolver()

solver.from_pretrained(checkpoint_name)

print(solver.check_accuracy(data['X_test'], data['y_test']))


plt.subplot(2, 1, 1)
loss_history = np.array(solver.loss_history)
window_size = 50
loss_moving_average = np.convolve(loss_history, np.ones((window_size,))/window_size, mode='valid')
plt.plot(loss_moving_average, '-.', label='loss')
plt.xlabel('iteration')
plt.ylabel('loss')

plt.subplot(2, 1, 2)
plt.plot(solver.train_acc_history, '-o')
plt.plot(solver.val_acc_history, '-o')
plt.legend(['train', 'val'], loc='upper left')
plt.xlabel('epoch')
plt.ylabel('accuracy')
os.makedirs('output', exist_ok=True)
plt.savefig('output/loss_and_accuracy.png')