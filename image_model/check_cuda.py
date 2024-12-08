import torch
import tensorflow as tf

if torch.cuda.is_available():
    print("CUDA is available for pytorch.")
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available.")

if tf.test.is_built_with_cuda():
    print("TensorFlow is built with CUDA.")
else:
    print("TensorFlow is not built with CUDA.")