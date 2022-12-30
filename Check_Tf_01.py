import tensorflow as tf
import os
print(os.environ.get('LD_LIBRARY_PATH'))
# /usr/local/cuda/lib64:/home/kv/miniconda3/lib/:/home/kv/miniconda3/envs/tf/lib/
tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.test.gpu_device_name()