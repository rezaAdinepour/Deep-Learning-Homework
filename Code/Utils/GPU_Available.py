import torch
import tensorflow as tf


def print_tensorflow():
    '''
    Output Tensorflow related build
    '''
    print('Tensorflow', tf.__version__)
    print('isBuild with CUDA:', tf.test.is_built_with_cuda())

def print_tf_gpu_info():
    '''
    Output GPU(s) info that is compatible to CUDA on the machine in TF
    '''
    print('Number of GPU(s):', len(tf.config.list_physical_devices('GPU')))
    print('GPU(s):', tf.config.list_physical_devices('GPU'))


def print_torch():
    print('PyTorch', torch.__version__)

def print_torch_gpu_info():
    '''
    Output GPU(s) info that is compatible to CUDA on the machine in Torch
    '''
    print('isBuild with CUDA:', torch.cuda.is_available())
    print('Your GPU model is:', torch.cuda.get_device_name(0))