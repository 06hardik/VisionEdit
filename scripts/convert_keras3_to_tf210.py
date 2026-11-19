"""
convert_keras3_to_tf210.py - Convert Colab Keras 3 Model to Native TF 2.10 Format
==================================================================================
Colab runs modern Keras 3.13, which saves models as zip archives.
Windows native TensorFlow 2.10 uses Keras 2.10 format.
This script extracts the exact weights from the Keras 3 zip container,
loads them into the identical TF 2.10 architecture, verifies predictions,
and saves in native TF 2.10 format.
"""

import os
# Inject CUDA and cuDNN paths
os.environ['PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin;' + r'c:\Users\ayush\OneDrive\Desktop\study\CV project\cudnn_temp\cudnn-windows-x86_64-8.9.7.29_cuda11-archive\bin;' + os.environ.get('PATH', '')

import zipfile
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, Dense
import config
from model import build_model_with_attention

def convert_model(input_keras3_path=r"results/best_model (2).keras", output_path=r"results/best_model.keras"):
    print(f"[+] Loading Keras 3 container: {input_keras3_path}")
    if not os.path.exists(input_keras3_path):
        raise FileNotFoundError(f"File not found: {input_keras3_path}")
        
    # 1. Extract weights file
    extract_dir = "results_temp"
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(input_keras3_path, 'r') as z:
        z.extract('model.weights.h5', extract_dir)
        
    weights_path = os.path.join(extract_dir, 'model.weights.h5')
    h5_file = h5py.File(weights_path, 'r')
    h5_layers = h5_file['layers']
    
    # 2. Build identical TF 2.10 model architecture (128x128 input)
    print("[+] Building native TensorFlow 2.10 model architecture...")
    model = build_model_with_attention(num_classes=config.NUM_CLASSES, img_size=128)
    
    # 3. Map weights layer-by-layer sequentially
    conv_idx = 0
    depth_idx = 0
    bn_idx = 0
    dense_idx = 0
    
    matched_count = 0
    total_weights_loaded = 0
    
    print("[+] Transferring layer weights from Keras 3 to TF 2.10...")
    for layer in model.layers:
        if isinstance(layer, Conv2D):
            key = 'conv2d' if conv_idx == 0 else f'conv2d_{conv_idx}'
            conv_idx += 1
            if key in h5_layers and 'vars' in h5_layers[key]:
                vars_dict = h5_layers[key]['vars']
                weights = [vars_dict[str(i)][:] for i in range(len(vars_dict))]
                layer.set_weights(weights)
                matched_count += 1
                total_weights_loaded += len(weights)
                
        elif isinstance(layer, DepthwiseConv2D):
            key = 'depthwise_conv2d' if depth_idx == 0 else f'depthwise_conv2d_{depth_idx}'
            depth_idx += 1
            if key in h5_layers and 'vars' in h5_layers[key]:
                vars_dict = h5_layers[key]['vars']
                weights = [vars_dict[str(i)][:] for i in range(len(vars_dict))]
                layer.set_weights(weights)
                matched_count += 1
                total_weights_loaded += len(weights)
                
        elif isinstance(layer, BatchNormalization):
            key = 'batch_normalization' if bn_idx == 0 else f'batch_normalization_{bn_idx}'
            bn_idx += 1
            if key in h5_layers and 'vars' in h5_layers[key]:
                vars_dict = h5_layers[key]['vars']
                # In BatchNorm: [gamma, beta, moving_mean, moving_variance]
                weights = [vars_dict[str(i)][:] for i in range(len(vars_dict))]
                layer.set_weights(weights)
                matched_count += 1
                total_weights_loaded += len(weights)
                
        elif isinstance(layer, Dense):
            key = 'dense' if dense_idx == 0 else f'dense_{dense_idx}'
            dense_idx += 1
            if key in h5_layers and 'vars' in h5_layers[key]:
                vars_dict = h5_layers[key]['vars']
                weights = [vars_dict[str(i)][:] for i in range(len(vars_dict))]
                layer.set_weights(weights)
                matched_count += 1
                total_weights_loaded += len(weights)

    h5_file.close()
    print(f"[+] Successfully transferred {matched_count} layers ({total_weights_loaded} weight tensors)!")
    
    # 4. Verify model with test prediction
    test_input = np.ones((1, 128, 128, 3), dtype=np.float32)
    test_out = model.predict(test_input, verbose=0)[0]
    print(f"[+] Sanity Check Output Shape: {test_out.shape}, Sum: {np.sum(test_out):.4f}")
    assert len(test_out) == 7, "Output should have 7 emotion classes!"
    
    # Backup existing best_model.keras if present
    if os.path.exists(output_path):
        backup_path = output_path.replace(".keras", "_previous.keras")
        print(f"[+] Backing up previous model to: {backup_path}")
        if os.path.exists(backup_path):
            os.remove(backup_path)
        os.rename(output_path, backup_path)
        
    # 5. Save model in native TF 2.10 format
    print(f"[+] Saving converted model to: {output_path}")
    model.save(output_path)
    print(f"[+] Conversion complete! File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    return model

if __name__ == '__main__':
    convert_model()

