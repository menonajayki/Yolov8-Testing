from ultralytics import YOLO
import torch

# Check if CUDA is available
if not torch.cuda.is_available():
    print("CUDA is not available. Training will proceed on the CPU.")
    device = 'cpu'
else:
    print("CUDA is available. Training will proceed on the GPU.")
    device = 'cuda:0'
