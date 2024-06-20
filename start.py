from ultralytics import YOLO
import torch

# Check if CUDA is available
if not torch.cuda.is_available():
    print("CUDA is not available. Training will proceed on the CPU.")
    device = 'cpu'
else:
    print("CUDA is available. Training will proceed on the GPU.")
    device = 'cuda:0'

# Load the model.
model = YOLO('yolov8n.pt')

# Training.
results = model.train(
    data='head and chocolate/data.yaml',
    imgsz=640,
    epochs=1000,
    batch=-1,
    plots=True,
    patience=250,
    device=device,
    name='yolov8n_ll'
)
