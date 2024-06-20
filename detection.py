from ultralytics import YOLO

# Load the model.
model = YOLO('runs/detect/yolov8n_ll6/weights/best.pt')
results = model.predict(source='test.jpg', conf=0.25)

