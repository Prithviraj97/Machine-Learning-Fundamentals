from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='onnx', imgsz=(96,96), opset=12, simplify= True)