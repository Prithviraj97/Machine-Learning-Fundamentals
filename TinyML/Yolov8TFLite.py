from ultralytics import YOLO
model = YOLO('C:\\Users\\Admin\\Machine Learning Fundamentals\\yolov5s.pt')

model.export(format = 'tflite', int8=True)