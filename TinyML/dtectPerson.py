import torch
import cv2

# Load YOLOv8 small model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # YOLOv5s is similar to YOLOv8 small

# Function to process video and detect persons
def detect_persons_in_video(video_path, output_path):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model(frame)

        # Filter detections for persons (class 0 in COCO dataset)
        for *box, conf, cls in results.xyxy[0]:
            if int(cls) == 0:  # Class 0 is 'person'
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Person {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write frame to output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    print("Detection completed and saved to:", output_path)

# Example usage
video_file = 'TinyML\Input_Video.mp4'
output_file = 'TinyML\output_video.mp4'
detect_persons_in_video(video_file, output_file)