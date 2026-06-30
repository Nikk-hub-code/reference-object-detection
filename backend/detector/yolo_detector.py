from ultralytics import YOLO
import cv2

class YOLODetector:

    def __init__(
        self,
        model_path="models/yolov8n.pt",
        confidence_threshold=0.3
    ):

        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def detect_objects(self, image_path):

        results = self.model(image_path)

        detections = []

        for result in results:

            boxes = result.boxes

            for box in boxes:

                confidence = float(box.conf[0])

                # Skip low-confidence detections
                if confidence < self.confidence_threshold:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                class_id = int(box.cls[0])

                class_name = self.model.names[class_id]

                detections.append({
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": [
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    ]
                })

        return detections