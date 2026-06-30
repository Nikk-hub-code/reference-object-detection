from detector.yolo_detector import YOLODetector

detector = YOLODetector()

results = detector.detect_objects(
"test_images/fruits.png"
)

for detection in results:
    print(detection)
