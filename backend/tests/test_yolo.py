from detector.yolo_detector import YOLODetector


def main():

    detector = YOLODetector()

    detections = detector.detect_objects(
        "test_images/fruits.png"
    )

    print("\nDetected Objects:\n")

    for detection in detections:

        print(f"Class: {detection['class_name']}")
        print(f"Confidence: {detection['confidence']:.2f}")
        print(f"Bounding Box: {detection['box']}")
        print("-" * 40)


if __name__ == "__main__":
    main()