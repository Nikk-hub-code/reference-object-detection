from detector.yolo_detector import YOLODetector
from utils.image_utils import crop_image


def main():

    image_path = "test_images/fruits.png"

    # Initialize YOLO detector
    detector = YOLODetector()

    # Detect objects
    detections = detector.detect_objects(image_path)

    # Take first detection
    first_detection = detections[0]

    print("\nFirst Detection:")
    print(first_detection)

    # Extract bounding box
    box = first_detection["box"]

    # Crop object
    cropped_object = crop_image(
        image_path,
        box
    )

    # Show cropped image
    cropped_object.show()


if __name__ == "__main__":
    main()