from PIL import Image


def crop_image(
    image_path: str,
    box: list
):

    # Open image
    image = Image.open(image_path)

    # Extract coordinates
    x1, y1, x2, y2 = box

    # Crop image
    cropped_image = image.crop(
        (x1, y1, x2, y2)
    )

    return cropped_image