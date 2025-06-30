import cv2
import os
import random
import logging
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Constants
CAR_CASCADE_PATH = 'haarcascade_cars.xml'
STOP_CASCADE_PATH = 'haarcascade_stopsign.xml'
IMAGE_FOLDER = 'road_images'
MIN_OBJECT_SIZE = (20, 20)
CAR_COLOR = (0, 255, 0)       # Green
STOP_COLOR = (255, 0, 0)      # Red
BOX_THICKNESS = 2


def get_random_image(folder):
    """Select a random image file from the given folder."""
    image_files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    if not image_files:
        raise FileNotFoundError(f'No image files found in folder: {folder}')
    selected = random.choice(image_files)
    logging.info(f'Randomly selected image: {selected}')
    return os.path.join(folder, selected)


def load_image(path):
    """Load an image from disk."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f'Failed to load image: {path}')
    return img


def detect_objects(gray_img, cascade_path, label):
    """Detect objects using Haar cascade."""
    cascade = cv2.CascadeClassifier(cascade_path)
    try:
        detections = cascade.detectMultiScale(gray_img, minSize=MIN_OBJECT_SIZE)
        logging.info(f'{label} detected: {len(detections)}')
        return detections.tolist()
    except Exception as e:
        logging.warning(f'{label} detection failed: {e}')
        return []


def draw_boxes(img, coords, color, label):
    """Draw bounding boxes and labels on the image."""
    for (x, y, w, h) in coords:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, BOX_THICKNESS)
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def check_forward(cars, stops, img_width):
    """
    Determine if driving forward is allowed.

    Returns:
        bool: True if safe to drive forward, False if blocked.
    """
    if stops:
        return False

    if not cars:
        return True

    left_border = img_width // 3
    right_border = 2 * img_width // 3

    for (x, _, w, _) in cars:
        in_path = x + w > left_border and x < right_border
        is_big = w / img_width > 0.15
        if in_path and is_big:
            return False

    return True


def main():
    # Get a random image
    image_path = get_random_image(IMAGE_FOLDER)
    img = load_image(image_path)

    # Preprocess
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect objects
    car_coords = detect_objects(gray_img, CAR_CASCADE_PATH, 'Car')
    stop_coords = detect_objects(gray_img, STOP_CASCADE_PATH, 'Stop Sign')

    # Draw bounding boxes
    draw_boxes(rgb_img, car_coords, CAR_COLOR, 'Car')
    draw_boxes(rgb_img, stop_coords, STOP_COLOR, 'Stop')

    # Decision logic
    img_width = img.shape[1]
    can_drive = check_forward(car_coords, stop_coords, img_width)
    logging.info(f'Driving allowed: {can_drive}')

    # Display the result
    plt.figure(figsize=(10, 6))
    plt.imshow(rgb_img)
    plt.title(f'Driving Allowed: {can_drive}')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
