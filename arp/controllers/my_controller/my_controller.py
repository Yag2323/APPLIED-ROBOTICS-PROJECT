from controller import Robot
import cv2 as cv
import numpy as np

def get_bgr_image(image, width, height):
    """Convert Webots RGBA image to OpenCV BGR."""
    img_array = np.frombuffer(image, dtype=np.uint8).reshape((height, width, 4))
    return img_array[:, :, :3]  # Drop alpha

robot = Robot()
TIME_STEP = int(robot.getBasicTimeStep())

# Motors and sensors
left_wheel = robot.getDevice('left wheel motor')
right_wheel = robot.getDevice('right wheel motor')
left_wheel.setPosition(float('inf'))
right_wheel.setPosition(float('inf'))

proximity_sensors = []
for i in range(8):
    sensor = robot.getDevice(f'ps{i}')
    sensor.enable(TIME_STEP)
    proximity_sensors.append(sensor)

# Camera
camera = robot.getDevice("camera")
camera.enable(TIME_STEP)

# HSV color ranges (TUNE if needed)
color_ranges = {
    'red':   [([0, 120, 70], [10, 255, 255]), ([170, 120, 70], [180, 255, 255])],
    'green': [([35, 80, 80], [85, 255, 255])],
    'blue':  [([90, 80, 80], [130, 255, 255])],
    'white': [([0, 0, 200], [180, 40, 255])],   # for horse detection
}

color_seen = []
horse_photo_saved = False

while robot.step(TIME_STEP) != -1:
    # --- Sensors and camera ---
    readings = [s.getValue() for s in proximity_sensors]
    image = camera.getImage()
    width = camera.getWidth()
    height = camera.getHeight()
    img_bgr = get_bgr_image(image, width, height)

    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
    mid_x, mid_y = width // 2, height // 2
    center_hsv = hsv[mid_y, mid_x, :]
    print(f"Center HSV: {center_hsv}")

    # --- Box color detection (threshold now 1000) ---
    for color in ['red', 'green', 'blue']:
        mask = np.zeros((height, width), dtype=np.uint8)
        for lower, upper in color_ranges[color]:
            mask |= cv.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
        count = np.sum(mask) // 255
        print(f"{color} mask pixel count: {count}")
        if count > 1000 and color not in color_seen:
            print(f"I see {color} box!")
            color_seen.append(color)
            print("Boxes seen so far:", ", ".join(color_seen))

    # --- Horse (white) detection & photo (now triggers for any detected white pixel) ---
    horse_mask = np.zeros((height, width), dtype=np.uint8)
    for lower, upper in color_ranges['white']:
        horse_mask |= cv.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
    white_pixel_count = np.sum(horse_mask) // 255
    print(f"Horse (white) mask pixel count: {white_pixel_count}")

    if white_pixel_count >= 1 and not horse_photo_saved:
        print("Horse detected! Saving photo as 32x32 (horse_seen.png)")
        horse_img = cv.resize(img_bgr, (32, 32))
        cv.imwrite("horse_seen.png", horse_img)
        horse_photo_saved = True

    # --- Obstacle avoidance logic ---
    obstacle_ahead = any(value > 100 for value in readings)
    if obstacle_ahead:
        print("Obstacle detected. Rotating...")
        left_wheel.setVelocity(4.0)
        right_wheel.setVelocity(-4.0)
    else:
        left_wheel.setVelocity(5.0)
        right_wheel.setVelocity(5.0)

    # --- Optionally, show camera image if not running in Webots GUI ---
    # cv.imshow("Webots Camera", img_bgr)
    # if cv.waitKey(1) & 0xFF == ord('q'):
    #     break

# cv.destroyAllWindows()
