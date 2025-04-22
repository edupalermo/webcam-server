from flask import Flask, Response, request
import cv2
import numpy as np
import threading
import datetime
import time
import gpiod
from gpiod.line import Direction, Value

CHIP_NAME = '/dev/gpiochip1'
LINE_OFFSET = 72

app = Flask(__name__)
camera_lock = threading.Lock()


def capture_image(camera=0, width=640, height=480, with_date_time_label=False, flip=False, light=False):
    with camera_lock:
        if (light):
            turn_light_on()
            time.sleep(0.3)
        cap = cv2.VideoCapture(camera, cv2.CAP_V4L2)
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
#            cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.3)   # Range: 0.0–1.0 or device-specific
#            cap.set(cv2.CAP_PROP_EXPOSURE, -4)      # Often negative values work (depends on camera)
#            cap.set(cv2.CAP_PROP_GAIN, 0)           # Lower gain if image is noisy
            ret, frame = cap.read()
        finally:
            cap.release()
            if (light):
                turn_light_off()
    if not ret:
        return None
    if flip:
        frame = cv2.flip(frame, -1)
    frame = auto_brightness_contrast(frame)
    if with_date_time_label:
        frame = print_date_time_label(frame)
    _, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()


def print_date_time_label(frame):
    # Get current datetime as string
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Set font and position
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    border_thickness = thickness + 2

    # Calculate the size of the text
    (text_width, text_height), _ = cv2.getTextSize(timestamp, font, scale, thickness)
    x = frame.shape[1] - text_width - 10  # 10 px from right
    y = frame.shape[0] - 10               # 10 px from bottom

    # Draw black border (stroke)
    cv2.putText(frame, timestamp, (x, y), font, scale, (0, 0, 0), border_thickness, cv2.LINE_AA)
    # Draw white text
    cv2.putText(frame, timestamp, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


def auto_brightness_contrast(image, clip_hist_percent=1):
    """
    Automatically adjusts brightness and contrast using histogram clipping.
    clip_hist_percent: percentage of histogram to clip for contrast stretching.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = hist.cumsum()

    # Locate points to clip
    maximum = accumulator[-1]
    clip_amount = clip_hist_percent * maximum / 100.0
    clip_low = np.searchsorted(accumulator, clip_amount)
    clip_high = np.searchsorted(accumulator, maximum - clip_amount)

    # Calculate alpha and beta
    alpha = 255 / (clip_high - clip_low)
    beta = -clip_low * alpha

    # Apply to original image
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted


def turn_light_on():
    set_gpio(1)


def turn_light_off():
    set_gpio(0)


def set_gpio(value: int):
    val = Value.ACTIVE if value else Value.INACTIVE

    with gpiod.request_lines(
        CHIP_NAME,
        consumer="flask-light",
        config={
            LINE_OFFSET: gpiod.LineSettings(
                direction=Direction.OUTPUT,
                output_value=val,
            )
        },
    ) as request:
        request.set_value(LINE_OFFSET, val)


@app.route('/')
def serve_foto_with_date_label():

    resolution = request.args.get("res", "640x480")  # default to 640x480 if not provided

    try:
        width, height = map(int, resolution.lower().split("x"))
    except Exception:
        return "Invalid resolution format. Use ?res=WIDTHxHEIGHT", 400

    camera = request.args.get("camera", "0")

    flip = not(request.args.get("flip", "0") == "0")

    light = not(request.args.get("light", "0") == "0")

    image = capture_image(int(camera), width, height, True, flip, light)
    if image is None:
        return "Erro ao capturar imagem", 500
    return Response(image, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=True)
