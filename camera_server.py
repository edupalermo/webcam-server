from flask import Flask, Response, request
import cv2
import numpy as np
import threading
import datetime

app = Flask(__name__)
camera_lock = threading.Lock()

def capture_image(camera=0, width=640, height=480, with_date_time_label = False, flip = False):
    with camera_lock:
        print(f"Camera: {camera}")
        cap = cv2.VideoCapture(camera)
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)        
            ret, frame = cap.read()
        finally:
            cap.release()
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


@app.route('/old')
def serve_foto():

    resolution = request.args.get("res", "640x480")  # default to 640x480 if not provided

    try:
        width, height = map(int, resolution.lower().split("x"))
    except Exception:
        return "Invalid resolution format. Use ?res=WIDTHxHEIGHT", 400

    image = capture_image(0, width, height, True)
    if image is None:
        return "Erro ao capturar imagem", 500
    return Response(image, mimetype='image/jpeg')


@app.route('/')
def serve_foto_with_date_label():

    resolution = request.args.get("res", "640x480")  # default to 640x480 if not provided

    try:
        width, height = map(int, resolution.lower().split("x"))
    except Exception:
        return "Invalid resolution format. Use ?res=WIDTHxHEIGHT", 400

    camera = request.args.get("camera", "0")

    flip = not(request.args.get("flip", "0") == "0")

    image = capture_image(int(camera), width, height, True, flip)
    if image is None:
        return "Erro ao capturar imagem", 500
    return Response(image, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=True)
