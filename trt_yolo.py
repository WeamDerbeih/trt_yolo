import os
import time
import argparse
import cv2
import numpy as np
import face_recognition
import pycuda.autoinit
import requests
from queue import Queue
import threading

from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO

WINDOW_NAME = 'TrtYOLODemo'
# Pushover settings
PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN")
PUSHOVER_USER = os.getenv("PUSHOVER_USER")
NOTIFICATION_COOLDOWN = 10
last_notification_time = 0
last_notified_name = None

# Face Recognition Thread Class
class FaceRecognizerThread:
    def __init__(self, known_face_encodings, known_face_names):
        self.known_face_encodings = known_face_encodings
        self.known_face_names = known_face_names
        self.input_queue = Queue(maxsize=1)
        self.output_name = "Unknown"
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while self.running:
            try:
                image = self.input_queue.get(timeout=1)
                self.output_name = self._recognize(image)
            except:
                continue

    def _recognize(self, image):
        if image is None or image.size == 0:
            return "Unknown"
        rgb_small_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")
        if not face_locations:
            return "Unknown"
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        if not face_encodings:
            return "Unknown"
        for face_encoding in face_encodings:
            distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            idx = np.argmin(distances)
            if distances[idx] < 0.6:
                return self.known_face_names[idx]
        return "Unknown"

    def recognize_async(self, image):
        if not self.input_queue.full():
            self.input_queue.put(image)

    def get_result(self):
        return self.output_name

    def stop(self):
        self.running = False
        self.thread.join()

# Load known faces
def load_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []
    print(f"Loading faces from: {known_faces_dir}")
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(('.jpg', '.png')):
            path = os.path.join(known_faces_dir, filename)
            print(f"Processing: {filename}")
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])
                print(f"✅ Loaded face: {filename}")
            else:
                print(f"❌ No face found in: {filename}")
    return known_face_encodings, known_face_names

# Notification function
def send_push_notification(name):
    global last_notification_time, last_notified_name
    
    current_time = time.time()
    
    # Only send notification if:
    # 1. Cooldown period has passed OR
    # 2. It's a different person than last notification
    if (current_time - last_notification_time > NOTIFICATION_COOLDOWN) or (name != last_notified_name):
        try:
            message = f"Recognized: {name}" if name != "Unknown" else "Unknown person detected!"
            
            response = requests.post(
                "https://api.pushover.net/1/messages.json",
                data={
                    "token": PUSHOVER_TOKEN,
                    "user": PUSHOVER_USER,
                    "message": message,
                    "title": "Security Alert"
                }
            )
            if response.status_code == 200:
                last_notification_time = current_time
                last_notified_name = name
                print(f"Push notification sent for: {name}")
            else:
                print(f"Failed to send notification: {response.text}")
        except Exception as e:
            print(f"Notification error: {str(e)}")


def loop_and_detect(cam, trt_yolo, conf_th, vis, recognizer):
    full_scrn = False
    fps = 0.0
    tic = time.time()
    frame_count = 0
    last_detected_name = None
    detection_start_time = None
    min_detection_duration = 2  # seconds

    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break

        img = cam.read()
        if img is None:
            break

        frame_count += 1
        boxes, confs, clss = trt_yolo.detect(img, conf_th)

        current_names = []
        for box, cls in zip(boxes, clss):
            if cls == 0:  # "person"
                x_min, y_min, x_max, y_max = map(int, box)
                expand = 0.35
                w, h = x_max - x_min, y_max - y_min
                x_min = max(0, x_min - int(w * expand))
                y_min = max(0, y_min - int(h * expand))
                x_max = min(img.shape[1], x_max + int(w * expand))
                y_max = min(img.shape[0], y_max + int(h * expand))
                person_img = img[y_min:y_max, x_min:x_max]

                small_img = cv2.resize(person_img, (0, 0), fx=0.9, fy=0.9)
                if frame_count % 8 == 0:
                    recognizer.recognize_async(small_img)

                name = recognizer.get_result()
                current_names.append(name)
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(img, name, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Notification logic
        if current_names:  # If there are any detections
            # Get the most frequently detected name in this frame
            most_common_name = max(set(current_names), key=current_names.count)
            
            if most_common_name != last_detected_name:
                last_detected_name = most_common_name
                detection_start_time = time.time()
            elif detection_start_time and (time.time() - detection_start_time > min_detection_duration):
                send_push_notification(most_common_name)
                detection_start_time = None  # Reset to avoid repeated notifications
        else:
            last_detected_name = None
            detection_start_time = None

        img = show_fps(img, fps)
        cv2.imshow(WINDOW_NAME, img)

        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        fps = curr_fps if fps == 0.0 else (fps * 0.95 + curr_fps * 0.05)
        tic = toc

        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord('F') or key == ord('f'):
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)


def parse_args():
    desc = 'Capture and display live camera video with real-time object detection (YOLO + Face Recognition)'
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument('-c', '--category_num', type=int, default=80,
                        help='number of object categories [default: 80]')
    parser.add_argument('-t', '--conf_thresh', type=float, default=0.3,
                        help='detection confidence threshold [default: 0.3]')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='model name (e.g., yolov4-tiny-416)')
    parser.add_argument('-l', '--letter_box', action='store_true',
                        help='inference with letterboxed image')
    return parser.parse_args()

def main():
    args = parse_args()

    if args.category_num <= 0:
        raise SystemExit('ERROR: Invalid category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: TensorRT model file (yolo/%s.trt) not found!' % args.model)

    known_face_encodings, known_face_names = load_known_faces('known_faces')
    recognizer = FaceRecognizerThread(known_face_encodings, known_face_names)

    cam = Camera(args)
    if not cam.isOpened():
        recognizer.stop()
        raise SystemExit('ERROR: Failed to open camera!')

    cls_dict = get_cls_dict(args.category_num)
    vis = BBoxVisualization(cls_dict)
    trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)

    open_window(WINDOW_NAME, 'YOLO + Face Recognition Demo', cam.img_width, cam.img_height)

    try:
        loop_and_detect(cam, trt_yolo, args.conf_thresh, vis, recognizer)
    finally:
        recognizer.stop()
        cam.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()