import cv2
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import tflite_runtime.interpreter as tflite
import time

np.set_printoptions(suppress=True)

# Load YOLO model with GPU support
yolo_model = YOLO('iuo_saved_model/iuo_full_integer_quant.tflite', device='cuda:0')

# Initialize the TFLite interpreter with GPU delegate
interpreter = tflite.Interpreter(model_path="model.tflite", experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
interpreter.allocate_tensors()

# Get input and output details for TFLite model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get input shape details
input_shape = input_details[0]['shape']
height, width = input_shape[1], input_shape[2]

# Define class names
class_names = ['good', 'bad', 'mediocre']

# Initialize DeepSort tracker
tracker = DeepSort(max_age=30)

# Define color map for classes
colors = {
    "Good": (0, 255, 0),
    "Bad": (0, 0, 255),
    "Mediocre": (255, 255, 0),
    "Classifying": (255, 255, 255)
}

def preprocess_image(image):
    size = (width, height)
    image = ImageOps.fit(image, size, Image.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) - 127.5) / 127.5
    return normalized_image_array

def predict_image(image_array):
    image_array = np.expand_dims(image_array, axis=0)
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    prediction = np.squeeze(output_data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[index]
    return class_name, confidence_score

# Initialize dictionary to store classification results
classification_results = {}

def update_classification_results(track_id, class_name):
    if track_id not in classification_results:
        classification_results[track_id] = {
            'good': 0,
            'bad': 0,
            'mediocre': 0,
            'start_time': time.time(),
            'final_class': None
        }
    classification_results[track_id][class_name] += 1

def determine_final_class(track_id):
    results = classification_results[track_id]
    total_classifications = sum(results[c] for c in class_names)
    if total_classifications == 0:
        return None

    good_ratio = results['good'] / total_classifications
    bad_ratio = results['bad'] / total_classifications
    mediocre_ratio = results['mediocre'] / total_classifications

    if good_ratio >= 0.7:
        return "Good"
    elif bad_ratio >= 0.3:
        return "Bad"
    elif mediocre_ratio >= 0.3:
        return "Mediocre"
    return None

# Function to detect oranges and classify them
def detect_and_classify():
    cap = cv2.VideoCapture("video.mp4")
    tracked_oranges = {}

    try:
        while cap.isOpened():
            start_time = time.time()

            ret, image = cap.read()
            if not ret:
                break

            original_size = image.shape[1], image.shape[0]  # Save the original size

            # Resize the image for YOLO processing
            resized_image = cv2.resize(image, (640, 640))
            results = yolo_model.predict(source=resized_image, imgsz=224, stream=True, conf=0.7)

            detections = []
            for result in results:
                boxes = result.boxes.xyxy
                confidences = result.boxes.conf
                for box, confidence in zip(boxes, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    detections.append([[x1, y1, x2 - x1, y2 - y1], confidence, 0])  # Class ID set to 0 for all oranges

            # Update tracks using DeepSort
            tracks = tracker.update_tracks(detections, frame=resized_image)

            for track in tracks:
                if track.is_confirmed():
                    track_id = track.track_id
                    ltrb = track.to_ltrb()
                    x1, y1, x2, y2 = map(int, ltrb)

                    cropped_frame = resized_image[y1:y2, x1:x2]
                    if cropped_frame.size > 0:
                        cropped_image = Image.fromarray(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
                    else:
                        continue
                    # Preprocess the image and classify
                    image_array = preprocess_image(cropped_image)
                    predicted_label, confidence_score = predict_image(image_array)

                    update_classification_results(track_id, predicted_label)

                    # Determine the final class if 5 seconds have passed since the first detection
                    if time.time() - classification_results[track_id]['start_time'] > 5 and \
                            classification_results[track_id]['final_class'] is None:
                        final_class = determine_final_class(track_id)
                        classification_results[track_id]['final_class'] = final_class
                        if final_class:
                            tracked_oranges[track_id] = final_class

                    final_class = classification_results[track_id]['final_class'] or "Classifying"

                    # Display the classification result on the resized image (640x640)
                    color = colors.get(final_class, (255, 255, 255))
                    cv2.rectangle(resized_image, (x1, y1), (x2, y2), color, 2)
                    display_text = f"ID: {track_id} {final_class}"
                    cv2.putText(resized_image, display_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Calculate FPS
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            cv2.putText(resized_image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Resize the image back to its original size
            final_image = cv2.resize(resized_image, original_size)
            cv2.imshow("Detect and classification orange", final_image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

        # Print the summary
        total_oranges = len(tracked_oranges)
        good_oranges = sum(1 for v in tracked_oranges.values() if v == "Good")
        bad_oranges = sum(1 for v in tracked_oranges.values() if v == "Bad")
        mediocre_oranges = sum(1 for v in tracked_oranges.values() if v == "Mediocre")

        print(f"Total oranges: {total_oranges}")
        print(f"Good oranges: {good_oranges}")
        print(f"Bad oranges: {bad_oranges}")
        print(f"Mediocre oranges: {mediocre_oranges}")

if __name__ == "__main__":
    detect_and_classify()
