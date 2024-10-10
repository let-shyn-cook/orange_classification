import cv2
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time

np.set_printoptions(suppress=True)

# Load YOLO model
yolo_model = YOLO('best(1).onnx')

# Load Keras model
keras_model = load_model("model3_class/keras_model.h5", compile=False)

# Load class names
class_names = [line.strip() for line in open("model3_class/labels.txt", "r").readlines()]

# Initialize DeepSort tracker
tracker = DeepSort(max_age=30)

# Define color map for classes
colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype=np.uint8)

def preprocess_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.LANCZOS)
    image_array = np.asarray(image, dtype=np.float32)
    normalized_image_array = (image_array / 127.5) - 1
    return normalized_image_array

def predict_image(image_array):
    start_time = time.time()
    data = np.expand_dims(image_array, axis=0)  # Add batch dimension
    prediction = keras_model.predict(data)
    keras_latency = time.time() - start_time
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name, confidence_score, keras_latency

# Initialize dictionary to store classification results
classification_results = {}

def update_classification_results(track_id, class_name):
    if track_id not in classification_results:
        classification_results[track_id] = {name: 0 for name in class_names}
        classification_results[track_id]['start_time'] = time.time()
        classification_results[track_id]['final_class'] = None
        classification_results[track_id]['keras_latency'] = []
    classification_results[track_id][class_name] += 1

def determine_final_class(track_id):
    results = classification_results[track_id]
    total_classifications = sum(results[name] for name in class_names)
    if total_classifications == 0:
        return None
    ratios = {name: results[name] / total_classifications for name in class_names}
    if ratios['0 good'] >= 0.7:
        return "Good"
    elif ratios['1 bad'] >= 0.3:
        return "Bad"
    elif ratios['2 mediocre'] >= 0.3:
        return "Mediocre"
    return None

def detect_and_classify():
    cap = cv2.VideoCapture("video.mp4")
    tracked_oranges = {}
    yolo_times = []
    keras_times = []
    latencies = []

    try:
        while cap.isOpened():
            frame_start_time = time.time()

            ret, image = cap.read()
            if not ret:
                break

            original_size = image.shape[1], image.shape[0]  # Save the original size

            # Resize the image for YOLO processing
            resized_image = cv2.resize(image, (640, 640))
            yolo_start_time = time.time()
            results = yolo_model.predict(source=resized_image, imgsz=640, stream=True, conf=0.7, device='cpu')
            yolo_latency = time.time() - yolo_start_time
            yolo_times.append(yolo_latency)

            detections = []
            for result in results:
                boxes = result.boxes.xyxy.numpy()
                confidences = result.boxes.conf.numpy()
                for box, confidence in zip(boxes, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    detections.append([[x1, y1, x2 - x1, y2 - y1], confidence, 0])  # Class ID set to 0 for all oranges

            # Update tracks using DeepSort
            tracks = tracker.update_tracks(detections, frame=resized_image)

            keras_latency = 0
            for track in tracks:
                if track.is_confirmed():
                    track_id = track.track_id
                    ltrb = track.to_ltrb()
                    x1, y1, x2, y2 = map(int, ltrb)

                    # Resize coordinates to match original image size
                    x1 = int(x1 * (original_size[0] / 640))
                    y1 = int(y1 * (original_size[1] / 640))
                    x2 = int(x2 * (original_size[0] / 640))
                    y2 = int(y2 * (original_size[1] / 640))

                    cropped_frame = image[y1:y2, x1:x2]  # Use original image here
                    if cropped_frame.size > 0:
                        cropped_image = Image.fromarray(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
                    else:
                        continue

                    # Preprocess the image and classify
                    image_array = preprocess_image(cropped_image)
                    predicted_label, confidence_score, keras_latency_single = predict_image(image_array)
                    keras_latency += keras_latency_single
                    keras_times.append(keras_latency_single)

                    update_classification_results(track_id, predicted_label)

                    # Determine the final class if 5 seconds have passed since the first detection
                    if time.time() - classification_results[track_id]['start_time'] > 5 and \
                            classification_results[track_id]['final_class'] is None:
                        final_class = determine_final_class(track_id)
                        classification_results[track_id]['final_class'] = final_class
                        if final_class:
                            tracked_oranges[track_id] = final_class

                    final_class = classification_results[track_id]['final_class']

                    if final_class:
                        if final_class == "Good":
                            color = (0, 255, 0)
                        elif final_class == "Bad":
                            color = (0, 0, 255)
                        elif final_class == "Mediocre":
                            color = (255, 255, 0)
                    else:
                        color = (255, 255, 255)  # White for undetermined class

                    # Display the classification result or placeholder on the original image
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    display_text = f"ID: {track_id} {final_class if final_class else 'Classifying...'}"
                    cv2.putText(image, display_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Calculate FPS
            end_time = time.time()
            fps = 1 / (end_time - frame_start_time)
            cv2.putText(resized_image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the image
            cv2.imshow("Detect and classification orange", image)

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

        # Print YOLO and Keras model inference times
        print(f"Average YOLO inference time: {np.mean(yolo_times):.4f} seconds")
        print(f"Average Keras inference time: {np.mean(keras_times):.4f} seconds")
        print(f"Average latency: {np.mean(latencies):.4f} seconds")

if __name__ == "__main__":
    detect_and_classify()
