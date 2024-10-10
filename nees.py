import cv2
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import tflite_runtime.interpreter as tflite

np.set_printoptions(suppress=True)

# Load YOLO model
yolo_model = YOLO('best(1).onnx')

# Initialize the TFLite interpreter
interpreter = tflite.Interpreter(model_path="model.tflite")
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
    "good": (0, 255, 0),
    "bad": (0, 0, 255),
    "mediocre": (255, 255, 0),
}

def preprocess_image(image):
    size = (width, height)
    image = ImageOps.fit(image, size, Image.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) - 127.5) / 127.5
    return normalized_image_array

def predict_image(image_array):
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    prediction = np.squeeze(output_data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[index]
    return class_name, confidence_score

def detect_and_classify(image_path):
    # Load image
    image = cv2.imread(image_path)
    original_image = image.copy()  # Save the original image for later use
    detection_image = image.copy()  # Image for YOLO detection output
    deepsort_image = image.copy()  # Image for DeepSort output
    classification_image = image.copy()  # Image for classification output

    # Resize the image for YOLO processing
    resized_image = cv2.resize(image, (640, 640))
    results = yolo_model.predict(source=resized_image, imgsz=640, stream=True, conf=0.5, device='cpu')

    detections = []
    for result in results:
        boxes = result.boxes.xyxy
        confidences = result.boxes.conf
        for box, confidence in zip(boxes, confidences):
            x1, y1, x2, y2 = map(int, box)
            detections.append([[x1, y1, x2 - x1, y2 - y1], confidence, 0])  # Class ID set to 0 for all oranges

            # Draw bounding box on the detection image
            cv2.rectangle(detection_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Save the detection image
    detection_image_path = "detection_image.jpg"
    cv2.imwrite(detection_image_path, detection_image)

    # Update tracks using DeepSort
    tracks = tracker.update_tracks(detections, frame=resized_image)

    for track in tracks:
        if track.is_confirmed():
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            # Draw bounding box on the deepsort image
            cv2.rectangle(deepsort_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            display_text = f"ID: {track_id}"
            cv2.putText(deepsort_image, display_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Crop the image for classification
            cropped_frame = resized_image[y1:y2, x1:x2]
            if cropped_frame.size > 0:
                cropped_image = Image.fromarray(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
            else:
                continue
            # Preprocess the image and classify
            image_array = preprocess_image(cropped_image)
            predicted_label, confidence_score = predict_image(image_array)

            color = colors.get(predicted_label, (255, 255, 255))

            # Draw bounding box and classification on the classification image
            cv2.rectangle(classification_image, (x1, y1), (x2, y2), color, 2)
            display_text = f"{predicted_label} ({confidence_score:.2f})"
            cv2.putText(classification_image, display_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Save the DeepSort image
    deepsort_image_path = "deepsort_image.jpg"
    cv2.imwrite(deepsort_image_path, deepsort_image)

    # Save the classification image
    classification_image_path = "classification_image.jpg"
    cv2.imwrite(classification_image_path, classification_image)

    print(f"Detection image saved to {detection_image_path}")
    print(f"DeepSort image saved to {deepsort_image_path}")
    print(f"Classification image saved to {classification_image_path}")

if __name__ == "__main__":
    image_path = input("Enter the path to the image: ")
    detect_and_classify(image_path)
