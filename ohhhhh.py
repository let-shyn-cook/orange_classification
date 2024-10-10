import cv2
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt

# Load Keras model
keras_model = load_model("model3_class/keras_model.h5", compile=False)

# Load class names
class_names = [line.strip() for line in open("model3_class/labels.txt", "r").readlines()]


# Preprocess image function
def preprocess_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.LANCZOS)
    image_array = np.asarray(image, dtype=np.float32)
    normalized_image_array = (image_array / 127.5) - 1
    return normalized_image_array


# Predict image function
def predict_image(image_array):
    data = np.expand_dims(image_array, axis=0)  # Add batch dimension
    prediction = keras_model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name, confidence_score


# Function to classify all images in folder and display results
def classify_images_in_folder(folder_path):
    images = []
    labels = []

    # Iterate through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Accept jpg and png images
            file_path = os.path.join(folder_path, filename)

            # Open the image
            image = Image.open(file_path)
            image_array = preprocess_image(image)

            # Predict the label and confidence
            class_name, confidence_score = predict_image(image_array)

            # Append the image and label
            images.append(np.array(image))
            labels.append(f"{class_name} ({confidence_score * 100:.2f}%)")

    # Create a mosaic of images and display labels
    display_images_with_labels(images, labels)


# Function to display images with labels
def display_images_with_labels(images, labels):
    n = len(images)
    fig, axs = plt.subplots(1, n, figsize=(15, 5))

    for i, (img, label) in enumerate(zip(images, labels)):
        axs[i].imshow(img)
        axs[i].set_title(label)
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    folder_path = "bad"  # Thay bằng đường dẫn đến thư mục chứa ảnh
    classify_images_in_folder(folder_path)
