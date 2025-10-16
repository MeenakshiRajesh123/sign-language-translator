import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

MODEL_WEIGHTS_PATH = "checkpoints/sign_language_model.weights.h5"
IMG_SIZE = (224, 224)
CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P",
    "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"
]

base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(len(CLASS_NAMES), activation="softmax")
])

print("🔁 Loading model weights...")
model.load_weights(MODEL_WEIGHTS_PATH)

print("Model loaded successfully!")

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, IMG_SIZE)
    img_array = np.expand_dims(img, axis=0) / 255.0

    preds = model.predict(img_array)
    class_index = np.argmax(preds[0])
    label = CLASS_NAMES[class_index]

    cv2.putText(frame, f"Prediction: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Real-Time Sign Language Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
