import os
import json
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LambdaCallback
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_PATH = "processed_dataset/train"
VAL_PATH = "processed_dataset/val"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
TOTAL_EPOCHS = 25
PROGRESS_FILE = "training_progress.json"
CHECKPOINT_PATH = "checkpoints/sign_language_model.weights.h5"

try:
    tf.config.set_visible_devices([], 'GPU')
    print("⚙️ Using CPU mode for stable training on macOS.")
except Exception as e:
    print(f"⚙️ GPU configuration skipped: {e}")

if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, "r") as f:
        progress = json.load(f)
    start_epoch = progress.get("completed_epochs", 0)
else:
    start_epoch = 0

remaining_epochs = TOTAL_EPOCHS - start_epoch
print(f"📈 Resuming training from epoch {start_epoch + 1} — {remaining_epochs} epochs remaining.")

if remaining_epochs <= 0:
    print("✅ Training already completed.")
    exit()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    VAL_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

os.makedirs("checkpoints", exist_ok=True)

checkpoint = ModelCheckpoint(
    CHECKPOINT_PATH,
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    save_weights_only=True
)

tensorboard_callback = TensorBoard(log_dir="logs")

# This callback saves progress after each epoch
progress_callback = LambdaCallback(
    on_epoch_end=lambda epoch, logs: (
        print(f"Epoch {epoch+1} done — "
              f"Loss: {logs['loss']:.4f}, Val Loss: {logs['val_loss']:.4f}, "
              f"Acc: {logs['accuracy']:.4f}, Val Acc: {logs['val_accuracy']:.4f}"),
        json.dump({"completed_epochs": epoch+1}, open(PROGRESS_FILE, "w"))
    )
)

if os.path.exists(CHECKPOINT_PATH):
    print("🔁 Loading checkpoint weights...")
    model.load_weights(CHECKPOINT_PATH)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=start_epoch + remaining_epochs,
    initial_epoch=start_epoch,
    callbacks=[checkpoint, tensorboard_callback, progress_callback]
)

print(" Training complete! Model weights saved.")
