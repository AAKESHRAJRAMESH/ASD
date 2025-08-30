# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib

# --- 1. Setup and Data Loading ---
# This section defines the paths to your data and loads the images into datasets.

# Define the main data directory
# IMPORTANT: Replace this with the actual path to the parent folder 
# containing your 'TS' and 'TC' subfolders.
data_dir = pathlib.Path('./data/') # Example: './gazemaps/' or 'C:/Users/YourUser/Desktop/gazemaps/'

if not data_dir.exists():
    print(f"❌ Error: The directory '{data_dir}' does not exist.")
    print("Please create it and place your 'TS' and 'TC' folders inside.")
    # As a fallback for demonstration, create dummy directories and images
    print("Creating dummy directories and images for demonstration purposes...")
    ts_dir = data_dir / 'TSI'
    tc_dir = data_dir / 'TCI'
    ts_dir.mkdir(parents=True, exist_ok=True)
    tc_dir.mkdir(parents=True, exist_ok=True)
    # Create a few blank dummy images
    for i in range(50):
        dummy_img_ts = np.zeros((224, 224, 3), dtype=np.uint8)
        dummy_img_tc = np.ones((224, 224, 3), dtype=np.uint8) * 255
        tf.keras.preprocessing.image.save_img(ts_dir / f'dummy_ts_{i}.png', dummy_img_ts)
        tf.keras.preprocessing.image.save_img(tc_dir / f'dummy_tc_{i}.png', dummy_img_tc)
    print("✅ Dummy data created.")


# --- Model and Image Parameters ---
BATCH_SIZE = 32
IMG_HEIGHT = 224 # EfficientNetV2B0 was trained on this size
IMG_WIDTH = 224
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

# --- Create Datasets ---
# Load images from the directories. Keras automatically infers the labels
# from the folder names ('TS' and 'TC').
print("Loading images and creating datasets...")
# Create a full dataset
full_dataset = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  labels='inferred',
  label_mode='binary', # For two classes (ASD/Non-ASD)
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)

# Get class names (e.g., ['TC', 'TS'])
class_names = full_dataset.class_names
print(f"✅ Found classes: {class_names}")

# --- Split Data into Training, Validation, and Test Sets (80-10-10 split) ---
dataset_size = tf.data.experimental.cardinality(full_dataset).numpy()
train_size = int(0.8 * dataset_size)
val_size = int(0.1 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset = full_dataset.take(train_size)
val_dataset = full_dataset.skip(train_size).take(val_size)
test_dataset = full_dataset.skip(train_size + val_size).take(test_size)

print(f"✅ Data split into:")
print(f"  - Training batches:   {tf.data.experimental.cardinality(train_dataset).numpy()}")
print(f"  - Validation batches: {tf.data.experimental.cardinality(val_dataset).numpy()}")
print(f"  - Test batches:       {tf.data.experimental.cardinality(test_dataset).numpy()}")


# --- 2. Configure Dataset for Performance ---
# Use caching and prefetching to optimize the data pipeline.
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)
print("✅ Datasets configured for performance.")


# --- 3. Data Augmentation ---
# Create a layer to apply random transformations to the training images.
# This helps prevent overfitting and improves model generalization.
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal", input_shape=IMG_SHAPE),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)
print("✅ Data augmentation layer created.")


# --- 4. Build the Model (Transfer Learning) ---
print("Building model with EfficientNetV2B0 base...")

# Load the pre-trained EfficientNetV2B0 model
# `include_top=False` removes the original final classification layer.
base_model = tf.keras.applications.EfficientNetV2B0(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)

# Freeze the base model's weights so they are not updated during initial training
base_model.trainable = False

# Create the new model on top
inputs = tf.keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs) # Apply augmentation
x = base_model(x, training=False) # Set the base model to inference mode
x = layers.GlobalAveragePooling2D()(x) # Pool features to a single vector
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x) # Regularization
# Final output layer for binary classification
outputs = layers.Dense(1, activation='sigmoid')(x) 

model = tf.keras.Model(inputs, outputs)

# --- 5. Compile the Model ---
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

print("✅ Model compiled successfully.")
model.summary()


# --- 6. Train the Model (Initial Phase) ---
# Train only the new "head" of the model.
print("\n--- Starting Initial Training (Training the Head) ---")
initial_epochs = 20

# Add callbacks for saving the best model and stopping early if needed
callbacks = [
    tf.keras.callbacks.ModelCheckpoint('best_gaze_model.keras', save_best_only=True, monitor='val_accuracy'),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=val_dataset,
                    callbacks=callbacks)


# --- 7. Fine-Tuning the Full Model ---
# Unfreeze the base model and train the entire network with a very low learning rate.
print("\n--- Starting Fine-Tuning (Training the Full Model) ---")

base_model.trainable = True # Unfreeze the base

# Re-compile the model with a much lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

fine_tune_epochs = 20
total_epochs =  initial_epochs + fine_tune_epochs

history_fine_tune = model.fit(train_dataset,
                              epochs=total_epochs,
                              initial_epoch=history.epoch[-1],
                              validation_data=val_dataset,
                              callbacks=callbacks)


# --- 8. Evaluate the Model ---
print("\n--- Evaluating Final Model on Test Set ---")
loss, accuracy = model.evaluate(test_dataset)
print(f'Test accuracy: {accuracy:.2%}')
print(f'Test loss: {loss:.4f}')


# --- 9. Visualize Training Results ---
def plot_training_history(history, history_fine_tune, initial_epochs):
    """Plots the accuracy and loss curves for training and validation."""
    acc = history.history['accuracy'] + history_fine_tune.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_fine_tune.history['val_accuracy']

    loss = history.history['loss'] + history_fine_tune.history['loss']
    val_loss = history.history['val_loss'] + history_fine_tune.history['val_loss']

    plt.figure(figsize=(12, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.axvline(initial_epochs - 1, color='gray', linestyle='--', label='Start Fine-Tuning')
    plt.ylim([min(plt.ylim()), 1.01])
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.axvline(initial_epochs - 1, color='gray', linestyle='--', label='Start Fine-Tuning')
    plt.ylim([0, max(plt.ylim())])
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

plot_training_history(history, history_fine_tune, initial_epochs)


# --- 10. Predict on a New Image ---
def predict_single_image(image_path, model, class_names):
    """Loads an image, preprocesses it, and predicts its class."""
    if not os.path.exists(image_path):
        print(f"❌ Cannot predict. Image not found at: {image_path}")
        return

    img = tf.keras.utils.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = predictions[0][0]

    print(f"\n--- Prediction for {image_path} ---")
    print(f"This image is most likely **{class_names[int(round(score))]}** with a {100 * (1-score):.2f}% confidence for '{class_names[0]}' and {100 * score:.2f}% for '{class_names[1]}'.")
    print("------------------------------------")


# Example usage: Replace with a path to one of your test images
# Find a sample image from the test set to predict on
try:
    predict_single_image("./data/TCImages/TC005_37.png", model, class_names)
except IndexError:
    print("\nCould not find a sample image to run a test prediction.")

