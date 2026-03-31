import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import json

# --- 1. CONFIGURATION ---
DATASET_PATH = r'C:\Users\aravi\Desktop\AgriVision_AI\plantvillage dataset\color'
MODEL_SAVE_NAME = "trained_model.keras"
IMAGE_SIZE = (224, 224)  # MobileNetV2 standard size
BATCH_SIZE = 32

if not os.path.exists(DATASET_PATH):
    print(f"❌ ERROR: Dataset path not found: {DATASET_PATH}")
else:
    # --- 2. ADVANCED DATA AUGMENTATION ---
    # This creates "new" images by flipping and rotating to prevent memorization
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # 20% for testing
    )

    print("🚀 Loading and Augmenting Data...")
    train_gen = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # --- 3. TRANSFER LEARNING ARCHITECTURE ---
    print("🧠 Initializing MobileNetV2 Base...")
    # Load pre-trained model without the top classification layer
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze the "base" knowledge

    # Build the custom "head" for your 38 plant classes
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),  # Dropout prevents overfitting
        layers.Dense(train_gen.num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # --- 4. SMART TRAINING CALLBACKS ---
    # Stops training if accuracy stops improving to save time and electricity
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=3, 
        restore_best_weights=True
    )

    print(f"🔥 Training started for up to 20 epochs...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        callbacks=[early_stop]
    )

    # --- 5. EXPORT MODEL & LABELS ---
    model.save(MODEL_SAVE_NAME)
    
    # Critical: Save the folder names so the App knows which ID is which plant
    with open('class_indices.json', 'w') as f:
        json.dump(train_gen.class_indices, f)
        
    print(f"✅ SUCCESS! Model saved as {MODEL_SAVE_NAME}")
    print(f"✅ Labels saved as class_indices.json")
    # --- 6. PLOT PERFORMANCE ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend(loc='lower right')

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend(loc='upper right')

# Save the graph as an image for your report
plt.savefig('training_performance.png')
print("📊 Performance graph saved as 'training_performance.png'")
plt.show()