import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_loaders(data_dir, batch_size=32, img_size=(224, 224)):
    # 1. Data Augmentation (Helps AI generalize to messy field photos)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2  # 20% for testing
    )

    # 2. Training Stream
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    # 3. Validation Stream
    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, val_generator

# Usage: 
# train_gen, val_gen = get_data_loaders(r'C:\Users\aravi\Desktop\AgriVision_AI\plantvillage\color')