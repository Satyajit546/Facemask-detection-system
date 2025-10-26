# Import necessary modules from Keras and TensorFlow
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the Model
# ==================================

# Create a Sequential model
mymodel = Sequential()

# --- First Convolutional Block ---
# 32 filters, 3x3 kernel, ReLU activation. 
# Input shape (150, 150, 3) is set for 150x150 color images.
mymodel.add(Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))
mymodel.add(MaxPooling2D(2,2))

# --- Second Convolutional Block ---
mymodel.add(Conv2D(32, (3,3), activation='relu'))
mymodel.add(MaxPooling2D(2,2))

# --- Third Convolutional Block ---
mymodel.add(Conv2D(32, (3,3), activation='relu'))
mymodel.add(MaxPooling2D(2,2))

# Flatten the 3D output to 1D for the Dense layers
mymodel.add(Flatten())

# --- Dense Layers ---
# Hidden Dense layer with 100 units and ReLU activation
mymodel.add(Dense(100, activation='relu'))
# Output Dense layer with 1 unit and Sigmoid activation (for binary classification)
mymodel.add(Dense(1, activation='sigmoid'))

# Compile the model
mymodel.compile(optimizer='Adam', 
                loss='binary_crossentropy', 
                metrics=['accuracy'])

# Display the model structure (optional but helpful)
mymodel.summary()

# ==================================
# Define the Data Generators
# ==================================

# Setup for training data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,                 # Normalize pixel values to 0-1
    shear_range=0.2,                # Apply random shearing
    zoom_range=0.2,                 # Apply random zoom
    horizontal_flip=True            # Allow random horizontal flips
)

# Setup for testing data (only rescaling/normalization is typically applied)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training images from the 'train' directory
# (Requires a structure like: train/class_a/img.jpg, train/class_b/img2.jpg)
train_img = train_datagen.flow_from_directory(
    'train',
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary'  # For 2-class classification
)

# Load validation/test images from the 'test' directory
test_img = test_datagen.flow_from_directory(
    'test',
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary'
)



# Train and Test the model

# Train the model and store the history object

print("\nStarting model training...")
mask_model_history = mymodel.fit(
    train_img,
    epochs=10,
    validation_data=test_img
)
print("Training complete.")


# ==================================
# Save the model
# ==================================

# Save the trained model to a file
# NOTE: The original image had an extra argument here (mask_model), 
# which is not used in the correct Keras save syntax.
mymodel.save('mask.keras')
print("Model saved to mask.keras")

# You can now load the model later with:
# from keras.models import load_model
# loaded_model = load_model('mask.h5')
