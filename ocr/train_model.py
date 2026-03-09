import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os

print("🚀 Training enhanced digit recognition model...")

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize images
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape for CNN
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Data augmentation for better generalization
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1
)

datagen.fit(x_train)

# Build enhanced CNN model
def create_enhanced_model():
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1), padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])
    
    return model

# Create model
model = create_enhanced_model()

# Compile with advanced optimizer
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Callbacks for better training
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2),
    tf.keras.callbacks.ModelCheckpoint('best_digit_model.h5', save_best_only=True)
]

# Train model with data augmentation
print("\n🔄 Training model with data augmentation...")
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=128),
    epochs=20,
    validation_data=(x_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n✅ Final test accuracy: {test_acc:.4f}")

# Save models in multiple formats
model.save("digit_model_enhanced.h5")
model.save("digit_model_fallback.h5")  # Also save as fallback
print("✅ Models saved as digit_model_enhanced.h5 and digit_model_fallback.h5")

# Plot training history
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
print("✅ Training history saved as training_history.png")

# Test with sample digits
print("\n🔍 Testing with sample digits...")
test_samples = x_test[:10]
predictions = model.predict(test_samples, verbose=0)
predicted_digits = np.argmax(predictions, axis=1)

print("Sample predictions:")
for i in range(10):
    print(f"  Sample {i}: Predicted={predicted_digits[i]}, Actual={y_test[i]}, Confidence={np.max(predictions[i]):.4f}")

# Create a simple test script
with open('test_model.py', 'w') as f:
    f.write('''
import tensorflow as tf
import numpy as np
import cv2

# Load model
model = tf.keras.models.load_model('digit_model_enhanced.h5')
print("✅ Model loaded")

# Test with generated digits
for digit in range(10):
    # Create a test image
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.putText(img, str(digit), (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
    
    # Preprocess
    img = cv2.resize(img, (28, 28))
    img = img.reshape(1, 28, 28, 1).astype('float32') / 255.0
    
    # Predict
    pred = model.predict(img, verbose=0)[0]
    predicted = np.argmax(pred)
    confidence = np.max(pred)
    
    print(f"Digit {digit}: Recognized as {predicted} (confidence: {confidence:.4f})")
''')

print("\n✅ Test script created as test_model.py")
print("\n🎉 Training complete!")