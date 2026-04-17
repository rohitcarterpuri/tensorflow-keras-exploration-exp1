
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

def basic_tensor_operations():
    """Demonstrate basic TensorFlow tensor operations"""
    print("\n=== Basic Tensor Operations ===")
    
    # Creating tensors
    scalar = tf.constant(5)
    vector = tf.constant([1, 2, 3, 4])
    matrix = tf.constant([[1, 2], [3, 4]])
    
    print(f"Scalar: {scalar}")
    print(f"Vector shape: {vector.shape}")
    print(f"Matrix shape: {matrix.shape}")
    
    # Basic operations
    a = tf.constant([[1, 2], [3, 4]])
    b = tf.constant([[5, 6], [7, 8]])
    
    c = tf.add(a, b)
    print(f"Addition:\n{c}")
    
    d = tf.matmul(a, b)
    print(f"Matrix multiplication:\n{d}")
    
    # Gradient computation
    variable = tf.Variable(initial_value=5.0, trainable=True)
    with tf.GradientTape() as tape:
        y = variable ** 2
    gradient = tape.gradient(y, variable)
    print(f"Gradient of x^2 at x=5: {gradient.numpy()}")

def build_and_train_sequential_model():
    """Build and train a basic neural network using Sequential API"""
    print("\n=== Sequential Model ===")
    
    # Load and preprocess MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28*28) / 255.0
    x_test = x_test.reshape(-1, 28*28) / 255.0
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    # Build model
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        x_train, y_train,
        batch_size=32,
        epochs=5,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    return model, history

def build_cnn_model():
    """Build and train a Convolutional Neural Network"""
    print("\n=== CNN Model ===")
    
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    # Build CNN
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train (using subset for speed)
    history = model.fit(
        x_train[:5000], y_train[:5000],
        batch_size=32,
        epochs=3,
        validation_split=0.2,
        verbose=1
    )
    
    return model, history

def plot_training_history(history, title="Training History"):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title(f'{title} - Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    ax2.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title(f'{title} - Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.show()

def save_and_load_model(model):
    """Demonstrate model saving and loading"""
    print("\n=== Model Saving and Loading ===")
    
    # Save model
    model.save('saved_model/my_model.h5')
    print("Model saved successfully!")
    
    # Load model
    loaded_model = keras.models.load_model('saved_model/my_model.h5')
    print("Model loaded successfully!")
    
    return loaded_model

def main():
    """Main function to run all demonstrations"""
    print("TensorFlow Version:", tf.__version__)
    print("Keras Version:", keras.__version__)
    
    # Run demonstrations
    basic_tensor_operations()
    
    print("\n" + "="*50)
    sequential_model, seq_history = build_and_train_sequential_model()
    plot_training_history(seq_history, "Sequential Model")
    
    print("\n" + "="*50)
    cnn_model, cnn_history = build_cnn_model()
    plot_training_history(cnn_history, "CNN Model")
    
    print("\n" + "="*50)
    save_and_load_model(sequential_model)
    
    print("\n✅ All demonstrations completed successfully!")

if __name__ == "__main__":
    main()
