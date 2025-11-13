import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
# ============================================
# STEP 1: LOAD AND PREPROCESS DATA
# ============================================

def load_data(data_dir, img_size=(224, 224)):
    """
    Load images from directory structure:
    data_dir/
        yes/  (tumor images)
        no/   (no tumor images)
    """
    images = []
    labels = []
    
    # Load tumor images (label = 1)
    tumor_path = os.path.join(data_dir, 'yes')
    if os.path.exists(tumor_path):
        for img_name in os.listdir(tumor_path):
            img_path = os.path.join(tumor_path, img_name)
            try:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(1)  # Tumor = 1
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    # Load non-tumor images (label = 0)
    no_tumor_path = os.path.join(data_dir, 'no')
    if os.path.exists(no_tumor_path):
        for img_name in os.listdir(no_tumor_path):
            img_path = os.path.join(no_tumor_path, img_name)
            try:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(0)  # No tumor = 0
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    # Convert to numpy arrays
    images = np.array(images, dtype='float32')
    labels = np.array(labels)
    
    # Normalize pixel values to [0, 1]
    images = images / 255.0
    
    print(f"Loaded {len(images)} images")
    print(f"Tumor images: {np.sum(labels == 1)}")
    print(f"No tumor images: {np.sum(labels == 0)}")
    
    return images, labels


def visualize_samples(images, labels, n_samples=9):
    """Visualize random samples from the dataset"""
    plt.figure(figsize=(12, 12))
    indices = np.random.choice(len(images), n_samples, replace=False)
    
    for i, idx in enumerate(indices):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[idx])
        plt.title(f"Label: {'Tumor' if labels[idx] == 1 else 'No Tumor'}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_images.png')
    print("Sample images saved as 'sample_images.png'")
    plt.show()


# ============================================
# STEP 2: BUILD THE CNN MODEL
# ============================================

def build_model(input_shape=(224, 224, 3)):
    """
    Build a CNN model similar to what you learned in Andrew Ng's course
    Architecture:
    - CONV -> RELU -> MAXPOOL
    - CONV -> RELU -> MAXPOOL
    - CONV -> RELU -> MAXPOOL
    - FLATTEN
    - DENSE -> RELU -> DROPOUT
    - DENSE (output)
    """
    model = keras.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                     input_shape=input_shape, name='conv1'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
        layers.MaxPooling2D((2, 2), name='pool2'),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3'),
        layers.MaxPooling2D((2, 2), name='pool3'),
        
        # Fourth Convolutional Block (optional, for deeper learning)
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv4'),
        layers.MaxPooling2D((2, 2), name='pool4'),
        
        # Flatten and Dense Layers
        layers.Flatten(name='flatten'),
        layers.Dense(512, activation='relu', name='dense1'),
        layers.Dropout(0.5, name='dropout'),  # Prevent overfitting
        layers.Dense(1, activation='sigmoid', name='output')  # Binary classification
    ])
    
    return model


# ============================================
# STEP 3: TRAIN THE MODEL
# ============================================

def train_model(model, X_train, y_train, X_val, y_val, epochs=25, batch_size=32):
    """Train the model with validation"""
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks for better training
    callbacks = [
        # Save the best model
        keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Reduce learning rate when stuck
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1,
            min_lr=1e-7
        ),
        # Stop if not improving
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return history


# ============================================
# STEP 4: EVALUATE AND VISUALIZE RESULTS
# ============================================

def plot_training_history(history):
    """Plot training and validation accuracy/loss"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history saved as 'training_history.png'")
    plt.show()


def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set"""
    # Get predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Calculate accuracy
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['No Tumor', 'Tumor']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Tumor', 'Tumor'],
                yticklabels=['No Tumor', 'Tumor'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'")
    plt.show()
    
    return y_pred, y_pred_prob


def visualize_predictions(X_test, y_test, y_pred, y_pred_prob, n_samples=9):
    """Visualize predictions on test samples"""
    plt.figure(figsize=(15, 15))
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    for i, idx in enumerate(indices):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X_test[idx])
        
        true_label = 'Tumor' if y_test[idx] == 1 else 'No Tumor'
        pred_label = 'Tumor' if y_pred[idx] == 1 else 'No Tumor'
        confidence = y_pred_prob[idx][0] if y_pred[idx] == 1 else 1 - y_pred_prob[idx][0]
        
        color = 'green' if y_test[idx] == y_pred[idx] else 'red'
        plt.title(f"True: {true_label}\nPred: {pred_label} ({confidence*100:.1f}%)",
                 color=color, fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png')
    print("Predictions saved as 'predictions.png'")
    plt.show()


# ============================================
# STEP 5: MAIN EXECUTION
# ============================================

def main():
    """Main function to run the complete pipeline"""
    
    print("="*60)
    print("BRAIN TUMOR DETECTION USING CNN")
    print("="*60)
    
    # Configuration
    DATA_DIR = 'brain_tumor_dataset'  # Change this to your dataset path
    IMG_SIZE = (224, 224)
    EPOCHS = 25
    BATCH_SIZE = 32
    
    # Step 1: Load data
    print("\n[1/5] Loading data...")
    X, y = load_data(DATA_DIR, IMG_SIZE)
    
    # Visualize samples
    visualize_samples(X, y)
    
    # Step 2: Split data (60% train, 20% validation, 20% test)
    print("\n[2/5] Splitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Training set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")
    print(f"Test set: {len(X_test)} images")
    
    # Step 3: Build model
    print("\n[3/5] Building model...")
    model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    model.summary()
    
    # Step 4: Train model
    print("\n[4/5] Training model...")
    history = train_model(model, X_train, y_train, X_val, y_val, 
                         epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    # Plot training history
    plot_training_history(history)
    
    # Step 5: Evaluate model
    print("\n[5/5] Evaluating model...")
    y_pred, y_pred_prob = evaluate_model(model, X_test, y_test)
    
    # Visualize predictions
    visualize_predictions(X_test, y_test, y_pred, y_pred_prob)
    
    # Save final model
    model.save('brain_tumor_model_final.h5')
    print("\nModel saved as 'brain_tumor_model_final.h5'")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


# ============================================
# PREDICTION FUNCTION FOR NEW IMAGES
# ============================================

def predict_single_image(model_path, image_path, img_size=(224, 224)):
    """
    Predict tumor presence in a single image
    
    Args:
        model_path: Path to saved model (.h5 file)
        image_path: Path to image to predict
        img_size: Size to resize image to
    """
    # Load model
    model = keras.models.load_model(model_path)
    
    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, img_size)
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    # Predict
    prediction = model.predict(img_batch)[0][0]
    
    # Display result
    plt.figure(figsize=(8, 6))
    plt.imshow(img_resized)
    
    if prediction > 0.5:
        result = f"TUMOR DETECTED\nConfidence: {prediction*100:.2f}%"
        color = 'red'
    else:
        result = f"NO TUMOR DETECTED\nConfidence: {(1-prediction)*100:.2f}%"
        color = 'green'
    
    plt.title(result, fontsize=14, color=color, weight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    print(result)
    return prediction


#if __name__ == "__main__":
    # Run the complete training pipeline
    #main()
    
    # Example: To predict on a new image after training:
    # predict_single_image('brain_tumor_model_final.h5', 'path/to/new/image.jpg')

