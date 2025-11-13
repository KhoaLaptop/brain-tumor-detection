import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt

def predict_image(model_path, image_path):
    """Predict tumor on a single image"""
    print(f"Loading model from: {model_path}")
    
    # Load model
    try:
        model = keras.models.load_model(model_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    print(f"Loading image from: {image_path}")
    
    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        print(f"✗ Error: Could not load image from {image_path}")
        print("Make sure the file exists and the path is correct")
        return
    
    print(f"✓ Image loaded: {img.shape}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224))
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    print("Making prediction...")
    # Predict
    prediction = model.predict(img_batch, verbose=1)[0][0]
    
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
    plt.savefig('prediction_result.png')
    print(f"\n{'='*50}")
    print(result)
    print(f"Image: {image_path}")
    print(f"Result saved as: prediction_result.png")
    print('='*50)
    
    # Show the plot
    print("\nDisplaying image... (close the window to continue)")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_model.py <image_path>")
        print("Example: python test_model.py brain_tumor_dataset/yes/Y1.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Check if model exists
    model_path = 'brain_tumor_model_final.h5'
    import os
    if not os.path.exists(model_path):
        print(f"✗ Error: Model file '{model_path}' not found!")
        print("Please train the model first by running: python main.py")
        sys.exit(1)
    
    predict_image(model_path, image_path)