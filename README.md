# Brain Tumor Detection using CNN

This project implements a Convolutional Neural Network (CNN) to detect brain tumors from MRI images. The model is built using TensorFlow and Keras.

## Model Architecture

The model uses a sequential CNN architecture designed for binary classification (Tumor vs. No Tumor).

- **Input Layer**: 224x224x3 (RGB Images)
- **Convolutional Blocks**:
  - **Block 1**: Conv2D (32 filters, 3x3) -> ReLU -> MaxPooling2D (2x2)
  - **Block 2**: Conv2D (64 filters, 3x3) -> ReLU -> MaxPooling2D (2x2)
  - **Block 3**: Conv2D (128 filters, 3x3) -> ReLU -> MaxPooling2D (2x2)
  - **Block 4**: Conv2D (128 filters, 3x3) -> ReLU -> MaxPooling2D (2x2)
- **Flatten Layer**: Converts 2D feature maps to 1D vector
- **Dense Layers**:
  - Dense (512 units) -> ReLU
  - Dropout (0.5) for regularization
  - Output Dense (1 unit) -> Sigmoid Activation

## Performance

The model was evaluated on a held-out test set.

- **Validation Accuracy**: 76.47%
- **Test Accuracy**: 76.47%

## Dataset

The dataset should be organized in the following structure:
```
brain_tumor_dataset/
    yes/  # Images with tumor
    no/   # Images without tumor
```

## Usage

### Prerequisites
- Python 3.x
- TensorFlow
- OpenCV (cv2)
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn

### Training the Model
To train the model from scratch, run:
```bash
python main.py
```
This will:
1. Load and preprocess the data.
2. Train the CNN model for 25 epochs.
3. Save the best model as `best_model.h5` and the final model as `brain_tumor_model_final.h5`.
4. Generate training history plots and confusion matrices.

### Testing the Model
To predict on a single image using the trained model:
```bash
python test_model.py path/to/image.jpg
```
Example:
```bash
python test_model.py brain_tumor_dataset/yes/Y1.jpg
```
