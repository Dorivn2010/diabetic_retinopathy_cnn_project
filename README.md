# Diabetic Retinopathy Classification using CNN

## Overview
A binary CNN classifier to detect Diabetic Retinopathy (DR vs No_DR) 
from retinal fundus images. Built as a coursework project using TensorFlow/Keras.

## Dataset
- 2,076 retinal fundus images across 2 classes: `DR` and `No_DR`
- Images loaded, resized to 224×224, and normalized (pixel values ÷ 255)
- 80/20 train-test split with random_state=42 for reproducibility

## Model Architecture
- 3× Conv2D layers (128 filters, 3×3, ReLU) each followed by MaxPooling2D
- Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(1, Sigmoid)
- Binary classification compiled with Adam + binary_crossentropy

## Results
- Final validation accuracy: ~91.35% (Epoch 1) to ~91.83% (Epoch 3)
- Training accuracy reached ~94%+ by Epoch 6

## Requirements
See `requirements.txt`

## How to Run
1. Clone the repo
2. `pip install -r requirements.txt`
3. Mount Google Drive with dataset at `/content/drive/My Drive/Diabetic_Retinopathy/train`
4. Run `48_Grade_Notebook2(CNN).ipynb`

## Author
Kevin Adjei Aholou — [@Dorivn2010](https://github.com/Dorivn2010)
```

## 📄 requirements.txt
```
tensorflow
numpy
pandas
matplotlib
scikit-learn
pathlib
