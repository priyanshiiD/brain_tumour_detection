# Brain Tumor Detection (Streamlit)

A simple Streamlit web app that uses a trained deep learning model to classify brain MRI images into:
- Tumor
- No Tumor

## Features
- Upload MRI image and get instant prediction
- Confidence score and probability bar
- Test directly with local sample images from `sample_images/`
- Sidebar dataset reference with Kaggle link

## Dataset
Brain MRI Images for Brain Tumor Detection:
https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection

## Project Structure
- `app.py` - Streamlit application
- `brain_tumor_model.h5` - Trained model file
- `class_indices.json` - Class label mapping
- `requirements.txt` - Python dependencies
- `sample_images/` - Sample MRI images for testing

## Requirements
- Python 3.10+ (project currently tested on Python 3.13)
- pip

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

```bash
python -m streamlit run app.py
```

Open in browser:
- http://localhost:8501

## How to Use
1. Choose image source:
   - Upload image, or
   - Use sample image
2. Select/upload MRI image
3. View prediction result and confidence

## Notes
- TensorFlow GPU support on native Windows is limited for recent TensorFlow versions. CPU inference works fine.
- For GPU acceleration on Windows, use WSL2 or compatible TensorFlow DirectML setup.
