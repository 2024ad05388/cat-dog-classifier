# cat-dog-classifier
# 🐱 Cat vs Dog Classifier 🐶

A deep learning image classifier that distinguishes cats from dogs, built using transfer learning and deployed as a live web app.

**🔗 Live Demo:** (https://cat-dog-classifier-oyecps4dac3fjvtfv7sert.streamlit.app/)

## Overview
This project fine-tunes a pretrained ResNet18 model (originally trained on ImageNet) to classify images as cats or dogs. Rather than training a CNN from scratch, transfer learning was used to leverage existing learned visual features, significantly reducing training time while achieving strong accuracy.

## Results
- **Training Accuracy:** 96.94%
- **Test Accuracy:** 97.78%
- Trained on 20,000 images (10,000 cats, 10,000 dogs), evaluated on 5,000 held-out images.

## Tech Stack
- **Model:** PyTorch, ResNet18 (transfer learning)
- **Training:** Google Colab (GPU)
- **Dataset:** [Dogs vs Cats (Kaggle)](https://www.kaggle.com/datasets/princelv84/dogsvscats)
- **Model Hosting:** Hugging Face Hub
- **Web App:** Streamlit
- **Deployment:** Streamlit Community Cloud

## How It Works
1. Images are preprocessed (resized to 224x224, normalized to ImageNet statistics)
2. A pretrained ResNet18 has its final layer replaced and fine-tuned to output 2 classes (cat/dog)
3. The trained model weights are hosted on Hugging Face Hub and downloaded at app runtime
4. Users upload a photo through the Streamlit interface and receive a prediction with confidence scores

## What I'd Improve Next
- Add data augmentation beyond horizontal flipping (rotation, color jitter) to improve robustness
- Try unfreezing more layers for further fine-tuning
- Add a confusion matrix and misclassified-image analysis to the README

## Run Locally
```bash
git clone https://github.com/Ak0026/cat-dog-classifier.git
cd cat-dog-classifier
pip install -r requirements.txt
streamlit run app.py
```
