import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾")

@st.cache_resource
def load_model():
    device = torch.device('cpu')

    model_path = hf_hub_download(
        repo_id="Ak0026/cat-dog-classifier-model",
        filename="cat_dog_model.pth"
    )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

st.title("🐱 Cat vs Dog Classifier 🐶")
st.write("Built with PyTorch (ResNet18 transfer learning), deployed on Streamlit Cloud.")

uploaded_file = st.file_uploader("Upload a photo of a cat or dog", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0]

    cat_prob = float(probs[0])
    dog_prob = float(probs[1])

    st.subheader("Prediction:")
    if cat_prob > dog_prob:
        st.success(f"🐱 Cat ({cat_prob*100:.1f}% confidence)")
    else:
        st.success(f"🐶 Dog ({dog_prob*100:.1f}% confidence)")

    st.write("**Full probabilities:**")
    st.progress(cat_prob, text=f"Cat: {cat_prob*100:.1f}%")
    st.progress(dog_prob, text=f"Dog: {dog_prob*100:.1f}%")
