import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from skimage.feature import local_binary_pattern
from PIL import Image

class CustomCNN(nn.Module):
    def __init__(self, feature_size):
        super(CustomCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 3 * 3 + feature_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 7)
        )

    def forward(self, image, features):
        cnn_features = self.conv_layers(image)
        cnn_features = cnn_features.view(cnn_features.size(0), -1)
        combined_features = torch.cat((cnn_features, features), dim=1)
        output = self.fc(combined_features)
        return output

def load_model(model_path, feature_size, device="cuda" if torch.cuda.is_available() else "cpu"):
    model = CustomCNN(feature_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def extract_lbp_features(image, radius=1, points=8):
    lbp = local_binary_pattern(image, points, radius, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, points + 3), range=(0, points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalize
    return hist

def extract_orb_features(image, max_features=100):
    orb = cv2.ORB_create(nfeatures=max_features)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    if descriptors is None:
        descriptors = np.zeros((max_features, 32), dtype=np.float32)
    return descriptors.flatten()[:max_features * 32]

# Preprocess OpenCV image
# Improved Preprocessing Function
def preprocess_cv2_image(image):
    # Ensure the image is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize to 48x48 while keeping aspect ratio
    image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_AREA)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor


# Predict class using OpenCV image
def predict_from_cv2(model_path, cv2_image, device="cuda" if torch.cuda.is_available() else "cpu"):
    image_tensor = preprocess_cv2_image(cv2_image).to(device)
    
    # Extract LBP + ORB features
    lbp_features = extract_lbp_features(cv2_image)
    orb_features = extract_orb_features(cv2_image)
    
    features = np.concatenate([lbp_features, orb_features])
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension
    model = load_model(model_path, 3210)

    with torch.no_grad():
        output = model(image_tensor, features)
        predicted_class = torch.argmax(output, dim=1).item()

    return predicted_class

# Mapping of class IDs to emotion labels
emotion_labels = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise"
}

# Example Usage:
cv2_image = cv2.imread("Training_80443921.jpg", cv2.IMREAD_GRAYSCALE)
class_id = predict_from_cv2("emotion_model.pth", cv2_image)
class_name = emotion_labels.get(class_id, "Unknown")

print(f"Predicted class: {class_id} ({class_name})")

