import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class ImageEncoder(nn.Module):
    def __init__(self,embed_size=128):
        super(ImageEncoder, self).__init__()
        # Use pretrained resnet18 with the final layer removed
        resnet = models.resnet18(weights=None)  # You can also use pretrained=True
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove classification head
        self.fc = nn.Linear(512, 512)

    def forward(self, images):
        features = self.backbone(images)  # [batch_size, 512, 1, 1]
        features = features.view(features.size(0), -1)  # flatten to [batch_size, 512]
        return self.fc(features)  # project to same dimension as text

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
    
def load_image(path):
    image = Image.open(path).convert("RGB")
    return transform(image)
