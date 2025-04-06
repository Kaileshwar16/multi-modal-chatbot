import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, 512)

    def forward(self, images):
        features = self.backbone(images)
        features = features.view(features.size(0), -1)
        return self.fc(features)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def load_image(path):
    image = Image.open(path).convert("RGB")
    return transform(image)
