import torch
import torch.nn as nn
import torchvision.models as models

class LinearBaseline(nn.Module):
    """
    Simple Logistic Regression / Linear baseline.
    Flattens the 28x28x3 input and maps it directly to 5 output logits.
    (Still available if run with the linear config on 28x28 data, 
    but we adapt it here to also dynamically handle 224x224 input).
    """
    def __init__(self, num_classes=5, input_size=28):
        super().__init__()
        self.flat = nn.Flatten()
        self.fc = nn.Linear(input_size * input_size * 3, num_classes)
        
    def forward(self, x):
        x = self.flat(x)
        return self.fc(x)

class CustomCNN(nn.Module):
    """
    Custom Convolutional Neural Network designed for 28x28 RetinaMNIST images.
    """
    def __init__(self, num_classes=5, dropout_rate=0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        features = self.features(x)
        logits = self.classifier(features)
        return logits

class ResNet18Transfer(nn.Module):
    """
    ResNet18 architecture for transfer learning on 224x224 RetinaMNIST images.
    Loads default pretrained weights, freezes early layers, and leaves the final 
    residual block (layer4) and classification head (fc) trainable.
    """
    def __init__(self, num_classes=5):
        super().__init__()
        # Load pre-trained ResNet18 (backward compatible with older torchvision)
        try:
            self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except AttributeError:
            self.resnet = models.resnet18(pretrained=True)
        
        # 1. Freeze all base parameters
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # 2. Unfreeze the final convolutional block (layer4) for fine-tuning
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
            
        # 3. Swap and unfreeze the classification head
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
            
    def forward(self, x):
        return self.resnet(x)

def get_model(model_name="cnn", num_classes=5, input_size=224, **kwargs):
    """
    Factory function to retrieve selected model.
    """
    name_lower = model_name.lower()
    if name_lower == "linear":
        return LinearBaseline(num_classes=num_classes, input_size=input_size)
    elif name_lower == "cnn":
        return CustomCNN(num_classes=num_classes, **kwargs)
    elif name_lower == "resnet18":
        return ResNet18Transfer(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose 'linear', 'cnn', or 'resnet18'.")

if __name__ == "__main__":
    # Test model shape computation for ResNet18
    dummy_input = torch.randn(2, 3, 224, 224)
    model = get_model("resnet18")
    
    # Verify parameter freezes
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    print(f"ResNet18 output shape: {model(dummy_input).shape} (Expected: [2, 5])")
    print(f"Trainable Parameters: {trainable_params}")
    print(f"Frozen Parameters: {frozen_params}")
