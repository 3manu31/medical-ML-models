import torch
import torch.nn as nn

class LinearBaseline(nn.Module):
    """
    Simple Logistic Regression / Linear baseline.
    Flattens the 28x28x3 input and maps it directly to 5 output logits.
    """
    def __init__(self, num_classes=5):
        super().__init__()
        self.flat = nn.Flatten()
        self.fc = nn.Linear(28 * 28 * 3, num_classes)
        
    def forward(self, x):
        x = self.flat(x)
        return self.fc(x)

class CustomCNN(nn.Module):
    """
    Custom Convolutional Neural Network designed for 28x28 RetinaMNIST images.
    Uses batch normalization, dropout, and residual-like deep features to perform stable training.
    """
    def __init__(self, num_classes=5, dropout_rate=0.4):
        super().__init__()
        
        # Input: 3 x 28 x 28
        self.features = nn.Sequential(
            # Block 1 -> Output: 32 x 14 x 14
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2 -> Output: 64 x 7 x 7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3 -> Output: 128 x 3 x 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Classifier
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

def get_model(model_name="cnn", num_classes=5, **kwargs):
    """
    Factory function to retrieve selected model model.
    """
    if model_name.lower() == "linear":
        return LinearBaseline(num_classes=num_classes)
    elif model_name.lower() == "cnn":
        return CustomCNN(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose 'linear' or 'cnn'.")

if __name__ == "__main__":
    # Test model shape computations
    dummy_input = torch.randn(8, 3, 28, 28)
    
    linear = get_model("linear")
    cnn = get_model("cnn")
    
    out_linear = linear(dummy_input)
    out_cnn = cnn(dummy_input)
    
    print(f"Linear output shape: {out_linear.shape} (Expected: [8, 5])")
    print(f"CNN output shape: {out_cnn.shape} (Expected: [8, 5])")
