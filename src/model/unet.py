import torch
from torch import nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Decoder
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
        
        # Final output
        self.conv4 = nn.Conv2d(128, 1, 1)  # Adjusted the input channels to 128

    def forward(self, x):
        # Encoder
        c1 = F.relu(self.conv1(x))
        x = self.pool(c1)
        x = F.relu(self.conv2(x))
        
        # Decoder
        x = self.up(x)
        x = F.relu(self.conv3(x))
        
        # Concatenate with earlier output
        x = torch.cat([x, c1], axis=1)
        
        # Final output
        x = self.conv4(x)
        
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        #add softmax activation
        x = F.softmax(x, dim=1)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters()) 
    
    
if __name__ == "__main__":
    # Create a random tensor representing a batch of images with size 128x128 and 3 channels
    x = torch.randn(1, 3, 128, 128)

    # Create an instance of the UNet model
    unet = UNet()

    # Pass the image through the model
    output = unet.forward(x)

    print("Output shape:", output.shape)

    # Create an instance of the Classifier model
    classifier = Classifier(num_classes=10)

    # Pass the output of the UNet model through the classifier
    class_output = classifier.forward(output)

    print("Classifier output shape:", class_output.shape)