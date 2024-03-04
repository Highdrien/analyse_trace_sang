import torch
import torchvision.models as models
import torch.nn as nn
from torchvision.models.resnet import ResNet18_Weights
from torch.nn import functional as F
from itertools import chain


class InceptionResNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, dropout_probability=0.5, hidden_size=100):   
        super(InceptionResNet, self).__init__() # Call parent's constructor

        # Load the pre-trained ResNet-18 model
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Freeze all the pre-trained layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Add a dropout layer and modify the last layer of the model
        self.dropout = nn.Dropout(dropout_probability)
        self.fc1 = torch.nn.Linear(1000, hidden_size)  # Change the input size to 1000
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

        # Unfreeze the newly added layers
        for param in self.fc1.parameters():
            param.requires_grad = True
        for param in self.fc2.parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        input: x is a tensor of shape (batch_size, 3, 512, 512)
        output: y is a tensor of shape (batch_size, num_classes)
        """
        x = self.model(x)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        y = self.fc2(x)
        
        return y
    
    def get_parameters(self):
        """
        Returns all the parameters of the model, the trainable ones and the frozen ones
        """
        return self.parameters()  # Use self.parameters() instead of self.model.parameters()

    def count_total_parameters(self):
        """
        Returns the total number of parameters in the model
        """
        return sum(p.numel() for p in self.parameters())  # Use self.parameters() instead of self.model.parameters()

    def count_trainable_parameters(self):
        """
        Returns the number of trainable parameters in the model
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)  # Use self.parameters() instead of self.model.parameters()


class AdversarialInceptionResNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, dropout_probability=0.5, hidden_size=100, background_classes=4):   
        super(AdversarialInceptionResNet, self).__init__() # Call parent's constructor

        # Load the pre-trained ResNet-18 model
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Freeze all the pre-trained layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Add a dropout layer and modify the last layer of the model
        self.dropout = nn.Dropout(dropout_probability)
        self.fc1 = torch.nn.Linear(1000, hidden_size)  # Change the input size to 1000
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

        # Add an adversary
        self.adversary = torch.nn.Linear(hidden_size, background_classes)

        # Unfreeze the newly added layers
        for param in self.fc1.parameters():
            param.requires_grad = True
        for param in self.fc2.parameters():
            param.requires_grad = True
        for param in self.adversary.parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        input: x is a tensor of shape (batch_size, 3, 512, 512)
        output: y is a tensor of shape (batch_size, num_classes)
        """
        x = self.model(x)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        y = self.fc2(x)
        
        # Adversarial output
        adv_output = self.adversary(x)  # detach to avoid backpropagation to the main model ??



        return y, adv_output
    
    def main_parameters(self):
        """
        Returns the parameters of the main model
        """
        return chain(self.fc1.parameters(), self.fc2.parameters())

    def adversary_parameters(self):
        """
        Returns the parameters of the adversary
        """
        return self.adversary.parameters()
    

if __name__ == '__main__':
    model = InceptionResNet()
    print("Total parameters:",model.count_total_parameters())
    print("Trainable parameters:",model.count_trainable_parameters())
    print("Parameters:",model.get_parameters())
    print(model.model)
    #test of the model
    #creata a random tensor
    x = torch.randn(1,3,512,512)
    #forward
    
    y = model.forward(x)

    print("y shape:",y.shape)
    print("y:",y)


    # Apply softmax to the logits
    #probabilities = torch.nn.functional.softmax(y.logits, dim=1) #que pour inception !

    # Get the predicted class
    #_, predicted_class = torch.max(probabilities, 1)

    #print("Classes:",predicted_class) #
    #print the output shape
    #print(y.shape)
    #print the output
    #print(y)