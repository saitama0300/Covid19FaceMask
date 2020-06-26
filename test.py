import torchvision 
import torch 
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from faceExtraction import *
from model import *
from dataLoder import *


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath,map_location='cpu')
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    
    return model.eval()


filepath = model_dir + "model.pth"
loaded_model = load_checkpoint(filepath)

train_transforms = transforms.Compose([
                                       transforms.Resize((224,224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                                       ])
'''

dataset = datasets.ImageFolder(directory, transform = train_transforms)

datasetSize = len(dataset)
trainSize = int(datasetSize * 0.6)
valSize = int(datasetSize * 0.2)
testSize = datasetSize - trainSize - valSize

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [trainSize, valSize, testSize])



BATCH_SIZE = 20



test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)
correct = 0
total = 0
    
with torch.no_grad():
    for X, y in test_loader:

        result = loaded_model(X)
        _, maximum = torch.max(result.data, 1)
        total += y.size(0)
        correct += (maximum == y).sum().item()

accuracy = correct/total

print("\n")
print("------------")
print("Accuracy: " + str(accuracy))
print("------------")
print("\n")
'''