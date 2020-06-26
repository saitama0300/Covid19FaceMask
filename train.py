import torchvision 
import torch 
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from faceExtraction import *
from model import *
from dataLoder import *

dataset = datasets.ImageFolder(directory, transform = train_transforms)

datasetSize = len(dataset)
trainSize = int(datasetSize * 0.6)
valSize = int(datasetSize * 0.2)
testSize = datasetSize - trainSize - valSize

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [trainSize, valSize, testSize])



BATCH_SIZE = 20

train_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)

LEARNING_RATE = 0.001

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

model.to(device)

EPOCH = 20
train_losses = []
val_losses = []

for epoch in range(EPOCH):

    train_loss = 0
    
    for x,y in train_loader:

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred,y)
        train_loss += loss

        loss.backward()
        optimizer.step()

    train_losses.append(train_loss)

    val_loss = 0

    inter, union = 0,0

    with torch.no_grad():
        for x,y in val_loader:

            x = x.cuda()
            y = y.cuda()

            pred = model(x)
            loss = criterion(pred, y)
            val_loss += loss.item()

            _, t = torch.max(pred.data, 1)

            union += y.size(0)
            inter += (t == y).sum().item()

    val_losses.append(val_loss)
    accuracy = inter/union

    print("EPOCH:", epoch, ", Training Loss:", train_loss, ", Validation Loss:", val_loss, ", Accuracy: ", accuracy)
    
    
    if min(val_losses) == val_losses[-1]:
        best_epoch = epoch
        checkpoint = {'model': model,
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict()}

        torch.save(checkpoint, model_dir + '{}.pth'.format(epoch))
        print("Model saved")