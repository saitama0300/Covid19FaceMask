import os
import cv2
from dataLoder import *
import torch


def cropImage(image_name):

    image_path, label_path = getPaths(image_name,"/media/tanmay/DATA/op/dataset")
    
    print(image_path)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    labels, size = useXML(label_path)

    faceLabels = []

    for label in labels:

        name, boundary = label
        cropImage = image[boundary[0][1]:boundary[1][1], boundary[0][0]:boundary[1][0]]
        lbl = 0

        if name == "good":
            lbl = 0
        elif name=="bad":
            lbl = 1
        else:
            lbl = 2

        cropedImgLabel = [cropImage, lbl]
        faceLabels.append(cropedImgLabel)
    
    return faceLabels
#cropImage("-1x-1.jpg")
def createDirectory(dirname):
    try:
        os.mkdir(dirname)
    except FileExistsError:
        print("Directory " + dirname + " already exists.")

directory = "./train/"
label0 = directory + "0/"
label1 = directory + "1/"
model_dir = "./model/"

createDirectory(directory)
createDirectory(label0)
createDirectory(label1)
createDirectory(model_dir)

def getTrainingData(image_names):

    cnt0 = 0
    cnt1 = 0

    for i in image_names:

        cropImg = cropImage(i)
        for j in cropImg:

            img = j[0]
            l = j[1]

            if l==0:
                imageCropName = str(cnt0)+".jpg"
                cv2.imwrite(label0+imageCropName,img)
                cnt0+=1
            elif l==1:
                imageCropName = str(cnt1)+".jpg"
                cv2.imwrite(label1+imageCropName,img)
                cnt1+=1

image_names1 = imageNames("/media/tanmay/DATA/op/dataset")
#getTrainingData(image_names1)           
train_x0 = [f for f in os.listdir(label0) if os.path.isfile(os.path.join(label0, f))]
train_x1 = [f for f in os.listdir(label1) if os.path.isfile(os.path.join(label1, f))]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")