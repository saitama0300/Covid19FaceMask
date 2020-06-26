import os
import cv2
import xmltodict

PATH = "/media/tanmay/DATA/op/dataset"
def imageNames(PATH):
    image_names = []
    fullpath = os.path.join(PATH,"images")
    for fileName in os.listdir(fullpath):

        image_names.append(fileName)

    return image_names
#a = imageNames(PATH)
def getPaths(image_name,PATH):

    if image_name[-4:] == 'jpeg':
        label_name = image_name[:-5]+'.xml'
    else:
        label_name = image_name[:-4]+'.xml'

    label_path = os.path.join(PATH,"labels",label_name)
    image_path = os.path.join(PATH,"images",image_name)
    return image_path , label_path
# = getPaths(a[0],PATH)
def useXML(label_path):

    x = xmltodict.parse(open(label_path,'rb'))
    items = x["annotation"]["object"]
    if not isinstance(items, list):
        items = [items]
    result = []

    for item in items:
        name = item["name"]
        boundary = [(int(item['bndbox']['xmin']), int(item['bndbox']['ymin'])),
                  (int(item['bndbox']['xmax']), int(item['bndbox']['ymax']))] 
        result.append((name,boundary))

    size = [int(x['annotation']['size']['width']), 
            int(x['annotation']['size']['height'])]
    
    return result, size
#useXML(B[1])
