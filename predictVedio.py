from test import *

import cvlib as cv
import cv2
from PIL import Image

cap = cv2.VideoCapture(0)

font_scale=1
thickness = 2
red = (0,0,255)
green = (0,255,0)
blue = (255,0,0)
font=cv2.FONT_HERSHEY_SIMPLEX

#File must be downloaded
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'./haarcascade_frontalface_default.xml')


while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.4, 4)
        
        for (x, y, w, h) in faces:
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), blue, 2)
            
            croped_img = frame[y:y+h, x:x+w]
            pil_image = Image.fromarray(croped_img, mode = "RGB")
            pil_image = train_transforms(pil_image)
            image = pil_image.unsqueeze(0)
            
            
            result = loaded_model(image)
            _, maximum = torch.max(result.data, 1)
            prediction = maximum.item()

            
            if prediction == 0:
                cv2.putText(frame, "Masked", (x,y - 10), font, font_scale, green, thickness)
                cv2.rectangle(frame, (x, y), (x+w, y+h), green, 2)
            elif prediction == 1:
                cv2.putText(frame, "No Mask", (x,y - 10), font, font_scale, red, thickness)
                cv2.rectangle(frame, (x, y), (x+w, y+h), red, 2)
        
        cv2.imshow('frame',frame)
        
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()