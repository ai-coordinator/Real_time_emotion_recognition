# coding:utf-8
import os
import re
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

root_dir = "./emotions_image/"
directory = os.listdir(root_dir)
categories = [f for f in directory if os.path.isdir(os.path.join(root_dir, f))]
num_classes = len(categories)

model_path = "./emotions_saved_models/emotions_trained_model.h5"
model = load_model(model_path)
output_path = "./results/output.jpg"
img_path = "./face02.jpg"
pic = cv2.imread(img_path)

image_size = 48

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# start the webcam feed
facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
faces = facecasc.detectMultiScale(gray,scaleFactor=1.11, minNeighbors=8)

for (x, y, w, h) in faces:
    roi_gray = gray[y:y + h, x:x + w]
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (image_size, image_size)), -1), 0)
    prediction = model.predict(cropped_img)

    bestnum = 0.0
    bestclass = 0
    for n in range(num_classes):
        # print("[{}] : {}%".format(categories[n], round(prediction[0][n]*100,2)))
        if bestnum < prediction[0][n]:
            bestnum = prediction[0][n]
            bestclass = n

    cv2.rectangle(pic,(x,y),(x+w,y+h),(0,225,0),5)
    cv2.putText(pic, categories[bestclass], (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, w*0.01, (255, 255, 255), 5, cv2.LINE_AA)

pic1 = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
plt.imshow(pic1)
plt.show()
cv2.imwrite(output_path,pic)
