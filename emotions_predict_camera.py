# coding:utf-8
import os
import re
import cv2
import keras
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import array_to_img, img_to_array,load_img

root_dir = "./emotions_image/"
directory = os.listdir(root_dir)
categories = [f for f in directory if os.path.isdir(os.path.join(root_dir, f))]
num_classes = len(categories)

image_size = 48

if __name__ == "__main__":

    model = load_model('./emotions_saved_models/emotions_trained_model.h5')

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # start the webcam feed
    cap = cv2.VideoCapture('6.mp4')
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,225,0),2)
            cv2.putText(frame, categories[bestclass], (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, w*0.02, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("People's analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
