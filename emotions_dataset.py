#dataset:https://www.kaggle.com/msambare/fer2013?select=train
from sklearn.model_selection import train_test_split
from PIL import Image
import os, glob
import numpy as np

# read image
root_dir = "./emotions_image/"
directory = os.listdir(root_dir)
categories = [f for f in directory if os.path.isdir(os.path.join(root_dir, f))]
print(categories)

num_classes = len(categories)
# resize
image_size = 48

# make image dataset
X = [] # image
Y = [] # label

# read image
for idx, cat in enumerate(categories):

    label = [0 for i in range(num_classes)]
    label[idx] = 1

    image_dir = root_dir + "/" + cat
    files = glob.glob(image_dir+"/*.jpg")
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("L")
        img = img.resize((image_size, image_size))
        data = np.asarray(img)
        X.append(data)
        Y.append(label)

X = np.array(X)
Y = np.array(Y)

# make npy
X_train, X_test, y_train, y_test = \
    train_test_split(X, Y, test_size=0.1)
xy = (X_train, X_test, y_train, y_test)
np.save("./emotions_image/" + "emotions_obj.npy", xy)

print('X_train shape:', X_train.shape)
print("ok,", len(X_train), len(X_test))
