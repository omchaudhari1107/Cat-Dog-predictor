## Face Classification between Dog(🐶) and Cat(😺) with accuracy of 97.02%

### Import's required
```
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import pandas as pd
from joblib import dump
import cv2
import pywt
import os
```
### Confusion Matrix(cat-0 & dog-1)
![download](https://github.com/omchaudhari1107/Cat-Dog-predictor/assets/90174038/d46399c2-cadf-4734-9141-2445b555795d)

### Method
- Firstly we need to detect the face of cat or dog on the given image and fetch it from the defined function by using CascadeClassifier of [cat](https://github.com/timatooth/catscanface/blob/master/haarcascade_frontalcatface_extended.xml) and [dog](https://github.com/kskd1804/dog_face_haar_cascade/blob/master/cascade.xml)
    ```
    cat_cascade = cv2.CascadeClassifier("./haarcascade_frontalcatface_extended.xml")
    dog_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt2.xml')
    
    def detect_faces(img, cascade):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.04, minNeighbors=4, minSize=(30, 30))
        return faces
    ```
- After fetching it we need to import the data sample into Jupyter notebook or any other IDE, then we need to do Wavelet Transformation of that image

  ![Screenshot 2024-03-16 091128](https://github.com/omchaudhari1107/Cat-Dog-predictor/assets/90174038/116d0ee9-36af-4ed4-94ff-6564ae4f49ef)

- Then we need to Vstack or vertically stack the image for accurate measures
     ```
     X, y = [], []
    c_list = os.listdir('./cat_train')
    d_list = os.listdir('./dog_train')
    for i in c_list:
        img = cv2.imread(f'./cat_train/{i}')
        img_har = w2d(f'./cat_train/{i}','db1',5)
        combined_img = np.vstack((img.reshape(200*200*3,1),img_har.reshape(200*200,1)))
        X.append(combined_img)
        y.append(0)
        
    for i in d_list:
        img = cv2.imread(f'./dog_train/{i}')
        img_har = w2d(f'./dog_train/{i}','db1',5)
        combined_img = np.vstack((img.reshape(200*200*3,1),img_har.reshape(200*200,1)))
        X.append(combined_img)
        y.append(1)
     ```
- Then finally we get the features(x & y).
- AFter it just an Standard Scaling and Model building task remains.
- You can use diffrent models such as Ramdom Forest,KNN,Decision Tree etc... (I prefer SVM due to it's Kernal's)
    
