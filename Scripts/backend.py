import cv2
import numpy as np
from keras.models import load_model
from keras.utils import img_to_array, load_img


eyesmodel=cv2.CascadeClassifier("eye.xml ")
drowsinessmodel = load_model("eye.h5")
# To read an video and show it
vid=cv2.VideoCapture("video3.mp4 ")
pic=1
while(vid.isOpened()):
    flag,frame=vid.read()
    if(flag):
         eye=eyesmodel.detectMultiScale(frame)
         for(x,y,l,w)in eye:
             cv2.imwrite("temp.jpg", frame[y:y+w, x:x+l])
             # Load the image and convert it to grayscale
             eye_img = load_img('temp.jpg', target_size=(64, 64), color_mode='grayscale')
             eye_img = img_to_array(eye_img)
             eye_img = np.expand_dims(eye_img, axis=0)
             # Predict using the drowsiness model
             pred = drowsinessmodel.predict(eye_img)
             if(pred==1):
                 cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),4)
             else:
                 cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),4)

         cv2.namedWindow("ayushi window",cv2.WINDOW_NORMAL)
         cv2.imshow("ayushi window",frame)
         k=cv2.waitKey(33)
         if(k==ord('x')):
             break
    else:
       break
cv2.destroyAllWindows()





'''import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
model = tf.keras.models.load_model('drowsiness_detection_model.h5')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64)).reshape(1, 64, 64, 1) / 255.0
    
    prediction = model.predict(resized)
    if prediction > 0.5:
        label = 'Drowsy'
        color = (0, 0, 255)
    else:
        label = 'Alert'
        color = (0, 255, 0)
    
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow('Drowsiness Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()'''
