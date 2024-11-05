
import streamlit as st
import cv2
from keras.models import load_model
from keras.utils import img_to_array,load_img
import numpy as np
st.set_page_config(page_title='Drowsiness Detection System',page_icon='https://5.imimg.com/data5/PI/FD/NK/SELLER-5866466/images-500x500.jpg')
eyesmodel=cv2.CascadeClassifier("eye.xml ")
drowsinessmodel = load_model("eye.h5")
s=st.sidebar.selectbox('Menu',('Home','IMAGE','WEB CAM','File'))
st.sidebar.image('https://editor.analyticsvidhya.com/uploads/96673dd%201.jfif')
if(s=='Home'):
    st.markdown('<p style="font-family:serif; color:#FFFFFF;font-size:40px;text-align:center;text-decoration: underline;"><b>DROWSINESS DETECTION  SYSTEM </b></p>',unsafe_allow_html=True)

    st.image('https://learnopencv.com/wp-content/uploads/2022/09/02-driver-driver-drowsiness-detection-Driver-drowsing.gif')
elif(s=='IMAGE'):
     st.markdown('<p style="font-family:serif; color:#FFFFFF;font-size:40px;text-align:center;text-decoration: underline;"><b>IMAGE DETECTION </b></p>',unsafe_allow_html=True)
     file=st.file_uploader("Upload an Image")
     if file:
         b=file.getvalue()
         a=np.frombuffer(b,np.uint8)
         img = cv2.imread("pic2.jpg")
         eye = eyesmodel.detectMultiScale(img)
         for (x, y, l, w) in eye:
             cv2.imwrite("temp.jpg", img[y:y+w, x:x+l])
         # Load the image and convert it to grayscale
             eye_img = load_img('temp.jpg', target_size=(64, 64), color_mode='grayscale')
             eye_img = img_to_array(eye_img)
             eye_img = np.expand_dims(eye_img, axis=0)
          # Predict using the drowsiness model
             pred = drowsinessmodel.predict(eye_img)
             if(pred==1):
                  cv2.rectangle(img,(x,y),(x+l,y+w),(0,255,0),4)
             else:
                  cv2.rectangle(img,(x,y),(x+l,y+w),(0,0,255),4)
         st.image(img,channels='BGR')
elif(s=='WEB CAM'):
     st.markdown('<p style="font-family:serif; color:#FFFFFF;font-size:40px;text-align:center;text-decoration: underline;"><b>WEB CAM DETECTION </b></p>',unsafe_allow_html=True)
     k=st.text_input("Enter 0 for primary Camera or 1 from Secondary Camera")
     btn=st.button('Start Camera')
     if btn:
         window=st.empty()
         k=int(k)
         vid=cv2.VideoCapture(k)
         btn2=st.button("Stop Camera")
         if(btn2):
             vid.release()
             st.experimental_rerun()
         while(vid.isOpened()):
             flag,frame=vid.read()
             if(flag):
                 eye=eyesmodel.detectMultiScale(frame)
                 for(x,y,l,w)in eye:
                    cv2.imwrite("temp.jpg",frame[y:y+w,x:x+l])
                    eye_img = load_img('temp.jpg', target_size=(64, 64), color_mode='grayscale')
                    eye_img = img_to_array(eye_img)
                    eye_img = np.expand_dims(eye_img, axis=0)
          # Predict using the drowsiness model
                    pred = drowsinessmodel.predict(eye_img)
                    if(pred==1):
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),4)
                    else:
                         cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),4)
                  
                 window.image(frame,channels='BGR')
elif(s=='File'):
    st.markdown('<p style="font-family:serif; color:#FFFFFF;font-size:40px;text-align:center;text-decoration: underline;"><b>FILE DETECTION </b></p>',unsafe_allow_html=True)
    u=st.text_input('enter the URL or file name')
    window=st.empty()
    btn=st.button('start')
    if btn:
        drowsinessmodel = load_model('eye.h5',compile=False)
        eyesmodel=cv2.CascadeClassifier("eye.xml ")
        v=cv2.VideoCapture(int(u))
        btn2=st.button('Stop')
        if btn2:
            v.release()
            st.experimental_rerun()
        while True:
            flag,frame=v.read()
            if flag:
                ey=eyesmodel.detectMultiScale(frame)
                for(x,y,l,w)in ey:
                    cv2.imwrite("temp.jpg",frame[y:y+w,x:x+l])
                    eye_img = load_img('temp.jpg', target_size=(64, 64), color_mode='grayscale')
                    eye_img = img_to_array(eye_img)
                    eye_img = np.expand_dims(eye_img, axis=0)
          # Predict using the drowsiness model
                    pred = drowsinessmodel.predict(eye_img)
                    if(pred==1):
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),4)
                    else:
                         cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),4)
                  
                window.image(frame,channels='BGR')




 

        



