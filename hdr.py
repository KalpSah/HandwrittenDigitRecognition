import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.set_page_config(page_title="handWritten Digit Recogition")

st.title('Select Image')
global img
img=st.file_uploader("Choose a image file", type=['png','jpg','jpeg'])

def pred():
    global img
    abc=img
    file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)[:,:,0]
    img = np.invert(np.array([img]))
    model = tf.keras.models.load_model('handwritten_digits.model')
    prediction = model.predict(img)
    print(max(prediction))
    st.write("The Number is "+format(np.argmax(prediction)))
    st.image(abc,width=200)
    

if(st.button("recognize")):
    pred()


   