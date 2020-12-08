import streamlit as st
from streamlit import caching
import os
import torch
from src.core.detect import Detector
from src.core.utils import utils
from PIL import Image
import cv2

st.title('1stDayKit Object Detection')
st.write('1stDayKit is a high-level Deep Learning toolkit for solving generic tasks.')

uploaded_file = st.file_uploader("Choose an image...", type=["png","jpg"])
if uploaded_file is not None:
    st.spinner()
    with st.spinner(text='Loading...'):
        det = Detector(name="DemoDet")
        img = Image.open(uploaded_file)
        img_cv = utils.pil_to_cv2(img)
        output = det.predict(img_cv)
        out_img = det.visualize(img_cv,output,figsize=(18,18))
        cv2.imwrite('tempImage.jpg', out_img)
        st.image('tempImage.jpg',width=700)