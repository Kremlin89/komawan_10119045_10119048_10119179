import nltk
from unittest import result
import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import cv2 
from PIL import Image, ImageEnhance
import numpy as np

FACE_CASCADE = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_faces(image):
    new_img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 6)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+h]
        roi_color = img[y:y+h, x:x+h]
        
    return img, faces

def cartonize_image(image):
    new_img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

    #Edge
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

    #Color
    color = cv2.bilateralFilter(img, 9, 300, 300)

    #Cartoon
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def main():
    """Face Detection App"""
    st.title ("Aplikasi Deteksi Wajah dan Analisis Sentimen Komentar Aplikasi")
    st.text ("Menggunakan OpenCV, NLTK, Github dan Streamlit")

    activities = ["Deteksi Wajah", "Analisis Sentimen"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == 'Deteksi Wajah':
        st.subheader("Deteksi Wajah")

        image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

        if image_file is not None:
            image = Image.open(image_file)
            st.text("Original Image")
            st.image(image)       
        
        task = ["Deteksi Wajah", "Cartonize"]
        feature_choice = st.sidebar.selectbox("Task", task)
        if st.button("Process"):
            if feature_choice == 'Deteksi Wajah':
                result_img, result_face = detect_faces(image)
                st.success("Found {} faces".format(len(result_face)))
                st.image(result_img)
            elif feature_choice == 'Cartonize':
                result_img = cartonize_image(image)
                st.image(result_img)
    
    elif choice == 'Analisis Sentimen':
        #Dataset
        nltk.download('vader_lexicon')

        #User Input
        usr_input = st.text_input("Nilai Produk Kami: ")

        #Proses
        sentiment = SentimentIntensityAnalyzer()
        score = sentiment.polarity_scores(usr_input)
        
        st.write(score)

        #Kondisi
        if score["neu"] > score["neg"] and score["neu"] > score["pos"]:
               st.write("# Netral")
        elif score["neg"] > score["pos"]:
               st.write("# Buruk")
        else:
               st.write("# Positif")
        

if __name__ == '__main__':
    main()
