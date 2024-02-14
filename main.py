import streamlit as st
import tensorflow as tf
import numpy as np
import keras
#tensorflow Model
def model_prediction(test_image):
     model =tf.keras.models.load_model('leaf.h5')
     image=tf.keras.preprocessing.image.load_img(test_image,target_size=(224,224))
     input_arr = tf.keras.preprocessing.image.img_to_array(image)
     input_arr = np.array([input_arr])  # Convert single image to a batch.
     predictions = model.predict(input_arr)
     return np.argmax(predictions)#return index of max element

st.sidebar.title('Dashboard')
appmode = st.sidebar.selectbox("Menu",["Home","About Project","Prediction","Github"])

#Main Page
if (appmode == "Home"):
    st.header("Medicinal Plant Leaf Recognition")
    image_path="3.jpg"
    st.image(image_path, width=900)


#About Project
elif appmode == "About Project":
    st.header("About Project")
    st.subheader("About Project")
    st.write("This project is about Medicinal Plant Leaf Recognition. It is a web application that can be used to predict the medicinal plant leaf.")
    st.text("This project is developed by using Machine Learning and Deep Learning techniques.")
    st.text("The dataset is collected from the Kaggle.")
    st.text("The dataset contains 32 classes of medicinal plant leaves.")
    st.subheader("Content")
    st.text("This dataset contains three folders:")
    st.text("1. Train")
    st.text("2. Test")
    st.text("3. Validation")

#Prediction Page
    
elif appmode == "Prediction":
        st.header("Prediction")
        st.subheader("Prediction")
        st.write("This page is used to predict the medicinal plant leaf.")
        st.text("Upload the image of the medicinal plant leaf and click on the predict button to get the prediction.")
        test_image =  st.file_uploader("Upload Image", type=["jpg","png","jpeg"])
        if(st.button("Show Image")):
           st.image(test_image, width=4,use_column_width=True)
        if(st.button("Predict")):
            st.write("Prediction is : ")
            st.snow()
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            #Reading Labels
            with open("labels.txt") as f:
                content = f.readlines()
            label = []
            for i in content:
                label.append(i[:-1])
            st.success("Model is Predicting it's a {}".format(label[result_index]))