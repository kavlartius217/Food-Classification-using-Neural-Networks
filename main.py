import streamlit as st
import tensorflow as tf
import numpy as np

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("/content/trained_model (1).h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Sidebar
st.sidebar.title("Dashboard")
app_mode = "Prediction"  # Setting default page to Prediction

# Main Page
if app_mode == "Prediction":
    st.header("Model Prediction")
    test_image = st.file_uploader("Choose an Image:")
    if st.button("Show Image") and test_image is not None:
        st.image(test_image, width=200, use_column_width=True)
    # Predict button
    if st.button("Predict") and test_image is not None:
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        # Reading Labels
        with open("/content/labels.txt") as f:
            content = f.readlines()
        label = [i.strip() for i in content]
        st.success("Model predicts it's a {}".format(label[result_index]))
