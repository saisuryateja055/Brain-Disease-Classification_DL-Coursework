import os
import streamlit as st
import tensorflow as tf
from keras.models import model_from_config
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from PIL import Image
import numpy as np
import h5py

# Set page configuration
st.set_page_config(layout="wide")

def main():
    # Initialize the 'page' attribute if it's not already set
    if 'page' not in st.session_state:
        st.session_state['page'] = 'Home'

    # Use columns to create a layout for the title and navbar
    col1, col2 = st.columns([3, 1])

    with col1:
        st.title("Brain Disease Classifier")

    with col2:
        st.write("##")  # Just to add vertical space and align the buttons with the title
        if st.button("Home"):
            st.session_state.page = "Home"
        if st.button("Classify"):
            st.session_state.page = "Classify"
        if st.button("About Brain Diseases"):
            st.session_state.page = "About Brain Diseases"

    # Content of the pages based on the selection
    if st.session_state.page == "Home":
        render_home_page()
    elif st.session_state.page == "Classify":
        render_classify_page()
    elif st.session_state.page == "About Brain Diseases":
        render_about_page()

# Define the path to the models
MODEL_PATHS = {
    "Brain Stroke": os.path.join(os.path.dirname(__file__), "brain_stroke.h5"),
    "Alzheimer's": os.path.join(os.path.dirname(__file__), "alzheimer.h5"),
    "Tumor": os.path.join(os.path.dirname(__file__), "tumor.h5")
}

# Helper function to modify and load model configuration
def load_model_without_time_major(path):
    with h5py.File(path, 'r') as f:
        model_config = f.attrs.get('model_config')
        model_config = model_config.decode('utf-8')
        model_config_dict = eval(model_config)
        
        # Remove 'time_major' argument if it exists
        def remove_time_major(layer_config):
            if 'time_major' in layer_config['config']:
                del layer_config['config']['time_major']
            return layer_config
        
        # Apply the removal to all layers in the model configuration
        if 'layers' in model_config_dict['config']:
            model_config_dict['config']['layers'] = [
                remove_time_major(layer) for layer in model_config_dict['config']['layers']
            ]
        elif 'network' in model_config_dict['config']:
            model_config_dict['config']['network'] = [
                remove_time_major(layer) for layer in model_config_dict['config']['network']
            ]

    model = model_from_config(model_config_dict)
    model.load_weights(path)
    return model

# Load your models
MODELS = {}
for name, path in MODEL_PATHS.items():
    try:
        custom_objects = {'GlorotUniform': glorot_uniform}
        with CustomObjectScope(custom_objects):
            model = load_model_without_time_major(path)
        MODELS[name] = model
    except Exception as e:
        st.error(f"Error loading model '{name}': {e}")

# Print model paths for debugging
for name, path in MODEL_PATHS.items():
    st.write(f"Model '{name}' path: {path}")

# Function to load image and preprocess it
def load_image(image_file):
    image = Image.open(image_file)
    image = image.resize((124, 124))  # Resize the image if your model expects a different size
    image = np.array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to generate report
def generate_report(patient_name, patient_age, test_type, prediction_label):
    report_text = f"""
    Medical Report
    --------------
    Patient Name: {patient_name}
    Patient Age: {patient_age}
    Test Type: {test_type}
    Prediction: {'Condition Positive' if prediction_label[0] == 1 else 'Condition Negative'}

    Note: This is a preliminary assessment and not a definitive diagnosis.
    """
    return report_text

# Function to render the home page
def render_home_page():
    st.header("Welcome to the Brain Disease Classifier")
    st.markdown(
        """
        <div style="text-align: justify;">
            This web application utilizes advanced deep learning techniques to classify brain diseases
            based on medical imaging. Specifically, it can identify conditions such as Alzheimer's, brain tumors,
            and strokes. Our goal is to provide a preliminary analysis tool that can assist healthcare professionals
            and individuals in understanding brain scans more effectively.
        </div>
        """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Features of the App:")
        st.markdown("""
            - **Classify Your Brain Scan:** Upload a brain scan image, and the app will analyze it to detect
              possible brain diseases.
            - **Learn About Brain Diseases:** Get detailed information about various brain diseases,
              including symptoms, treatment options, and more.
        """)

        st.markdown("### How to Use:")
        st.markdown("""
            1. Navigate to the **Classify** page to upload a brain scan image.
            2. Read about different brain diseases in the **About Brain Diseases** section.
            3. After uploading an image, view the classification results along with a detailed report.
        """)

    with col3:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.image("brain1.jpg", width=500)  # Replace with your own image path

    st.markdown(
        """
        <div style="text-align: justify;">
            We hope this tool aids in the early detection and awareness of brain diseases. Please note that
            this app is not a substitute for professional medical advice, diagnosis, or treatment.
        </div>
        """, unsafe_allow_html=True)

# Function to render the classify page
def render_classify_page():
    st.header("Classify Your Brain Scan")

    patient_name = st.text_input("Patient Name:")
    patient_age = st.text_input("Patient Age:")

    test_type = st.selectbox(
        "Select the type of test you want to take:",
        ("Alzheimer's", "Brain Stroke", "Tumor")
    )

    uploaded_file = st.file_uploader("Upload your MRI scan image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None and patient_name and patient_age:
        st.image(uploaded_file, caption='Uploaded MRI scan.', use_column_width=True)
        image = load_image(uploaded_file)

        with st.spinner('Analyzing the MRI scan...'):
            model = MODELS[test_type]
            prediction = model.predict(image)
            prediction_label = np.argmax(prediction, axis=1)

        st.write(f"Prediction: {'Condition Positive' if prediction_label[0] == 1 else 'Condition Negative'}")

        report = generate_report(patient_name, patient_age, test_type, prediction_label)
        st.write(report)

        st.download_button(label="Download Report", data=report, file_name="medical_report.txt", mime='text/plain')

# Function to render the about page
def render_about_page():
    st.title("About Brain Diseases")

    st.write("Brain diseases affect millions of people each year. This section provides information on some of the most common conditions, including Alzheimer's disease, brain tumors, and strokes. Click on each section below to learn more.")

    with st.expander("Alzheimer's Disease"):
        st.image(r"alz.jpeg", caption="Alzheimer's Disease", width=300)
        st.write("""
        Alzheimer's disease is a progressive neurologic disorder that causes the brain to shrink (atrophy) and brain cells to die. Alzheimer's disease is the most common cause of dementia — a continuous decline in thinking, behavioral and social skills that affects a person's ability to function independently.
        """)
        st.write("""
                **Symptoms:** Memory loss, difficulty in planning or solving problems, difficulty completing familiar tasks, confusion with time or place, challenges in understanding visual images and spatial relationships, new problems with words in speaking or writing, misplacing things and losing the ability to retrace steps, decreased or poor judgment, withdrawal from work or social activities, changes in mood and personality.

                **Causes:** The exact causes of Alzheimer's disease are not fully understood, but a combination of genetic, lifestyle, and environmental factors that affect the brain over time are implicated.

                **Diagnosis:** Doctors conduct a series of tests to rule out other conditions, perform cognitive assessments, and neurological exams.

                **Treatment:** Treatments can temporarily slow the worsening of symptoms and improve quality of life for those with Alzheimer's disease and their caregivers.
                """)

    with st.expander("Brain Tumor"):
        st.image(r"tum.jpeg", caption="Brain Tumor", width=300)
        st.write("""
        A brain tumor is a mass or growth of abnormal cells in your brain. Many different types of brain tumors exist. Some brain tumors are noncancerous (benign), and some brain tumors are cancerous (malignant). Brain tumors can begin in your brain (primary brain tumors), or cancer can begin in other parts of your body and spread to your brain (secondary, or metastatic, brain tumors).
        """)
        st.write("""
                **Symptoms:** Headaches, seizures, nausea, vomiting, weakness or loss of movement in a part of the body, loss of balance, speech difficulties, confusion in everyday matters, personality or behavior changes, hearing problems.

                **Causes:** The causes of most brain tumors are unknown. Genetic factors, environmental exposures, or a combination thereof may play a role.

                **Diagnosis:** Brain tumors are diagnosed using MRI scans, CT scans, and, if necessary, biopsy.

                **Treatment:** Treatment options include surgery, radiation therapy, chemotherapy, targeted drug therapy, and immunotherapy, depending on the type, size, and location of the tumor.
                """)

    with st.expander("Stroke"):
        st.image(r"brain.png", caption="Stroke", width=300)
        st.write("""
        A stroke occurs when the blood supply to part of your brain is interrupted or reduced, preventing brain tissue from getting oxygen and nutrients. Brain cells begin to die in minutes. A stroke is a medical emergency, and prompt treatment is crucial. Early action can reduce brain damage and other complications.
        """)
        st.write("""
                **Symptoms:** Trouble speaking and understanding what others are saying, paralysis or numbness of the face, arm, or leg, problems seeing in one or both eyes, headache, trouble walking, dizziness, loss of balance or coordination.

                **Causes:** Blocked artery (ischemic stroke) or leaking or bursting of a blood vessel (hemorrhagic stroke) are the main causes of stroke.

                **Diagnosis:** Diagnosis involves medical history, physical exam, blood tests, CT scans, MRI, carotid ultrasound, cerebral angiogram, and echocardiogram.

                **Treatment:** Immediate treatment aims at restoring blood flow for an ischemic stroke or controlling bleeding for a hemorrhagic stroke. Long-term treatments focus on preventing future strokes and may include medication, surgery, and lifestyle changes.
                """)

if __name__ == '__main__':
    main()
