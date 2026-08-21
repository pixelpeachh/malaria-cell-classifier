import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Malaria Cell Classifier",
    page_icon="🩸",
    layout="centered"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

/* =========================================================
   GLOBAL
   ========================================================= */

[data-testid="stAppViewContainer"] {
    background-color: #0B1117;
}

[data-testid="stHeader"] {
    background-color: transparent;
}

.block-container {
    max-width: 1000px;
    padding-top: 3rem;
    padding-bottom: 2rem;
}


/* =========================================================
   TITLE
   ========================================================= */

.main-title {
    text-align: center;
    color: #F1F5F9;

    font-size: 42px;
    font-weight: 700;

    line-height: 1.2;

    margin-top: 10px;
    margin-bottom: 6px;
}

.subtitle {
    text-align: center;
    color: #94A3B8;

    font-size: 16px;

    margin-bottom: 35px;
}


/* =========================================================
   DISCLAIMER
   ========================================================= */

.disclaimer {
    background-color: #151F28;

    border: 1px solid #334155;
    border-left: 4px solid #F59E0B;

    border-radius: 12px;

    padding: 17px 20px;

    margin: 0 auto 38px auto;

    max-width: 850px;

    text-align: left;
}

.disclaimer-title {
    color: #FBBF24;

    font-size: 15px;
    font-weight: 600;

    margin-bottom: 7px;
}

.disclaimer-text {
    color: #CBD5E1;

    font-size: 13px;
    line-height: 1.65;
}


/* =========================================================
   UPLOAD SECTION
   ========================================================= */

.section-title {
    text-align: center;

    color: #F1F5F9;

    font-size: 21px;
    font-weight: 600;

    margin-bottom: 7px;
}

.section-description {
    text-align: center;

    color: #94A3B8;

    font-size: 14px;

    margin-bottom: 18px;
}


/* =========================================================
   UPLOADER
   ========================================================= */

[data-testid="stFileUploader"] {
    background-color: #151F28;

    border: 1px solid #263541;

    border-radius: 12px;

    padding: 8px;

    max-width: 700px;

    margin: 0 auto;
}


/* =========================================================
   IMAGE + RESULT AREA
   ========================================================= */

.image-result-wrapper {
    margin-top: 30px;
}


/* =========================================================
   IMAGE CARD
   ========================================================= */

.image-title {
    color: #94A3B8;

    font-size: 12px;
    font-weight: 600;

    letter-spacing: 1.5px;
    text-transform: uppercase;

    margin-bottom: 15px;
}


/* =========================================================
   PREDICT BUTTON
   ========================================================= */

.predict-area {
    display: flex;
    justify-content: center;

    margin-top: 20px;
    margin-bottom: 5px;
}

div.stButton {
    display: flex;
    justify-content: center;
}

div.stButton > button {
    background-color: #151F28;

    color: #F1F5F9;

    border: 1px solid #475569;

    border-radius: 10px;

    padding: 8px 24px;

    font-size: 15px;
    font-weight: 600;

    min-width: 125px;

    transition: all 0.2s ease;
}

div.stButton > button:hover {
    background-color: #17252D;

    color: #2DD4BF;

    border-color: #2DD4BF;
}


/* =========================================================
   RESULT CARD
   ========================================================= */

.result-card {
    background: linear-gradient(
        145deg,
        #151F28,
        #101820
    );

    border: 1px solid #263541;

    border-radius: 16px;

    padding: 32px 25px;

    min-height: 390px;

    display: flex;
    flex-direction: column;

    justify-content: center;
    align-items: center;

    text-align: center;

    box-shadow:
        0 8px 25px rgba(0, 0, 0, 0.25);
}

.result-label {
    color: #94A3B8;

    font-size: 12px;
    font-weight: 600;

    letter-spacing: 2px;
    text-transform: uppercase;

    margin-bottom: 14px;
}

.result-icon {
    font-size: 34px;

    margin-bottom: 7px;
}

.result-value {
    font-size: 29px;
    font-weight: 700;

    margin-bottom: 28px;
}

.confidence-label {
    color: #94A3B8;

    font-size: 13px;

    margin-bottom: 5px;
}

.confidence-value {
    color: #2DD4BF;

    font-size: 29px;
    font-weight: 700;

    margin-bottom: 10px;
}


/* =========================================================
   CONFIDENCE BAR
   ========================================================= */

.confidence-bar {
    width: 85%;

    height: 8px;

    background-color: #263541;

    border-radius: 20px;

    overflow: hidden;

    margin-top: 8px;
}

.confidence-fill {
    height: 100%;

    border-radius: 20px;

    background-color: #2DD4BF;
}


/* =========================================================
   RESULT DESCRIPTION
   ========================================================= */

.result-description {
    color: #94A3B8;

    font-size: 12px;

    line-height: 1.5;

    margin-top: 18px;

    max-width: 320px;
}


/* =========================================================
   FOOTER
   ========================================================= */

.footer {
    text-align: center;

    color: #64748B;

    font-size: 12px;

    margin-top: 45px;
    margin-bottom: 10px;
}


/* =========================================================
   MOBILE
   ========================================================= */

@media (max-width: 700px) {

    .main-title {
        font-size: 32px;
    }

    .subtitle {
        font-size: 14px;
    }

    .image-card,
    .result-card {
        min-height: auto;
    }

}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🩸 Malaria Cell Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-assisted classification of malaria blood-cell images'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# MODEL SETUP
# ============================================================

device = torch.device("cpu")


# Locate the model relative to app.py
model_path = (
    Path(__file__).resolve().parent
    / "malaria_resnet18.pth"
)


# Check that model exists
if not model_path.exists():

    st.error(
        "Model file 'malaria_resnet18.pth' was not found "
        "in the same folder as app.py."
    )

    st.stop()


# Create ResNet18 architecture
model = models.resnet18(
    weights=None
)


# Replace original classification layer
# with our two-class classifier
model.fc = nn.Linear(
    model.fc.in_features,
    2
)


# Load trained weights
model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)


# Move model to CPU
model = model.to(device)


# Evaluation mode
model.eval()


# ============================================================
# PREPROCESSING
# ============================================================

# EXACT MEAN VALUES CALCULATED FROM OUR TRAINING DATA
mean_values = [
    0.5297750234603882,
    0.4240841567516327,
    0.4531458020210266
]


# EXACT STANDARD DEVIATION VALUES
std_values = [
    0.32734930515289307,
    0.2637665569782257,
    0.27798572182655334
]


# Same preprocessing used for validation/test images
predict_transform = transforms.Compose([
    transforms.Resize((128, 128)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=mean_values,
        std=std_values
    )
])


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Upload a Cell Image'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Upload a blood-smear cell image in JPG, JPEG, or PNG format.'
    '</div>',
    unsafe_allow_html=True
)


# Center uploader using outer columns
upload_left, upload_center, upload_right = st.columns(
    [1, 2, 1]
)


with upload_center:

    uploaded_file = st.file_uploader(
        "Choose a cell image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

# ============================================================
# IMAGE + PREDICTION
# ============================================================

if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # ========================================================
    # IMAGE + RESULT COLUMNS
    # ========================================================

    image_column, result_column = st.columns(
        [1, 1],
        gap="large",
        vertical_alignment="center"
    )


    # ========================================================
    # LEFT COLUMN — IMAGE + BUTTON
    # ========================================================

    with image_column:

        # Image card
        with st.container(
            border=True
        ):

            st.markdown(
                """
<div class="image-title">
UPLOADED CELL IMAGE
</div>
""",
                unsafe_allow_html=True
            )

            # Center the image
            img_left, img_center, img_right = st.columns(
                [0.2, 1, 0.2]
            )

            with img_center:

                st.image(
                    image,
                    width=300
                )


            # ------------------------------------------------
            # CENTERED PREDICT BUTTON
            # ------------------------------------------------

            btn_left, btn_center, btn_right = st.columns(
                [1, 1, 1]
            )

            with btn_center:

                predict_clicked = st.button(
                    "🔍 Predict",
                    width="content"
                )


    # ========================================================
    # RIGHT COLUMN — RESULT
    # ========================================================

    with result_column:

        # Default state before prediction

        prediction = None
        confidence = None


        # ====================================================
        # PREDICTION
        # ====================================================

        if predict_clicked:

            # Apply EXACT same preprocessing
            # used during testing

            image_tensor = predict_transform(
                image
            )


            # Add batch dimension
            #
            # 3 × 128 × 128
            #        ↓
            # 1 × 3 × 128 × 128

            image_tensor = image_tensor.unsqueeze(0)


            # Move tensor to model device
            image_tensor = image_tensor.to(device)


            # ------------------------------------------------
            # MODEL INFERENCE
            # ------------------------------------------------

            with torch.no_grad():

                outputs = model(
                    image_tensor
                )


                # Convert logits into probabilities

                probabilities = torch.softmax(
                    outputs,
                    dim=1
                )


                # Get highest probability
                # and corresponding class

                confidence, predicted_class = torch.max(
                    probabilities,
                    dim=1
                )


            # ------------------------------------------------
            # CLASS NAMES
            # ------------------------------------------------

            class_names = [
                "Parasitized",
                "Uninfected"
            ]


            prediction = class_names[
                predicted_class.item()
            ]


            # Convert probability
            # into percentage

            confidence = (
                confidence.item() * 100
            )


        # ====================================================
        # RESULT CARD
        # ====================================================

        if prediction is not None:

            if prediction == "Uninfected":

                result_color = "#4ADE80"
                result_icon = "✓"

                description = (
                    "The model classified the uploaded "
                    "cell image as uninfected."
                )

            else:

                result_color = "#FB7185"
                result_icon = "⚠"

                description = (
                    "The model classified the uploaded "
                    "cell image as parasitized."
                )


            st.markdown(
                f"""
<div class="result-card">

<div class="result-label">
ANALYSIS RESULT
</div>

<div
class="result-icon"
style="color: {result_color};">
{result_icon}
</div>

<div
class="result-value"
style="color: {result_color};">
{prediction.upper()}
</div>

<div class="confidence-label">
Model Confidence
</div>

<div class="confidence-value">
{confidence:.2f}%
</div>

<div class="confidence-bar">

<div
class="confidence-fill"
style="width: {confidence:.2f}%;">
</div>

</div>

<div class="result-description">
{description}
</div>

</div>
""",
                unsafe_allow_html=True
            )


        # ====================================================
        # BEFORE PREDICTION
        # ====================================================

        else:

            st.markdown(
                """
<div class="result-card">

<div class="result-label">
ANALYSIS RESULT
</div>

<div
class="result-value"
style="color: #64748B;">
Awaiting Prediction
</div>

<div class="result-description">
Click the Predict button to analyze the uploaded
cell image using the trained ResNet18 model.
</div>

</div>
""",
                unsafe_allow_html=True
            )

st.markdown(
    """
<div class="footer">
Malaria Cell Classifier &nbsp;•&nbsp;
ResNet18 &nbsp;•&nbsp;
Deep Learning Project
</div>
""",
    unsafe_allow_html=True
)

# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    """
<div class="disclaimer">
<div class="disclaimer-title">
⚠️ Educational &amp; Research Use Only
</div>

<div class="disclaimer-text">
This application is not a medical diagnostic tool and should not be
used for diagnosis, treatment, or clinical decision-making.
Model predictions may be incorrect. Please consult a qualified
healthcare professional for medical advice.
</div>
</div>
""",
    unsafe_allow_html=True
)
