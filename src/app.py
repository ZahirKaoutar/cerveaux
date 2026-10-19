
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# 1. Configuration de la page
st.set_page_config(
    page_title="BrainScan AI",
    page_icon="🧠",
    layout="centered"
)

# 2. Charger le modèle une seule fois
@st.cache_resource
def load_cnn():
    return load_model("models/brainscan_cnn.keras")

model = load_cnn()

# 3. Les classes dans le même ordre que pendant l'entraînement
classes = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

# 4. Fonction de prédiction
def predict_image(image):
    # Convertir l'image en RGB
    image = image.convert("RGB")

    # Redimensionner à 224 x 224
    image = image.resize((224, 224))

    # Convertir en tableau NumPy
    image = np.array(image, dtype=np.float32)

    # Normaliser les pixels entre 0 et 1
    image = image / 255.0

    # Ajouter la dimension du batch
    image = np.expand_dims(image, axis=0)

    # Prédiction du modèle
    probabilities = model.predict(image, verbose=0)[0]

    # Classe la plus probable
    index = np.argmax(probabilities)

    predicted_class = classes[index]
    confidence = float(probabilities[index])

    return predicted_class, confidence, probabilities


# 5. Interface Streamlit
st.title("🧠 BrainScan AI")
st.subheader("Classification des images IRM cérébrales")

st.write(
    "Importez une image IRM pour tester la prédiction "
    "de votre modèle CNN."
)

uploaded_file = st.file_uploader(
    "Choisir une image IRM",
    type=["png", "jpg", "jpeg"]
)

# 6. Traiter l'image importée
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Image IRM importée",
        width=350
    )

    if st.button("Analyser l'image", type="primary"):

        with st.spinner("Analyse en cours..."):

            classe, confiance, probabilities = predict_image(image)

        st.divider()

        st.subheader("Résultat de la prédiction")

        st.write("Classe prédite :")

        st.success(classe)

        st.metric(
            label="Probabilité associée à la classe prédite",
            value=f"{confiance * 100:.2f}%"
        )

        st.subheader("Probabilités des quatre classes")

        for nom_classe, proba in zip(classes, probabilities):
            st.write(f"**{nom_classe}** : {proba * 100:.2f}%")
            st.progress(float(proba))

        st.warning(
            "Cette application est un prototype éducatif. "
            "La prédiction du modèle ne constitue pas un diagnostic "
            "médical. Toute interprétation doit être réalisée "
            "par un professionnel de santé."
        )
else:
    st.info("Veuillez importer une image pour commencer.")
