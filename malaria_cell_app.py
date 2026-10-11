import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Sayfa Yapılandırması
st.set_page_config(
    page_title="Sıtma Hücresi Tespiti Uygulaması", page_icon="🔬", layout="centered"
)


# Eğitilmiş modeli yükleme
@st.cache_resource
def load_malaria_model():
  model = load_model("malaria_model.h5")
  return model


st.title("🔬 Sıtma (Malaria) Hücresi Tespit Sistemi")
st.write(
    "Bu uygulama, kan hücresi mikroskop görüntüsünü analiz ederek hücrenin"
    " **Sağlıklı (Uninfected)** mi yoksa **Parazitli (Parasitized)** mi olduğunu"
    " tespit eder[cite: 7]."
)

uploaded_file = st.file_uploader(
    "Bir kan hücresi görüntüsü yükleyin (JPG, PNG)...",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
  image_pil = Image.open(uploaded_file)
  st.image(image_pil, caption="Yüklenen Hücre Görüntüsü", use_column_width=True)

  if st.button("Hücreyi Analiz Et"):
    with st.spinner("Model analiz yapıyor..."):
      image_resized = image_pil.resize((64, 64))
      img_array = image.img_to_array(image_resized)
      img_array = np.expand_dims(img_array, axis=0) / 255.0

      try:
        model = load_malaria_model()
        prediction = model.predict(img_array)[0][0]

        st.subheader("Analiz Sonucu:")
        if prediction > 0.5:
          st.success(
              f"✅ **Sağlıklı (Uninfected)** hücresi görünmektedir. (Güven Oranı:"
              f" %{prediction * 100:.2f})"
          )
          st.balloons()
        else:
          st.error(
              f"❌ **DİKKAT! Parazitli / Sıtma Enfeksiyonlu (Parasitized)**"
              f" hücre tespit edildi! (Risk Olasılığı:"
              f" %{(1 - prediction) * 100:.2f})"
          )
      except Exception as e:
        st.error(
            "Eğitilmiş model dosyası ('malaria_model.h5') bulunamadı! Lütfen"
            " önce modeli eğittiğinizden emin olun."
        )