import cv2
import face_recognition
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title="Biometría Facial - Proyecto Final", layout="centered")
st.title("🛡️ Sistema de Reconocimiento Facial en Tiempo Real")
st.subheader("Proyecto Final de Machine Learning")

# 1. CARGAR FOTOS DE REFERENCIA (Caché para que Render no las recargue en cada fotograma)
@st.cache_resource
def cargar_rostros_referencia():
    try:
        imagen_jair = face_recognition.load_image_file("jair.jpg")
        jair_encodings = face_recognition.face_encodings(imagen_jair)[0]

        imagen_cataleya = face_recognition.load_image_file("cataleya.jpg")
        cataleya_encodings = face_recognition.face_encodings(imagen_cataleya)[0]
        
        encodings = [jair_encodings, cataleya_encodings]
        nombres = ["JAIR", "CATALEYA"]
        return encodings, nombres
    except Exception as e:
        st.error(f"Error al cargar imágenes de referencia: {e}")
        return [], []

rostros_conocidos_encodings, rostros_conocidos_nombres = cargar_rostros_referencia()

# 2. CLASE PROCESADORA DE VIDEO (Se ejecuta en el servidor de Render)
class AnalizadorFacial(VideoTransformerBase):
    def transform(self, frame):
        # Convertir el fotograma de WebRTC a una matriz NumPy de OpenCV (BGR)
        cuadro = frame.to_ndarray(format="bgr24")
        
        # Efecto espejo natural
        cuadro = cv2.flip(cuadro, 1)

        # Reducir el tamaño a 1/4 para que el procesamiento en Render sea ultra rápido
        cuadro_pequeno = cv2.resize(cuadro, (0, 0), fx=0.25, fy=0.25)
        cuadro_rgb = cv2.cvtColor(cuadro_pequeno, cv2.COLOR_BGR2RGB)

        # Detectar y codificar rostros
        ubicaciones_rostros = face_recognition.face_locations(cuadro_rgb) 
        encodings_rostros = face_recognition.face_encodings(cuadro_rgb, ubicaciones_rostros)

        for (superior, derecho, inferior, izquierdo), rostro_encoding in zip(ubicaciones_rostros, encodings_rostros):
            if len(rostros_conocidos_encodings) > 0:
                distancias = face_recognition.face_distance(rostros_conocidos_encodings, rostro_encoding)
                mejor_coincidencia_idx = np.argmin(distancias)
                
                if distancias[mejor_coincidencia_idx] < 0.6:
                    nombre = rostros_conocidos_nombres[mejor_coincidencia_idx]
                else:
                    nombre = "DESCONOCIDO"
            else:
                nombre = "DESCONOCIDO"

            # Reescalar coordenadas al tamaño nativo de la cámara
            superior *= 4
            derecho *= 4
            inferior *= 4
            izquierdo *= 4

            # --- AJUSTE DE MARGEN (PADDING) ---
            alto_rostro = inferior - superior
            ancho_rostro = derecho - izquierdo
            padding_alto = int(alto_rostro * 0.25)
            padding_ancho = int(ancho_rostro * 0.20)

            alto_pantalla, ancho_pantalla, _ = cuadro.shape
            superior = max(0, superior - padding_alto)
            inferior = min(alto_pantalla, inferior + padding_alto)
            izquierdo = max(0, izquierdo - padding_ancho)
            derecho = min(ancho_pantalla, derecho + padding_ancho)

            # --- INTERFAZ GRÁFICA ---
            color = (0, 0, 255) if nombre == "DESCONOCIDO" else (0, 255, 0)
            cv2.rectangle(cuadro, (izquierdo, superior), (derecho, inferior), color, 2)
            cv2.rectangle(cuadro, (izquierdo, superior - 35), (derecho, superior), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(cuadro, nombre, (izquierdo + 6, superior - 10), font, 0.8, (255, 255, 255), 1)

        return cuadro

# 3. INICIAR STREAMING WEB
if len(rostros_conocidos_encodings) > 0:
    st.write("✅ Sistema listo. Haz clic en 'Start' para encender tu cámara.")
    webrtc_streamer(key="face-recognition", video_transformer_factory=AnalizadorFacial)
else:
    st.warning("Asegúrate de subir las fotos de referencia al repositorio.")
