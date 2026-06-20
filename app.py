import cv2
import face_recognition
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Configuración de la interfaz de la página
st.set_page_config(page_title="Control de Acceso Biométrico", layout="centered")

# --- CONTROL DE ESTADO DE SESIÓN (LOGIN) ---
if "autenticado" not in s_state:
    st.session_state["autenticado"] = False
if "usuario_detectado" not in s_state:
    st.session_state["usuario_detectado"] = None

# --- VISTA 2: PANEL DE CONTROL DE ACCESO AUTORIZADO ---
if st.session_state["autenticado"]:
    st.success(f"🔓 ¡Acceso Concedido! Bienvenido al sistema, {st.session_state['usuario_detectado']}.")
    st.title("🖥️ Panel de Control de Sistemas")
    st.write("Has ingresado exitosamente mediante validación biométrica facial.")
    
    # Botón para cerrar sesión y volver a bloquear el sistema
    if st.button("🔴 Cerrar Sesión / Bloquear Sistema"):
        st.session_state["autenticado"] = False
        st.session_state["usuario_detectado"] = None
        st.rerun()

# --- VISTA 1: INTERFAZ DE LOGEO BIOMÉTRICO ---
else:
    st.title("🔒 Sistema Web de Control de Acceso")
    st.subheader("Por favor, mire a la cámara para validar su identidad")

    # 1. CARGAR ROSTROS DE REFERENCIA EN CACHÉ
    @st.cache_resource
    def cargar_rostros_referencia():
        try:
            imagen_jair = face_recognition.load_image_file("jair.jpg")
            jair_encodings = face_recognition.face_encodings(imagen_jair)[0]

            imagen_cataleya = face_recognition.load_image_file("cataleya.jpg")
            cataleya_encodings = face_recognition.face_encodings(imagen_cataleya)[0]
            
            return [jair_encodings, cataleya_encodings], ["JAIR", "CATALEYA"]
        except Exception as e:
            st.error(f"Error crítico al cargar las imágenes de referencia: {e}")
            return [], []

    rostros_conocidos_encodings, rostros_conocidos_nombres = cargar_rostros_referencia()

    # 2. PROCESADOR DE VIDEO Y LÓGICA DE DETECCIÓN
    class ValidadorAcceso(VideoTransformerBase):
        def transform(self, frame):
            cuadro = frame.to_ndarray(format="bgr24")
            cuadro = cv2.flip(cuadro, 1)

            # Reducción dimensional para rendimiento óptimo en la nube
            cuadro_pequeno = cv2.resize(cuadro, (0, 0), fx=0.25, fy=0.25)
            cuadro_rgb = cv2.cvtColor(cuadro_pequeno, cv2.COLOR_BGR2RGB)

            ubicaciones_rostros = face_recognition.face_locations(cuadro_rgb) 
            encodings_rostros = face_recognition.face_encodings(cuadro_rgb, ubicaciones_rostros)

            nombre_detectado = "DESCONOCIDO"

            for (superior, derecho, inferior, izquierdo), rostro_encoding in zip(ubicaciones_rostros, encodings_rostros):
                if len(rostros_conocidos_encodings) > 0:
                    distancias = face_recognition.face_distance(rostros_conocidos_encodings, rostro_encoding)
                    mejor_coincidencia_idx = np.argmin(distancias)
                    
                    if distancias[mejor_coincidencia_idx] < 0.6:
                        nombre_detectado = rostros_conocidos_nombres[mejor_coincidencia_idx]

                # Restaurar coordenadas
                superior *= 4
                derecho *= 4
                inferior *= 4
                izquierdo *= 4

                # Ajuste de Padding (Margen dinámico)
                alto_rostro = inferior - superior
                ancho_rostro = derecho - izquierdo
                padding_alto = int(alto_rostro * 0.25)
                padding_ancho = int(ancho_rostro * 0.20)

                alto_pantalla, ancho_pantalla, _ = cuadro.shape
                superior = max(0, superior - padding_alto)
                inferior = min(alto_pantalla, inferior + padding_alto)
                izquierdo = max(0, izquierdo - padding_ancho)
                derecho = min(ancho_pantalla, derecho + padding_ancho)

                # Renderizado de Interfaz Gráfica
                color = (0, 0, 255) if nombre_detectado == "DESCONOCIDO" else (0, 255, 0)
                cv2.rectangle(cuadro, (izquierdo, superior), (derecho, inferior), color, 2)
                cv2.rectangle(cuadro, (izquierdo, superior - 35), (derecho, superior), color, cv2.FILLED)
                cv2.putText(cuadro, nombre_detectado, (izquierdo + 6, superior - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

                # Guardar en el estado de Streamlit el usuario identificado en tiempo real
                if nombre_detectado != "DESCONOCIDO":
                    st.session_state["usuario_detectado"] = nombre_detectado

            return cuadro

    # 3. INTERFAZ GRÁFICA DE STREAMLIT
    if len(rostros_conocidos_encodings) > 0:
        # Streamer de video WebRTC
        webrtc_streamer(key="control-acceso", video_transformer_factory=ValidadorAcceso)
        
        # Validación dinámica del estado
        if st.session_state["usuario_detectado"] is not None:
            st.success(f"👤 Rostro identificado: {st.session_state['usuario_detectado']}")
            
            # Botón dinámico que solo aparece si se reconoció un rostro autorizado
            if st.button("🟢 Ingresar al Sistema Abierto"):
                st.session_state["autenticado"] = True
                st.rerun()
        else:
            st.warning("🔒 Estado: Esperando rostro autorizado...")
    else:
        st.error("Por favor, configure las imágenes de referencia en el servidor.")