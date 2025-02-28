
from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import psutil
import io

app = Flask(__name__)

# Ruta de la imagen de referencia que se utilizará para la comparación
reference_path = "/home/davidalvarado/Escritorio/ProgramaFInal/src3/images/sift/ecuador.jpeg"
reference_image = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT_create()
keypoints_ref, descriptors_ref = sift.detectAndCompute(reference_image, None)

# Convertir la imagen de referencia a color para dibujar las coincidencias
reference_color = cv2.cvtColor(reference_image, cv2.COLOR_GRAY2BGR)

# Usar el matcher de características para comparar las imágenes
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Función para procesar y generar frames
def generar_frames():
    # Obtener el video de la cámara (puede ser una cámara USB o webcam)
    cap = cv2.VideoCapture(0)  # O puedes poner la ruta de un archivo de video si prefieres
    prev_time = cv2.getTickCount()
    fps = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        try:
            # Medir FPS
            current_time = cv2.getTickCount()
            time_diff = (current_time - prev_time) / cv2.getTickFrequency()
            prev_time = current_time
            fps = 1 / time_diff if time_diff > 0 else 0

            # Medir uso de memoria
            memory_usage = psutil.virtual_memory().percent

            # Convertir el frame a escala de grises para la detección de características con SIFT
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detectar keypoints y descriptores en el frame
            keypoints_frame, descriptors_frame = sift.detectAndCompute(frame_gray, None)

            if descriptors_frame is not None and len(descriptors_frame) > 0:
                matches = bf.match(descriptors_ref, descriptors_frame)
                matches = sorted(matches, key=lambda x: x.distance)

                # Filtrar los matches con una distancia menor a un umbral
                category_matches = sum(1 for m in matches if m.distance < 100)

                if category_matches > 5:
                    # Dibujar las coincidencias en el frame
                    matched_frame = cv2.drawMatches(
                        reference_color, keypoints_ref,
                        frame, keypoints_frame,
                        matches[:35], None,
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                    )

                    cv2.putText(matched_frame, "Categoria detectada", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    matched_frame = frame.copy()
                    cv2.putText(matched_frame, "Categoria desconocida", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                matched_frame = frame.copy()

            # Mostrar FPS y uso de memoria en el video
            cv2.putText(matched_frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(matched_frame, f"Memoria: {memory_usage:.1f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Convertir el frame procesado a JPEG
            ret, buffer = cv2.imencode('.jpg', matched_frame)
            matched_frame = buffer.tobytes()

            # Enviar el frame procesado como parte de un flujo multipart
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:
            print(f"Error procesando el frame: {e}")

    cap.release()

# Ruta para el index HTML
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para la alimentación de video
@app.route('/video_feed')
def video_feed():
    return Response(generar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Ruta para procesar la imagen recibida desde Android
@app.route('/procesar-imagen', methods=['POST'])
def procesar_imagen():
    try:
        # Obtener la imagen enviada desde Android
        file = request.files['image']
        img_bytes = file.read()
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Convertir la imagen recibida a escala de grises
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detectar keypoints y descriptores de la imagen recibida
        keypoints_img, descriptors_img = sift.detectAndCompute(img_gray, None)

        if descriptors_img is not None and len(descriptors_img) > 0:
            matches = bf.match(descriptors_ref, descriptors_img)
            matches = sorted(matches, key=lambda x: x.distance)

            category_matches = sum(1 for m in matches if m.distance < 100)

            if category_matches > 5:
                matched_img = cv2.drawMatches(
                    reference_color, keypoints_ref,
                    img, keypoints_img,
                    matches[:35], None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )

                cv2.putText(matched_img, "Categoria detectada", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                matched_img = img.copy()
                cv2.putText(matched_img, "Categoria desconocida", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Convertir la imagen procesada a JPEG para enviarla de vuelta
            _, buffer = cv2.imencode('.jpg', matched_img)
            return Response(buffer.tobytes(), mimetype='image/jpeg')
        else:
            return "No se encontraron descriptores en la imagen enviada.", 400

    except Exception as e:
        return f"Error procesando la imagen: {e}", 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
