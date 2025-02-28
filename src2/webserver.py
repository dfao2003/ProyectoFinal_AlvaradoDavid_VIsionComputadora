from flask import Flask, render_template, Response
import cv2
import numpy as np
import psutil  # Para medir el uso de memoria

app = Flask(__name__)

reference_path = "/home/davidalvarado/Escritorio/ProgramaFInal/src2/images/sift/cuenca.jpeg"
reference_image = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT_create()


keypoints_ref, descriptors_ref = sift.detectAndCompute(reference_image, None)
reference_color = cv2.cvtColor(reference_image, cv2.COLOR_GRAY2BGR)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

def generar_frames():
    cap = cv2.VideoCapture(0)
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
            memory_usage = psutil.virtual_memory().percent  # % de memoria usada

            # Convertir frame a escala de grises para SIFT
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detectar keypoints y descriptores en el frame
            keypoints_frame, descriptors_frame = sift.detectAndCompute(frame_gray, None)

            if descriptors_frame is not None and len(descriptors_frame) > 0:
                matches = bf.match(descriptors_ref, descriptors_frame)
                matches = sorted(matches, key=lambda x: x.distance)

                category_matches = sum(1 for m in matches if m.distance < 100)

                if category_matches > 5:
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

            # Convertir a JPEG y enviar el frame
            ret, buffer = cv2.imencode('.jpg', matched_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:
            print(f"Error procesando el frame: {e}")

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
 