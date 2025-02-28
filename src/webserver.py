from flask import Flask, Response, render_template
import cv2
import json
import os

app = Flask(__name__)

def get_object_cascades(filename: str) -> dict:
    """Carga clasificadores en cascada desde un archivo JSON."""
    object_cascades = {}
    with open(filename, 'r') as fs:
        object_cascades = json.load(fs)
    
    if not object_cascades:
        raise ValueError('Load cascades into cascades.json.')

    for object_cascade_name, object_cascade_path in object_cascades.items():
        if os.path.exists(object_cascade_path):
            object_cascades[object_cascade_name] = cv2.CascadeClassifier(object_cascade_path)
            print(f"Cargado clasificador para: {object_cascade_name}")
        else:
            print(f"Error: No se encontró el archivo para {object_cascade_name} en {object_cascade_path}")
    
    return object_cascades

def generar_frames():
    """Captura video en tiempo real desde la cámara de la computadora y detecta objetos."""
    cap = cv2.VideoCapture(0)  # Usa la cámara integrada

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    object_cascades = get_object_cascades('/home/davidalvarado/Escritorio/ProgramaFInal/data/cascades.json')
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for object_cascade_name, object_cascade in object_cascades.items():
            scaleFactor = 1.08
            minNeighbors = 16

            objects = object_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=(30, 30))

            for (x, y, w, h) in objects:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, object_cascade_name, (x + 5, y - 10), font, 0.9, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    """Renderiza la página principal."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Transmite el video procesado con detección en tiempo real."""
    return Response(generar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

