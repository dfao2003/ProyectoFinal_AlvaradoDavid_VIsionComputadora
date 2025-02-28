from flask import Flask, request, Response
import cv2
import numpy as np
import json
import os

app = Flask(__name__)

# Cargar clasificadores en cascada desde un archivo JSON
def get_object_cascades(filename: str) -> dict:
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

# Cargar los clasificadores al iniciar el servidor
object_cascades = get_object_cascades('/home/davidalvarado/Escritorio/ProgramaFInal/data/cascades.json')

@app.route('/procesar-imagen', methods=['POST'])
def procesar_imagen():
    print("Petición recibida de:", request.remote_addr)
    print("Headers:", request.headers)
    print("Datos:", request.files)

    """Recibe una imagen, detecta objetos y devuelve la imagen procesada."""
    if 'image' not in request.files:
        return "No se envió ninguna imagen", 400

    file = request.files['image']
    image_np = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)  # Leer en COLOR

    if frame is None:
        print("Error al decodificar la imagen.")
        return "Error al decodificar la imagen", 400

    print("Imagen recibida con dimensiones:", frame.shape)  # Depuración

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises para la detección
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Aplicar detección con los clasificadores cargados
    for object_cascade_name, object_cascade in object_cascades.items():
        objects = object_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=16, minSize=(50, 50))
        for (x, y, w, h) in objects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Dibujar en imagen a color
            cv2.putText(frame, object_cascade_name, (x + 5, y - 10), font, 0.9, (255, 255, 255), 2)

    # Codificar la imagen de vuelta a JPEG
    _, buffer = cv2.imencode('.jpg', frame)

    print(f"Imagen procesada con tamaño: {len(buffer)} bytes.")  # Depuración
    return Response(buffer.tobytes(), mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
