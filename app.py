from ultralytics import YOLO
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from waitress import serve
from PIL import Image
import os

# --------------------------------------
# APP
# --------------------------------------

app = Flask(__name__)

# CORS GLOBAL
CORS(
    app,
    origins="*",
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "OPTIONS"]
)

# --------------------------------------
# MODELO
# --------------------------------------

MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modelo '{MODEL_PATH}' não encontrado.")

model = YOLO(MODEL_PATH)

# --------------------------------------
# ROTAS
# --------------------------------------

# @app.route("/", methods=["GET"])
# def health():
#     return jsonify({
#         "service": "Dentiscan AI",
#         "status": "online"
#     })

@app.route("/")
def root():
    """Retorna a interface HTML."""
    return send_from_directory("templates", "index.html")


@app.route("/detect", methods=["POST"])
def detect():
    if "image_file" not in request.files:
        return jsonify({"error": "Nenhuma imagem enviada"}), 400

    file = request.files["image_file"]

    try:
        results = detect_objects_on_image(file.stream)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------------------------
# DETECÇÃO
# --------------------------------------

def detect_objects_on_image(buf):
    img = Image.open(buf).convert("RGB")

    results = model.predict(img)
    result = results[0]

    output = []

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_id = int(box.cls[0])
        conf = float(box.conf[0])

        output.append([
            x1,
            y1,
            x2,
            y2,
            result.names[class_id],
            round(conf, 4)
        ])

    return output

# --------------------------------------
# SERVER
# --------------------------------------

# if __name__ == "__main__":
#     serve(app, host="0.0.0.0", port=8080)

if __name__ == "__main__":
    print("Servidor rodando em http://127.0.0.1:8080")
    serve(app, host='0.0.0.0', port=8080)


# from ultralytics import YOLO
# from flask import request, Flask, jsonify, send_from_directory
# from flask_cors import CORS
# from waitress import serve
# from PIL import Image
# import os

# # --------------------------------------
# # CONFIGURAÇÕES
# # --------------------------------------

# app = Flask(__name__)

# # CORS GLOBAL
# CORS(
#     app,
#     origins="*",
#     allow_headers=["Content-Type", "Authorization"],
#     methods=["GET", "POST", "OPTIONS"]
# )

# # --------------------------------------
# # MODELO
# # --------------------------------------

# MODEL_PATH = "best.pt"

# # Carrega o modelo apenas 1 vez (otimização)
# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(
#         f"O arquivo '{MODEL_PATH}' não foi encontrado. "
#         f"Coloque um modelo YOLO válido na raiz do projeto."
#     )

# model = YOLO(MODEL_PATH)


# # --------------------------------------
# # ROTAS
# # --------------------------------------

# @app.route("/")
# def root():
#     """Retorna a interface HTML."""
#     return send_from_directory("templates", "index.html")


# @app.route("/detect", methods=["POST"])
# def detect():
#     """
#     Recebe a imagem enviada pelo frontend,
#     executa o YOLO e retorna os bounding boxes.
#     """
#     if "image_file" not in request.files:
#         return jsonify({"error": "Nenhuma imagem enviada"}), 400

#     file = request.files["image_file"]

#     try:
#         boxes = detect_objects_on_image(file.stream)
#         return jsonify(boxes)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# # --------------------------------------
# # FUNÇÃO PRINCIPAL DE DETECÇÃO
# # --------------------------------------

# def detect_objects_on_image(buf):
#     """Executa o modelo YOLO na imagem enviada."""

#     img = Image.open(buf).convert("RGB")

#     # Faz predição
#     results = model.predict(img)
#     result = results[0]

#     output = []

#     for box in result.boxes:
#         x1, y1, x2, y2 = [int(x) for x in box.xyxy[0].tolist()]
#         class_id = int(box.cls[0])
#         prob = float(box.conf[0])

#         output.append([
#             x1,
#             y1,
#             x2,
#             y2,
#             result.names[class_id],
#             f"{prob * 100:.2f}%"
#         ])

#     return output


# # --------------------------------------
# # INICIALIZAÇÃO DO SERVIDOR
# # --------------------------------------

# if __name__ == "__main__":
#     print("Servidor rodando em http://127.0.0.1:8080")
#     serve(app, host='0.0.0.0', port=8080)
