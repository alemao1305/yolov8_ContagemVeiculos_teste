import cv2
import numpy as np
import torch
from ultralytics import YOLO
import json

# Carregar modelo YOLOv8 treinado para detectar carro, moto, ônibus e caminhão
model_path = 'yolov8n.pt'
model = YOLO(model_path)
model.to('cuda' if torch.cuda.is_available() else 'cpu')  # Usar GPU se disponível

# Carregar vídeo
cap = cv2.VideoCapture('video.mp4')

# Variáveis para a ROI
roi_x, roi_y, roi_w, roi_h = 0, 0, 640, 480
drawing = False
ix, iy = -1, -1

# Ajustar o limiar de confiança
confidence_threshold = 0.3

# Dicionário de nomes de classes
#class_names = {0: "carro", 1: "moto", 2: "ônibus", 3: "caminhão"}
class_names = {0: "cars", 1: "motorcycle", 2: "bus", 3: "truck"}

# Inicializar lista para armazenar detecções
detections = []

# Inicializar contadores de veículos
vehicle_count = {"cars": 0, "motorcycle": 0, "bus": 0, "truck": 0}

# Função para verificar se um veículo cruzou a linha de contagem
def check_line_crossing(center_y, line_y, direction):
    if direction == 'up' and center_y < line_y:
        return True
    elif direction == 'down' and center_y > line_y:
        return True
    return False

# Função de callback do mouse para desenhar a ROI
def draw_roi(event, x, y, flags, param):
    global roi_x, roi_y, roi_w, roi_h, drawing, ix, iy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            frame_copy = frame.copy()
            cv2.rectangle(frame_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('Video', frame_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi_x, roi_y, roi_w, roi_h = ix, iy, x - ix, y - iy

# Coordenadas da linha de contagem (horizontal)
line_y = roi_y + roi_h // 2

# Sentido de contagem: 'up' para cima, 'down' para baixo
count_direction = 'down'

cv2.namedWindow('Video')
cv2.setMouseCallback('Video', draw_roi)

while True:
    # Capturar frame do vídeo
    ret, frame = cap.read()

    if not ret:
        break

    # Converter frame para RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Extrair ROI
    roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    # Detectar objetos na ROI
    results = model(roi)

    # Verificar se há resultados de detecção
    if results and results[0].boxes:
        # Processar resultados da detecção
        for detection in results[0].boxes:
            # Obter informações da detecção
            bbox = detection.xyxy[0]  # Coordenadas da caixa delimitadora
            confidence = detection.conf[0]  # Probabilidade de detecção
            class_id = int(detection.cls[0])  # Classe do objeto

            # Filtrar por confiança e classe
            if confidence >= confidence_threshold and class_id in [0, 1, 2, 3]:  # Filtrar por classes de interesse (carro, moto, ônibus, caminhão)
                # Converter coordenadas da caixa delimitadora para ROI
                bbox_x1 = int(bbox[0])
                bbox_y1 = int(bbox[1])
                bbox_x2 = int(bbox[2])
                bbox_y2 = int(bbox[3])

                # Verificar se o centro do bounding box cruzou a linha de contagem
                bbox_center_y = (bbox_y1 + bbox_y2) // 2
                if check_line_crossing(bbox_center_y, line_y, count_direction):
                    class_name = class_names[class_id]
                    vehicle_count[class_name] += 1

                # Desenhar boxes de detecção
                cv2.rectangle(roi, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), (0, 255, 0), 2)

                # Obter nome da classe
                class_name = class_names[class_id]

                # Criar dicionário de detecção
                detection_data = {
                    "class_name": class_name,
                    "confidence": float(confidence),
                    "bbox": [bbox_x1, bbox_y1, bbox_x2, bbox_y2]
                }

                # Adicionar dicionário de detecção à lista
                detections.append(detection_data)

    # Desenhar retângulo da ROI no frame original
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), 2)

    # Desenhar linha de contagem
    line_y = roi_y + roi_h // 2
    cv2.line(frame, (roi_x, line_y), (roi_x + roi_w, line_y), (255, 0, 0), 2)

    # Adicionar contagem de veículos ao frame
    y_offset = 30
    for vehicle, count in vehicle_count.items():
        text = f"{vehicle}: {count}"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30

    # Converter frame de volta para BGR para exibição
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Exibir o frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Salvar detecções em arquivo JSON
    with open('detections.json', 'w') as f:
        json.dump(detections, f, indent=4)

    # Limpar lista de detecções para o próximo frame
    detections = []

# Fechar janela e liberar recursos
cv2.destroyAllWindows()
cap.release()

# Exibir contagem de veículos
print("Contagem de veículos:")
print(json.dumps(vehicle_count, indent=4))
