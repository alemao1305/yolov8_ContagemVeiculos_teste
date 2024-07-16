import cv2
import numpy as np
import torch
from ultralytics import YOLO
import json
import qrcode
from tkinter import messagebox, Tk, Label, Entry, Button
import datetime
import sqlite3
import schedule
from collections import deque
from diretorio_db import caminho2
from scipy.spatial import distance as dist

# Variável global para armazenar a URL
url1 = ""

def gera_qr_code():
    global url1

    url = website_entry.get()

    if not url:
        messagebox.showinfo(title="Erro!", message="Favor insira uma URL válida")
        return

    opcao_escolhida = messagebox.askokcancel(
        title=url,
        message=f"O endereço URL é: \n Endereço: {url} \n Pronto para salvar?"
    )

    if opcao_escolhida:
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color='black', back_color='white')
        img.save('qrExport.png')
        url1 = url
        window.destroy()

def create_tkinter_interface():
    global window, website_entry
    window = Tk()
    window.title("Gerador de Código QR")
    window.config(padx=10, pady=100)

    Label(window, text="URL:").grid(row=2, column=0)
    website_entry = Entry(window, width=35)
    website_entry.grid(row=2, column=1, columnspan=2)
    website_entry.focus()

    add_button = Button(window, text="IP da Câmera", width=36, command=gera_qr_code)
    add_button.grid(row=4, column=1, columnspan=2)

    window.mainloop()

if __name__ == '__main__':
    create_tkinter_interface()

# Carregar modelo YOLOv8 treinado para detectar carro, moto, ônibus e caminhão
model_path = 'yolov8n.pt'
model = YOLO(model_path)
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Capturar vídeo
cap = cv2.VideoCapture('video.mp4')

# Variáveis para controle do envio de dados para o banco
last_db_update_time = datetime.datetime.now()

# Conexão com o banco de dados SQLite
con = sqlite3.connect(caminho2)
cur = con.cursor()

# Variáveis para a ROI
roi_x, roi_y, roi_w, roi_h = 300, 400, 500, 150
drawing = False
ix, iy = -1, -1

# Ajustar o limiar de confiança
confidence_threshold = 0.3

# Dicionário de nomes de classes
class_names = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Inicializar contadores de veículos
vehicle_count = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}

# Inicializar um dicionário para rastrear veículos contados
tracked_vehicles = {}
vehicle_id = 0

# Função para verificar se um veículo cruzou a linha de contagem
def check_line_crossing(center_y, line_y, vehicle_id):
    if center_y > line_y and vehicle_id not in tracked_vehicles:
        return True
    return False

# Função de callback do mouse para desenhar a ROI
def draw_roi(event, x, y, flags, param):
    global roi_x, roi_y, roi_w, roi_h, drawing, ix, iy, vehicle_count, tracked_vehicles

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
        vehicle_count = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
        tracked_vehicles = {}
        print(f"ROI definida: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")

cv2.namedWindow('Video')
cv2.setMouseCallback('Video', draw_roi)

# Função para enviar informações para o banco de dados SQLite
def enviar_informacoes_para_banco():
    global last_db_update_time
    data_atual = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    comando_sql = "INSERT INTO Contador_De_Veiculos (CAR, MOTORCYCLE, BUS, TRUCK, DATA_INICIO, DATA_FIM) VALUES (?, ?, ?, ?, ?, ?);"
    cur.execute(comando_sql, (vehicle_count["car"], vehicle_count["motorcycle"], vehicle_count["bus"], vehicle_count["truck"], last_db_update_time, data_atual))
    con.commit()
    last_db_update_time = data_atual
    print("Informações enviadas para o banco de dados SQLite.")

# Função agendada para enviar dados
def agendar_envio_dados():
    enviar_informacoes_para_banco()
    reset_vehicle_count()

# Função para resetar a contagem de veículos
def reset_vehicle_count():
    global vehicle_count
    vehicle_count = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}

# Agendar a função para ser executada a cada 5 minutos
schedule.every(0.5).minutes.do(agendar_envio_dados)

# Função para atualizar a lista de veículos rastreados
def update_tracked_vehicles():
    global tracked_vehicles
    for vehicle_id in list(tracked_vehicles.keys()):
        if tracked_vehicles[vehicle_id]["center_y"] > roi_h:
            del tracked_vehicles[vehicle_id]

# Função para calcular a distância entre dois pontos
def euclidean_distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

# Parâmetros de rastreamento
max_disappeared = 50
max_distance = 50

# Dicionário para armazenar os centroides dos veículos
vehicle_centroids = {}
disappeared = {}

# Função principal para rastreamento e contagem de veículos
def track_and_count_vehicles():
    global vehicle_centroids, disappeared, frame, roi, vehicle_id
    new_centroids = []

    # Detectar objetos na ROI
    results = model(roi)

    # Verificar se há resultados de detecção
    if results and results[0].boxes:
        for detection in results[0].boxes:
            bbox = detection.xyxy[0]
            confidence = detection.conf[0]
            class_id = int(detection.cls[0])

            if confidence >= confidence_threshold and class_id in class_names:
                bbox_x1 = int(bbox[0])
                bbox_y1 = int(bbox[1])
                bbox_x2 = int(bbox[2])
                bbox_y2 = int(bbox[3])

                bbox_center_x = (bbox_x1 + bbox_x2) // 2
                bbox_center_y = (bbox_y1 + bbox_y2) // 2

                new_centroids.append((bbox_center_x, bbox_center_y, class_id))

                cv2.circle(roi, (bbox_center_x, bbox_center_y), 3, (0, 0, 255), -1)
                cv2.rectangle(roi, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), (0, 255, 0), 2)

    if not vehicle_centroids:
        for i, centroid in enumerate(new_centroids):
            vehicle_centroids[vehicle_id] = centroid
            disappeared[vehicle_id] = 0
            vehicle_id += 1
    else:
        object_ids = list(vehicle_centroids.keys())
        object_centroids = list(vehicle_centroids.values())

        if object_centroids and new_centroids:
            D = dist.cdist(np.array(object_centroids)[:, :2], np.array(new_centroids)[:, :2])

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if D[row, col] > max_distance:
                    continue

                object_id = object_ids[row]
                vehicle_centroids[object_id] = new_centroids[col]
                disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            for row in unused_rows:
                object_id = object_ids[row]
                disappeared[object_id] += 1
                if disappeared[object_id] > max_disappeared:
                    del vehicle_centroids[object_id]
                    del disappeared[object_id]

            for col in unused_cols:
                vehicle_centroids[vehicle_id] = new_centroids[col]
                disappeared[vehicle_id] = 0
                vehicle_id += 1

    for object_id in list(vehicle_centroids.keys()):
        centroid = vehicle_centroids[object_id]
        class_id = centroid[2]
        class_name = class_names[class_id]
        if check_line_crossing(centroid[1], roi_h // 2, object_id):
            tracked_vehicles[object_id] = {"center_y": centroid[1], "class_id": class_id}
            vehicle_count[class_name] += 1
            print(f"Veículo contado: {class_name}, Total: {vehicle_count[class_name]}")

    update_tracked_vehicles()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    track_and_count_vehicles()

    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), 2)
    cv2.line(frame, (roi_x, roi_y + roi_h // 2), (roi_x + roi_w, roi_y + roi_h // 2), (255, 0, 0), 2)

    y_offset = 30
    for vehicle, count in vehicle_count.items():
        text = f"{vehicle}: {count}"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('Video', frame)

    schedule.run_pending()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

enviar_informacoes_para_banco()
cv2.destroyAllWindows()
cap.release()
cur.close()
con.close()

print("Contagem de veículos:")
print(json.dumps(vehicle_count, indent=4))
