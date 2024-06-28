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
import time
from diretorio_db import caminho2

# Variável global para armazenar a URL
url1 = ""

def gera_qr_code():
    global url1

    url = website_entry.get()

    if len(url) == 0:
        messagebox.showinfo(
            title="Erro!",
            message="Favor insira uma URL válida")
    else:
        opcao_escolhida = messagebox.askokcancel(
            title=url,
            message=f"O endereço URL é: \n "
                    f"Endereço: {url} \n "
                    f"Pronto para salvar?")

        if opcao_escolhida:
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(url)
            qr.make(fit=True)
            img = qr.make_image(fill_color='black', back_color='white')
            img.save('qrExport.png')
            url1 = url
            window.destroy()  # Fechar a janela após salvar o URL

# Interface Gráfica Tkinter para entrada da URL
if __name__ == '__main__':
    window = Tk()
    window.title("Gerador de Código QR")
    window.config(padx=10, pady=100)

    # Labels
    website_label = Label(text="URL:")
    website_label.grid(row=2, column=0)

    # Entries
    website_entry = Entry(width=35)
    website_entry.grid(row=2, column=1, columnspan=2)
    website_entry.focus()

    # Botão
    add_button = Button(text="IP da Câmera", width=36, command=gera_qr_code)
    add_button.grid(row=4, column=1, columnspan=2)

    window.mainloop()

# -----------------------------------------------------------------------------------------------------------

# Carregar modelo YOLOv8 treinado para detectar carro, moto, ônibus e caminhão
model_path = 'yolov8n.pt'
model = YOLO(model_path)
model.to('cuda' if torch.cuda.is_available() else 'cpu')  # Usar GPU se disponível

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
class_names = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck"}

# Inicializar contadores de veículos
vehicle_count = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}

# Inicializar um dicionário para rastrear veículos contados
counted_vehicles = set()

# Função para verificar se um veículo cruzou a linha de contagem
def check_line_crossing(center_y, line_y, direction):
    if direction == 'up' and center_y < line_y:
        return True
    elif direction == 'down' and center_y > line_y:
        return True
    return False

# Função de callback do mouse para desenhar a ROI
def draw_roi(event, x, y, flags, param):
    global roi_x, roi_y, roi_w, roi_h, drawing, ix, iy, vehicle_count, counted_vehicles

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
        counted_vehicles = set()

cv2.namedWindow('Video')
cv2.setMouseCallback('Video', draw_roi)

# Função para enviar informações para o banco de dados SQLite
def enviar_informacoes_para_banco():
    global last_db_update_time
    global vehicle_count
    data_atual = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    comando_sql = "INSERT INTO Contador_De_Veiculos (CAR, MOTORCYCLE, BUS, TRUCK, DATA_INICIO, DATA_FIM) VALUES (?, ?, ?, ?, ?, ?);"
    cur.execute(comando_sql, (vehicle_count["car"], vehicle_count["motorcycle"], vehicle_count["bus"], vehicle_count["truck"], last_db_update_time, data_atual))
    con.commit()
    last_db_update_time = data_atual
    print("Informações enviadas para o banco de dados SQLite.")

# Função agendada para enviar dados
def agendar_envio_dados():
    enviar_informacoes_para_banco()
    reset_vehicle_count()  # Resetar o contador de veículos

# Função para resetar a contagem de veículos
def reset_vehicle_count():
    global vehicle_count
    vehicle_count = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}

# Agendar a função para ser executada a cada 5 minutos
schedule.every(0.20).minutes.do(agendar_envio_dados)

# Loop principal
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

                # Calcular o centro do bounding box
                bbox_center_x = (bbox_x1 + bbox_x2) // 2
                bbox_center_y = (bbox_y1 + bbox_y2) // 2

                # Desenhar ponto vermelho no centro do bounding box
                cv2.circle(roi, (bbox_center_x, bbox_center_y), 3, (0, 0, 255), -1)

                vehicle_id = f"{bbox_x1}-{bbox_y1}-{bbox_x2}-{bbox_y2}"
                if vehicle_id not in counted_vehicles and check_line_crossing(bbox_center_y, roi_y + roi_h // 2, 'down'):
                    counted_vehicles.add(vehicle_id)
                    class_name = class_names[class_id]
                    vehicle_count[class_name] += 1
                    print(f"Veículo contado: {class_name}, Total: {vehicle_count[class_name]}")  # Debug

                # Desenhar boxes de detecção
                cv2.rectangle(roi, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), (0, 255, 0), 2)

    # Desenhar retângulo da ROI no frame original
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), 2)

    # Desenhar linha de contagem
    cv2.line(frame, (roi_x, roi_y + roi_h // 2), (roi_x + roi_w, roi_y + roi_h // 2), (255, 0, 0), 2)

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

    # Executar tarefas agendadas
    schedule.run_pending()

    # Ajustar tempo de espera entre frames para acelerar o vídeo
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Fechar janela e liberar recursos
enviar_informacoes_para_banco()
cv2.destroyAllWindows()
cap.release()
cur.close()
con.close()

# Exibir contagem de veículos
print("Contagem de veículos:")
print(json.dumps(vehicle_count, indent=4))
