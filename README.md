# yolov8_ContagemVeiculos_teste

# Detecção e Contagem de Veículos com YOLOv8

Este projeto implementa um sistema de detecção e contagem de veículos utilizando o modelo YOLOv8, OpenCV para processamento de vídeo, e SQLite para armazenamento de dados. A aplicação permite a captura de vídeo em tempo real a partir de uma URL de câmera, gera um código QR para fácil acesso à URL, e armazena informações de contagem de veículos em um banco de dados SQLite.

## Funcionalidades

- **Detecção de Veículos:** Detecta carros, motocicletas, ônibus, caminhões e bicicletas em vídeos.
- **Contagem de Veículos:** Conta os veículos que cruzam uma linha específica na região de interesse (ROI) do vídeo.
- **Interface Gráfica (GUI):** Uma interface gráfica simples com Tkinter para entrada da URL da câmera e do intervalo de tempo para contagem.
- **Geração de QR Code:** Gera um código QR a partir da URL fornecida.
- **Armazenamento em Banco de Dados:** Armazena as informações de contagem de veículos em um banco de dados SQLite.
- **Agendamento de Tarefas:** Usa a biblioteca `schedule` para agendar a contagem de veículos em intervalos definidos pelo usuário.

## Tecnologias Utilizadas

- **Linguagem de Programação:** Python
- **Modelo de Detecção:** [YOLOv8](https://github.com/ultralytics/ultralytics)
- **Processamento de Imagem:** OpenCV
- **Banco de Dados:** SQLite
- **Interface Gráfica:** Tkinter
- **Geração de QR Code:** qrcode

## Requisitos

Certifique-se de ter as seguintes bibliotecas instaladas antes de executar o programa:

- Python 3.6 ou superior
- OpenCV
- NumPy
- PyTorch
- Tkinter (geralmente incluído na instalação padrão do Python)
- Ultralytics (para YOLOv8)
- qrcode
- schedule
- scipy

Você pode instalar as dependências utilizando o arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Como Usar

  1- Configuração Inicial:
      * Certifique-se de ter um ambiente Python configurado.
      * Instale todas as dependências necessárias listadas acima.

  2- Execução do Programa:
      * Execute o arquivo Python principal para iniciar a interface gráfica:

```bash
python nome_do_arquivo.py
```

  3- Inserção da URL da Câmera:
        Insira a URL da câmera de onde o vídeo será capturado.
        Insira o intervalo de tempo (em minutos) para o envio de dados para o banco de dados. Caso não seja inserido, o intervalo padrão será de 5 minutos.

  4- Geração do QR Code:
        Após inserir a URL, clique no botão "IP da Câmera" para gerar e salvar um QR Code com a URL fornecida.

  5- Seleção de ROI:
        Ao iniciar a detecção, você poderá definir uma Região de Interesse (ROI) desenhando um retângulo na janela de vídeo.

  6- Contagem e Armazenamento:
        O sistema começará a detectar e contar os veículos que passam pela linha definida na ROI. As contagens serão armazenadas no banco de dados SQLite.

## Estrutura do Banco de Dados

O banco de dados contém uma tabela chamada Contador_De_Veiculos com os seguintes campos:

  * ID (INTEGER, PRIMARY KEY, AUTOINCREMENT): Identificador único de cada registro.
  * CAR (INTEGER): Número de carros detectados.
  * MOTORCYCLE (INTEGER): Número de motocicletas detectadas.
  * BUS (INTEGER): Número de ônibus detectados.
  * TRUCK (INTEGER): Número de caminhões detectados.
  * BICYCLE (INTEGER): Número de bicicletas detectadas.
  * DATA_INICIO (TEXT): Data e hora de início do intervalo de contagem.
  * DATA_FIM (TEXT): Data e hora de término do intervalo de contagem.

## Contribuições

Contribuições para a melhoria do projeto são bem-vindas. Sinta-se à vontade para abrir uma issue ou enviar um pull request no repositório.
## Licença

Este projeto é licenciado sob a Licença MIT - veja o arquivo LICENSE para mais detalhes.
## Contato

Para mais informações, entre em contato via olintoguarujas2@gmail.com

```

Este `README.md` fornece uma visão geral do projeto, suas funcionalidades, tecnologias utilizadas, instruções para configuração e execução, e detalhes sobre a estrutura do banco de dados. Certifique-se de ajustar as informações de contato, o nome do arquivo principal e outras configurações específicas do seu projeto conforme necessário.

```
