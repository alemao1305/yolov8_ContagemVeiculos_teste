import os

# Define o diretório onde os arquivos XML estão localizados
diretorio = 'recursos'

# Lista para armazenar os caminhos dos arquivos
caminhos_arquivos_db = []

# Percorre todos os arquivos no diretório
for arquivo in os.listdir(diretorio):
    # Verifica se o arquivo é um XML
    if arquivo.endswith('.db'):
        # Adiciona o caminho completo do arquivo à lista
        caminhos_arquivos_db.append(os.path.join(diretorio, arquivo))

# Escreve os caminhos em um arquivo de texto
with open('caminhos_arquivos_db.txt', 'w') as arquivo_saida:
    for caminho2 in caminhos_arquivos_db:
        arquivo_saida.write(caminho2 + '\n')

print('Caminhos dos arquivos db salvos com sucesso.', caminho2)