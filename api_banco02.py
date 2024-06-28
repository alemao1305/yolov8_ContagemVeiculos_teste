import psycopg2


# Defina as informações de conexão
host = "10.3.0.151"
port = "5433"
database = "TesteOlinto"
user = "Olinto"
password = "olinto1"

# Defina o script SQL para criar a tabela
sql_create_table = """
CREATE TABLE IF NOT EXISTS Contador_OCR (
  CAR          INT,
  MOTORCYCLE   INT,
  BUS          INT,
  TRUCK        INT, 
  DATA_INICIO  TIMESTAMP,
  DATA_FIM     TIMESTAMP
);
"""


# Defina o script SQL para inserir dados na tabela
sql_insert_data = """
INSERT INTO Contador_OCR ("CAR", "MOTORCYCLE", "BUS", "TRUCK", "DATA_INICIO", "DATA_FIM")
  VALUES (1, 1, 1, 1, "2024-04-27 00:00:00", "2024-04-27 00:00:00");
"""

# Tente estabelecer uma conexão
try:
    # Conecte ao banco de dados PostgreSQL
    con = psycopg2.connect(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password
    )

    # Crie um cursor para executar consultas
    cursor = con.cursor()

    # Execute o script SQL para criar a tabela
    cursor.execute(sql_create_table)
    print("Tabela criada com sucesso!")

    # Execute o script SQL para inserir os dados na tabela
    cursor.execute(sql_insert_data)
    print("Dados inseridos com sucesso!")

    # Não se esqueça de fazer commit para salvar as alterações
    con.commit()

    # Feche o cursor e a conexão
    cursor.close()
    con.close()

except psycopg2.Error as e:
    print("Erro ao conectar ao PostgreSQL:", e)
