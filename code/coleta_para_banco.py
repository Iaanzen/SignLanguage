import os
import sqlite3
import cv2

# Caminho para as imagens existentes
DATA_DIR = './data'

# Conecta (ou cria) o banco de dados
conn = sqlite3.connect('gestos.db')
cursor = conn.cursor()

# Cria a tabela se ainda não existir
cursor.execute('''
    CREATE TABLE IF NOT EXISTS imagens (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        classe INTEGER,
        nome_arquivo TEXT,
        imagem BLOB
    )
''')
conn.commit()

# Percorre todas as pastas de classes e imagens
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)

    if not os.path.isdir(dir_path):
        continue

    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)

        # Lê a imagem
        img = cv2.imread(file_path)
        if img is None:
            print(f'[!] Imagem inválida: {file_path}')
            continue

        # Codifica em bytes
        _, buffer = cv2.imencode('.jpg', img)
        image_bytes = buffer.tobytes()

        # Insere no banco
        cursor.execute('''
            INSERT INTO imagens (classe, nome_arquivo, imagem)
            VALUES (?, ?, ?)
        ''', (int(dir_), file_name, image_bytes))

        print(f'[✓] Inserido: Classe {dir_} - {file_name}')

conn.commit()
conn.close()

print('\n✅ Importação concluída!')
