import os
import shutil
import re

# Caminho da pasta com as subpastas de sinalizadores
ROOT_DIR = 'DATA'
DEST_DIR = 'VIDEOS_POR_PALAVRA'

# Cria a pasta destino se não existir
os.makedirs(DEST_DIR, exist_ok=True)

# Regex para extrair a palavra (entre números e "Sinalizador")
def extrair_palavra(nome):
    match = re.search(r'\d+([A-Za-zÇçÀ-ú]+)Sinalizador', nome)
    return match.group(1) if match else None

# Percorre todas as subpastas de Sinalizador01, 02...
for subfolder in os.listdir(ROOT_DIR):
    caminho_subpasta = os.path.join(ROOT_DIR, subfolder)

    if os.path.isdir(caminho_subpasta):
        for filename in os.listdir(caminho_subpasta):
            if filename.endswith(('.mp4', '.mov', '.avi')):
                palavra = extrair_palavra(filename)

                if palavra:
                    destino_palavra = os.path.join(DEST_DIR, palavra)
                    os.makedirs(destino_palavra, exist_ok=True)

                    origem = os.path.join(caminho_subpasta, filename)
                    destino = os.path.join(destino_palavra, filename)

                    shutil.copy2(origem, destino)
                    print(f"Copiado: {filename} → {palavra}/")
