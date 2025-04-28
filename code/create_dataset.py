import os
import numpy as np
import string

# Caminho onde os dados serão salvos
DATA_PATH = os.path.join('MP_Data')

# Ações que você quer detectar
custom_actions = np.array(['Olá', 'Obrigado', 'Eu te amo', 'Espaço'])
# alphabet_actions = list(string.ascii_uppercase)
# actions = np.concatenate([custom_actions, alphabet_actions])  # Usando np.concatenate para combinar as listas
# 30 vídeos de cada ação
no_sequences = 30

# Cada vídeo terá 30 frames
sequence_length = 30

for action in custom_actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass
