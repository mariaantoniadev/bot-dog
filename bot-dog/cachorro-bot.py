import cv2
import numpy as np
from tensorflow.keras.models import load_model

ALTURA_IMG, LARGURA_IMG = 128, 128

modelo = load_model('modelo_cachorro.h5')

def preprocessar_imagem(imagem):
    imagem = cv2.resize(imagem, (LARGURA_IMG, ALTURA_IMG))
    imagem = imagem / 255.0
    imagem = np.expand_dims(imagem, axis=0)
    return imagem

captura = cv2.VideoCapture(0)

while True:
    ret, quadro = captura.read()
    if not ret:
        break
    
    cv2.imshow('Frame Capturado', quadro)
    quadro_processado = preprocessar_imagem(quadro)
    previsao = modelo.predict(quadro_processado)
    limiar = 0.5
    rotulo = 'Cachorro na camera' if previsao[0] > limiar else 'Sem cachorro na camera'
    print(f'Previsão: {previsao[0]}') 
    cv2.putText(quadro, rotulo, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Detecção de Cachorro', quadro)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

captura.release()
cv2.destroyAllWindows()
