import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

ALTURA_IMG, LARGURA_IMG = 128, 128
TAMANHO_LOTE = 32
EPOCAS = 10

dataset_dir = 'dataset/'

if not os.path.exists(dataset_dir):
    raise FileNotFoundError(f"Diretório '{dataset_dir}' não encontrado.")

classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
if not classes:
    raise FileNotFoundError("Nenhum diretório de classe encontrado em 'dataset/'.")
print(f"Classes encontradas: {classes}")

class_map = {cls: idx for idx, cls in enumerate(sorted(classes))}
print(f"Mapeamento das classes: {class_map}")

leitor_de_treinamento = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    validation_split=0.2
)

gerador_de_treinamento = leitor_de_treinamento.flow_from_directory(
    dataset_dir,
    target_size=(ALTURA_IMG, LARGURA_IMG),
    batch_size=TAMANHO_LOTE,
    class_mode='binary',
    subset='training',
    classes=[cls for cls in sorted(classes)]
)

gerador_validacao = leitor_de_treinamento.flow_from_directory(
    dataset_dir,
    target_size=(ALTURA_IMG, LARGURA_IMG),
    batch_size=TAMANHO_LOTE,
    class_mode='binary',
    subset='validation',
    classes=[cls for cls in sorted(classes)]
)

for imagens, rotulos in gerador_de_treinamento:
    plt.figure(figsize=(10, 10))
    for i in range(min(9, len(imagens))):
        plt.subplot(3, 3, i+1)
        plt.imshow(imagens[i])
        plt.title('Cachorro' if rotulos[i] == 0 else 'Sem cachorro')
        plt.axis('off')
    plt.show()
    break

modelo = Sequential([
    Input(shape=(ALTURA_IMG, LARGURA_IMG, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

modelo.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

historico = modelo.fit(
    gerador_de_treinamento,
    epochs=EPOCAS,
    validation_data=gerador_validacao
)

modelo.save('modelo_cachorro.h5')
