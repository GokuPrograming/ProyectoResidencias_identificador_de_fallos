import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
import pickle
import numpy as np
import cv2
import os

# Definir el tamaño de la imagen y las categorías
TamañoImagen = 256
Categoria = [
    "3_relevadores_normal_26_5", 
    "3_relevadores_trabajando_28_30", 
    "fuente_poder_trabajando_40-50grados", 
    "FuenteDePoder_35_grados_prom", 
    "paneles_laterales"
]
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Usar solo CPU

# Parámetros de configuración para la red neuronal
# neuronas = [32, 64, 128]
neuronas = [128]
densas = [0, 1, 2]
convpoo = [1, 2, 3]
drop = [0]

# Cargar datos serializados
imagenes = pickle.load(open("./src/data/imagenes.pickle", "rb"))
etiquetas = pickle.load(open("./src/data/etiquetas.pickle", "rb"))

# Normalizar las imágenes
imagenes = imagenes / 255.0
etiquetas = np.array(etiquetas)

# Convertir etiquetas a formato one-hot
etiquetas = to_categorical(etiquetas, num_classes=len(Categoria))

# Preparar la imagen con reducción de ruido y colormap térmico
def prepare(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (TamañoImagen, TamañoImagen))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    return img.reshape(-1, TamañoImagen, TamañoImagen, 1)

# Función para entrenar el modelo CNN
def entrenar():
    if not os.path.exists("models"):
        os.makedirs("models")
        
    for neurona in neuronas:
        for conv in convpoo:
            for densa in densas:
                for d in drop:
                    nombreModelo = f"RedConv-n{neurona}-cl{conv}-d{densa}-dropout{d}"
                    tensorboard = TensorBoard(log_dir=f'logs/{nombreModelo}')
                    
                    model = Sequential()
                    model.add(Conv2D(64, (3, 3), input_shape=(TamañoImagen, TamañoImagen, 1)))
                    model.add(Activation("relu"))
                    model.add(MaxPooling2D(pool_size=(2, 2)))
                    
                    if d == 1:
                        model.add(Dropout(0.2))
                    
                    for i in range(conv):
                        model.add(Conv2D(64, (3, 3)))
                        model.add(Activation("relu"))
                        model.add(MaxPooling2D(pool_size=(2, 2)))
                    
                    model.add(Flatten())
                    
                    for i in range(densa):
                        model.add(Dense(neurona))
                        model.add(Activation("relu"))
                    
                    model.add(Dense(len(Categoria)))  # Cambiado para múltiples clases
                    model.add(Activation('softmax'))   # Cambiado a softmax
                    
                    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])  # Cambiado a categorical_crossentropy
                    model.fit(imagenes, etiquetas, batch_size=30, epochs=13, validation_split=0.3, callbacks=[tensorboard])
                    model.save(f"models/{nombreModelo}.keras")

entrenar()
