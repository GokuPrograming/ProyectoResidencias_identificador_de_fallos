import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import os
import cv2
import numpy as np

# Ruta a las carpetas de imágenes
data_dir = 'C:/Users/ronbo/Documents/Residencias/RESNET/database/Machine'

# Parámetros generales
img_size = (256, 256)  # Tamaño actualizado a 256x256 píxeles
input_shape = (256, 256, 3)  # Tamaño de imagen de 256x256 píxeles y 3 canales (RGB)
batch_size = 8  # Reducir tamaño del lote para menos consumo de memoria
num_classes = 5

# Crear directorios para logs y modelos
base_log_dir = "./resnet/logs"
base_model_dir = "./resnet/model"
os.makedirs(base_log_dir, exist_ok=True)
os.makedirs(base_model_dir, exist_ok=True)

# Función de preprocesamiento usando OpenCV
def custom_preprocessing(image):
    image = np.array(image, dtype=np.uint8)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    image = image.astype('float32') / 255.0
    return image

# Configurar el ImageDataGenerator
train_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocessing,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Bloque residual simplificado
def residual_block(x, filters, kernel_size, stride, activation='relu'):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation(activation)(x)
    
    return x

# Construir el modelo ResNet optimizado
def build_resnet(input_shape, num_classes, filters, kernel_size, stride, num_residual_blocks=2, dense_units=64, dropout_rate=0.3):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
    
    # Añadir bloques residuales con reducción
    for i in range(num_residual_blocks):
        filters_block = filters * (2 ** i)
        stride_block = 1 if i == 0 else 2
        x = residual_block(x, filters=filters_block, kernel_size=kernel_size, stride=stride_block)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Hiperparámetros fijos para reducir combinaciones
filters = 64
kernel_size = 3
stride = 1

# Crear el modelo ResNet
model = build_resnet(input_shape=input_shape, num_classes=num_classes, filters=filters, kernel_size=kernel_size, stride=stride)

# Crear directorios específicos para logs y modelos
log_dir = f"{base_log_dir}/optimized/"
model_dir = f"{base_model_dir}/optimized/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Configurar TensorBoard y Early Stopping
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Entrenar el modelo
model.fit(
    train_generator,
    epochs=20,  # Reducido el número de épocas
    validation_data=validation_generator,
    callbacks=[tensorboard_callback, early_stopping_callback]
)

# Guardar el modelo
model.save(f'{model_dir}/optimized_model.keras')

# Instrucciones para visualizar TensorBoard
print(f"Ejecuta el siguiente comando en tu terminal para visualizar TensorBoard:")
print(f"tensorboard --logdir={log_dir}")
