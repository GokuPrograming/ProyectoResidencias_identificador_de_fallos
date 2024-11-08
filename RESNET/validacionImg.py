import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns

# Parámetros
imagenes_rutas = [
    "C:/Users/ronbo/Documents/universidad/residencias/colab_charge/cnn/test/3_relevadores_normal.jpg",
    "C:/Users/ronbo/Documents/universidad/residencias/colab_charge/cnn/test/fuente_poder_35_1.jpg",
    "C:/Users/ronbo/Documents/universidad/residencias/colab_charge/cnn/test/fuente_poder_35.jpg",
    "C:/Users/ronbo/Documents/universidad/residencias/colab_charge/cnn/test/fuente_poder_work_40-50.jpg",
    "C:/Users/ronbo/Documents/universidad/residencias/colab_charge/cnn/test/panel_lateral.jpg"
]  # Lista de imágenes a analizar

ruta_modelo = 'C:/Users/ronbo/Documents/Residencias/RESNET/model/optimized/optimized_model.keras'
img_size = (256, 256)  # Tamaño de imagen que debe coincidir con el tamaño usado en el entrenamiento

# Clases
classes = [
    "3_relevadores_normal_26_5",
    "3_relevadores_trabajando_28_30",
    "fuente_poder_trabajando_40-50grados",
    "FuenteDePoder_35_grados_prom",
    "paneles_laterales",
]

# Etiquetas reales para cada imagen
labels_true = [1, 4, 4, 3, 5]  # Ejemplo de las etiquetas reales (índices de las clases)

# Definir la función de preprocesamiento
def custom_preprocessing(image):
    # Convertir la imagen de Tensor a NumPy
    image = np.array(image, dtype=np.uint8)
    
    # Mostrar la imagen original
    # plt.figure(figsize=(5, 5))
    # plt.imshow(image)
    # plt.title("Imagen Original")
    # plt.axis('off')
    # plt.show()
    
    # Aplicar desenfoque Gaussiano
    image_blur = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Mostrar la imagen después del desenfoque
    # plt.figure(figsize=(5, 5))
    # plt.imshow(image_blur)
    # plt.title("Imagen después del Desenfoque Gaussiano")
    # plt.axis('off')
    # plt.show()
    
    # Aplicar el mapa de color JET
    image_color_map = cv2.applyColorMap(image_blur, cv2.COLORMAP_JET)
    
    # Mostrar la imagen con el mapa de color JET
    # plt.figure(figsize=(5, 5))
    # plt.imshow(image_color_map)
    # plt.title("Imagen con Mapa de Color JET")
    # plt.axis('off')
    # plt.show()
    
    # Convertir de nuevo a float32 para Keras
    image = image_color_map.astype('float32') / 255.0
    
    return image

# Función para predecir varias imágenes y calcular la matriz de confusión, métricas y la curva ROC
def predecir_imagenes_con_roc(ruta_modelo, imagenes_rutas, labels_true):
    # Cargar el modelo
    model = tf.keras.models.load_model(ruta_modelo)
    
    y_true = []
    y_pred = []
    y_score = []  # Para almacenar las probabilidades predichas

    # Analizar cada imagen
    for i, ruta_imagen in enumerate(imagenes_rutas):
        print(f'Analizando imagen: {ruta_imagen}')
        
        # Cargar la imagen
        img = image.load_img(ruta_imagen, target_size=img_size)
        img_array = image.img_to_array(img)  # Convertir la imagen a un array
        
        # Preprocesar la imagen
        img_array = custom_preprocessing(img_array)
        img_array = np.expand_dims(img_array, axis=0)  # Añadir la dimensión del batch
        
        # Hacer la predicción
        prediction = model.predict(img_array)
        pred_class = np.argmax(prediction)
        
        # Almacenar las etiquetas verdaderas y las predicciones
        y_true.append(labels_true[i])
        y_pred.append(pred_class)
        y_score.append(prediction[0])  # Guardar las probabilidades predichas
        
        print(f'La predicción para la imagen {ruta_imagen} es: {classes[pred_class]}\n')

    # Calcular la matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    
    # Mostrar la matriz de confusión
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.show()

    # Calcular las métricas
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Mostrar las métricas
    print(f"Precisión (Accuracy): {accuracy:.2f}")
    print(f"Precisión (Precision): {precision:.2f}")
    print(f"Recall (Sensibilidad): {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    
    # Convertir las etiquetas verdaderas a formato binario (one-hot encoding)
    y_true_bin = label_binarize(y_true, classes=np.arange(len(classes)))
    
    # Calcular la curva ROC y el área bajo la curva (AUC) para cada clase
    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), np.array(y_score).ravel())
    roc_auc = auc(fpr, tpr)

    # Mostrar la curva ROC
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', label=f'Curva ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title('Curva ROC')
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.legend(loc='lower right')
    plt.show()

# Llamar a la función para predecir las imágenes, calcular la matriz de confusión, métricas y la curva ROC
predecir_imagenes_con_roc(ruta_modelo, imagenes_rutas, labels_true)
