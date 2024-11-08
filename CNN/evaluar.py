import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns

TamañoImagen = 256  # Cambiado a 256

# Lista de categorías
Categoria = [
    "3_relevadores_normal_26_5",
    "3_relevadores_trabajando_28_30",
    "fuente_poder_trabajando_40-50grados",
    "FuenteDePoder_35_grados_prom",
    "paneles_laterales",
]

def mostrar_imagen(img, title=""):
    """Función para mostrar una imagen con título."""
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def prepare(image_path):
    print(f"Preparando la imagen: {image_path}")
    
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("No se pudo cargar la imagen.")
        print("Imagen térmica cargada.")

        img_scaled = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        # print("Imagen escalada para normalización.")
        # mostrar_imagen(img_scaled, "Imagen escalada")

        img_denoised = cv2.GaussianBlur(img_scaled, (5, 5), 0)
        # print("Reducción de ruido aplicada (Filtro Gaussiano).")
        # mostrar_imagen(img_denoised, "Imagen sin ruido")

        img_resized = cv2.resize(img_denoised, (TamañoImagen, TamañoImagen))
        # print(f"Imagen redimensionada a {TamañoImagen}x{TamañoImagen} píxeles.")
        # mostrar_imagen(img_resized, "Imagen redimensionada")

        img_normalized = img_resized / 255.0
        # print("Imagen normalizada (valores entre 0 y 1).")
        # mostrar_imagen(img_normalized, "Imagen normalizada")

        img_resized_color = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
        heatmap = cv2.applyColorMap(img_resized, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_resized_color, 0.6, heatmap, 0.4, 0)
        # print("Generado el mapa de calor.")
        # mostrar_imagen(overlay, "Mapa de Calor")

        return img_normalized.reshape(-1, TamañoImagen, TamañoImagen, 1)

    except Exception as e:
        print(f"Error al preparar la imagen: {e}")

def predecir(model_path, image_paths):
    print("Cargando el modelo...")
    try:
        pred = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None, None

    predictions = []
    probabilities = []

    for image_path in image_paths:
        print(f"\nPrediciendo para la imagen: {image_path}")
        print("Preprocesando la imagen antes de la predicción...")
        processed_img = prepare(image_path)
        
        if processed_img is not None:
            prediction = pred.predict(processed_img)
            predicted_class = int(np.argmax(prediction, axis=1)[0])
            prob = prediction[0]  # Probabilidades para todas las clases
            predictions.append(Categoria[predicted_class])
            probabilities.append(prob)
            
            print(f"Predicción completada para {image_path}:")
            print(f"Clase predicha: {Categoria[predicted_class]} (Probabilidades: {prob})")
        else:
            predictions.append(None)
            probabilities.append(None)

    return predictions, np.array(probabilities)

def MatrizConfusion(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    print("\nMatriz de Confusión:")
    print(cm)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=Categoria, yticklabels=Categoria)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Matriz de Confusión')
    plt.show()

def ROC(true_labels, pred_probs):
    true_labels = np.array(true_labels)
    
    if len(set(true_labels)) != len(Categoria):
        raise ValueError("Las etiquetas verdaderas deben incluir todas las categorías disponibles.")

    true_labels_bin = np.zeros((len(true_labels), len(Categoria)))
    for i, label in enumerate(true_labels):
        true_labels_bin[i][label] = 1

    try:
        auc = roc_auc_score(true_labels_bin, pred_probs, multi_class='ovr')
        print(f"\nÁrea bajo la curva ROC (AUC): {auc}")
        fpr, tpr, thresholds = roc_curve(true_labels_bin.ravel(), pred_probs.ravel())
        plt.figure()
        plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title('Curva ROC')
        plt.legend(loc="lower right")
        plt.show()

    except ValueError as e:
        print(f"Error al calcular AUC: {e}")

if __name__ == "__main__":
    model_path = "./models/RedConv-n128-cl1-d2-dropout0.keras"
    image_paths = [
        "C:/Users/ronbo/Documents/universidad/residencias/colab_charge/cnn/test/3_relevadores_normal.jpg",
        "C:/Users/ronbo/Documents/universidad/residencias/colab_charge/cnn/test/fuente_poder_35_1.jpg",
        "C:/Users/ronbo/Documents/universidad/residencias/colab_charge/cnn/test/fuente_poder_35.jpg",
        "C:/Users/ronbo/Documents/universidad/residencias/colab_charge/cnn/test/fuente_poder_work_40-50.jpg",
        "C:/Users/ronbo/Documents/universidad/residencias/colab_charge/cnn/test/panel_lateral.jpg"
    ]

    # Etiquetas reales de las imágenes (usa índices correspondientes a las categorías)
    true_labels = [0, 1, 2, 3, 4]

    resultados, pred_probs = predecir(model_path, image_paths)
    
    if resultados is not None:
        pred_labels = [Categoria.index(resultado) for resultado in resultados if resultado is not None]

        accuracy = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=1)
        recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=1)
        f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=1)

        print("\nResultados de la predicción:")
        for i, (image_path, resultado) in enumerate(zip(image_paths, resultados)):
            print(f"La predicción para la imagen '{image_path}' es: {resultado}")

        print("\nMétricas de evaluación:")
        print(f"Precisión (Accuracy): {accuracy:.2f}")
        print(f"Precisión (Precision): {precision:.2f}")
        print(f"Recall (Sensibilidad): {recall:.2f}")
        print(f"F1-Score: {f1:.2f}")

        print(f'\nImprimiendo la matriz de confusión')
        MatrizConfusion(true_labels=true_labels, pred_labels=pred_labels)
        # Descomentar la siguiente línea si deseas también visualizar la curva ROC
        ROC(true_labels=true_labels, pred_probs=pred_probs)
