import os
import cv2
import numpy as np


def generate_binary_mask(mask):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    # Crear una imagen binaria basada en la mascara
    mask_alpha = mask_image[:, :, 3]  # Suponiendo un canal alfa en la mascara
    binary_mask = np.where(mask_alpha > 0, 255, 0).astype('uint8')
    return binary_mask


def delete_irrelevant_detected_pixels(binary_mask, min_area=1000):
    # Aplicar etiquetado de componentes conectados
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8)

    # Crear una nueva mascara para almacenar los objetos grandes
    filtered_mask = np.zeros_like(binary_mask)

    for i in range(1, num_labels):  # Ignorar el fondo (label=0)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            # Mantener solo los objetos grandes
            filtered_mask[labels == i] = 255

    return filtered_mask


def fill_little_spaces(binary_mask, kernel_size=15):
    # Suavizar la mascara con MORPH_CLOSE para eliminar pequenos agujeros
    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed_mask = cv2.morphologyEx(
        binary_mask, cv2.MORPH_CLOSE, closing_kernel)
    return closed_mask


def soften_contours(binary_mask, kernel_size=100):
    # Definir el kernel de dilatacion
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Aplicar dilatacion
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    return dilated_mask


def blur_mask(binary_mask):
    # (51,51) es el tamano del kernel, 20 es la desviacion estandar
    blurred_mask = cv2.GaussianBlur(binary_mask, (51, 51), 20)
    return blurred_mask


def delete_files(file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):  # Verifica si el archivo existe
            os.remove(file_path)
            print(f"Archivo '{file_path}' eliminado con exito.")