import numpy as np
import cv2
import matplotlib.pyplot as plt

def otsu_thresholding(image):

    hist = np.zeros(256, dtype=int)
    
    # histogramme
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            intensity = image[i, j]
            hist[intensity] += 1
    
    plt.plot(hist)
    plt.title("Histogramme de l'image")
    plt.xlabel("Niveau de gris")
    plt.ylabel("Fréquence")
    plt.show()
    
    # Nombre total de pixels
    total_pixels = image.size
    
    # Initialisation
    sum_total = np.sum(np.arange(256) * hist)  # Somme totale des intensités des pixels
    sum_background = 0
    weight_background = 0
    weight_foreground = 0
    max_variance = 0
    optimal_threshold = 0

    # Itérer sur tous les seuils possibles pour calculer la variance inter-classes
    for threshold in range(1, 256):
        # Calcul de la probabilité et des sommes des classes
        weight_background += hist[threshold - 1]  # Ajouter le poids de la classe d'arrière-plan
        weight_foreground = total_pixels - weight_background  # Poids de la classe avant-plan
        
        if weight_background == 0 or weight_foreground == 0:
            continue
        
        sum_background += (threshold - 1) * hist[threshold - 1]  # Mise à jour de la somme des intensités de l'arrière-plan
        sum_foreground = sum_total - sum_background  # Somme des intensités du premier plan
        
        # Calcul des moyennes des classes
        mean_background = sum_background / weight_background
        mean_foreground = sum_foreground / weight_foreground
        
        # Calcul de la variance inter-classes
        # Utilisation de float64 pour éviter les débordements
        between_class_variance = np.float64(weight_background) * np.float64(weight_foreground) * (np.float64(mean_background) - np.float64(mean_foreground)) ** 2
        
        # maximum de la variance inter-classes
        if between_class_variance > max_variance:
            max_variance = between_class_variance
            optimal_threshold = threshold
    
    return optimal_threshold

# image en niveaux de gris
image_path = "camera.png"  # Remplacer par votre image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Erreur lors du chargement de l'image.")
else:
    # seuillage
    optimal_threshold = otsu_thresholding(image)
    print(f"Le seuil optimal d'Otsu est : {optimal_threshold}")
    
    otsu_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Si l'intensité du pixel est supérieure ou égale au seuil, on le met à 255 (blanc)
            # Sinon, on le met à 0 (noir)
            otsu_image[i, j] = 255 if image[i, j] >= optimal_threshold else 0
    
    cv2.imshow("Seuillage d'Otsu", otsu_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
