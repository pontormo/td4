import cv2
import numpy as np
import random
import os

def etiquetage_composantes_connexes_2passes(im):
    hauteur, largeur = im.shape
    labels = np.zeros_like(im, dtype=int)  # Matrice pour stocker les labels
    next_label = 1  # Premier label à attribuer
    
    # Première passe
    equivalences = {}  # Dictionnaire pour stocker les équivalences de labels
    
    for i in range(hauteur):
        for j in range(largeur):
            if im[i, j] == 1:
                # Chercher les voisins
                voisins = []
                if i > 0 and im[i-1, j] == 1:  # Voisin du dessus
                    voisins.append(labels[i-1, j])
                if j > 0 and im[i, j-1] == 1:  # Voisin de gauche
                    voisins.append(labels[i, j-1])

                if not voisins:
                    # Si aucun voisin n'est déjà labellisé, on attribue un nouveau label
                    labels[i, j] = next_label
                    next_label += 1
                else:
                    # label du voisin existant
                    labels[i, j] = min(voisins)

                    for voisin in voisins:
                        if voisin != labels[i, j]:
                            if voisin not in equivalences:
                                equivalences[voisin] = []
                            equivalences[voisin].append(labels[i, j])
    
    # Deuxième passe
    for i in range(hauteur):
        for j in range(largeur):
            if im[i, j] == 1:
                # Pour chaque pixel, suivre les équivalences et assigner le plus petit label
                current_label = labels[i, j]
                while current_label in equivalences:
                    current_label = min(equivalences[current_label])
                labels[i, j] = current_label

    return labels

def afficher_image_avec_labels(image, labels):

    color_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    
    # couleur aléatoire pour chaque label
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]  # Ignorer le label 0 (arrière-plan)
    
    label_colors = {}
    for label in unique_labels:
        label_colors[label] = [random.randint(0, 255) for _ in range(3)]  # Générer une couleur aléatoire
    
    # on colorie l'image en fonction des labels
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            label = labels[i, j]
            if label > 0:
                color_image[i, j] = label_colors[label]

    return color_image


image_path = "binary.png"
if os.path.exists(image_path):
    # niveau de gris
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Erreur lors du chargement de l'image")
    else:
        # Convertion en binaire
        _, image_bin = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)

        # Appliquer l'étiquetage des composantes connexes en 2 passes
        labels = etiquetage_composantes_connexes_2passes(image_bin)

        # Afficher l'image avec les labels colorés
        image_avec_labels = afficher_image_avec_labels(image_bin, labels)

        cv2.imshow("Image labélisation", image_avec_labels)  # Afficher l'image avec étiquettes colorées
        cv2.waitKey(0)
        cv2.destroyAllWindows()
else:
    print(f"Le fichier {image_path} n'existe pas.")
