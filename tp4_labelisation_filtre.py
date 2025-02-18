import cv2
import numpy as np
import random
import os

def etiquetage_composantes_connexes(im):
    hauteur = len(im)
    largeur = len(im[0])
    
    # tableau pixels visités
    visite = [[False for _ in range(largeur)] for _ in range(hauteur)]
    
    # Directions des voisins (4-adjacence)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    composantes = []
    composantes_labels = [[0 for _ in range(largeur)] for _ in range(hauteur)]  # Matrice des labels des composantes
    
    # on collecte les pixels d'une composante
    def parcoursCC(p, label):
        s = [p]
        composante = [p]
        visite[p[0]][p[1]] = True
        composantes_labels[p[0]][p[1]] = label
        
        while s:
            r = s.pop()
            
            for direction in directions:
                v = (r[0] + direction[0], r[1] + direction[1])
                
                if 0 <= v[0] < hauteur and 0 <= v[1] < largeur and not visite[v[0]][v[1]] and im[v[0]][v[1]] == 1:
                    visite[v[0]][v[1]] = True
                    composante.append(v)
                    composantes_labels[v[0]][v[1]] = label
                    s.append(v)
        
        composantes.append(composante)  # Ajouter la composante à la liste
    
    # Parcourir l'image pour étiqueter les composantes connexes
    current_label = 1
    for i in range(hauteur):
        for j in range(largeur):
            if im[i][j] == 1 and not visite[i][j]:
                parcoursCC((i, j), current_label)
                current_label += 1
    
    return composantes, composantes_labels

def filtre_aire(im, seuil):
    composantes, composantes_labels = etiquetage_composantes_connexes(im)
    
    # Créer une image filtrée avec 3 canaux (couleur)
    image_filtrée = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.uint8)
    
    # Fonction pour générer une couleur aléatoire
    def couleur_aleatoire():
        return [random.randint(0, 255) for _ in range(3)]  # Retourne une couleur aléatoire (R, G, B)
    
    # Garder uniquement les composantes dont la taille est supérieure ou égale au seuil
    for composante in composantes:
        if len(composante) >= seuil:
            # couleur aléatoire
            couleur = couleur_aleatoire()
            for pixel in composante:
                image_filtrée[pixel[0], pixel[1]] = couleur  # Remplir les pixels de la composante avec la couleur aléatoire
    
    return image_filtrée


image_path = "binary.png"
if os.path.exists(image_path):
    # image en niveaux de gris
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Erreur lors du chargement de l'image")
    else:
        # conversion en binaire
        _, image_bin = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)

        # application du filtre a l'image
        seuil = 500
        image_filtrée = filtre_aire(image_bin, seuil)

        # Affichage de l'image filtrée
        cv2.imshow("Image filtrage", image_filtrée)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
else:
    print(f"Le fichier {image_path} n'existe pas.")

