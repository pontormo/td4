import cv2
import numpy as np
import random
import os

def etiquetage_composantes_connexes(im):
    hauteur = len(im)
    largeur = len(im[0])
    
    # image colorée pour afficher les composantes
    image_color = np.zeros((hauteur, largeur, 3), dtype=np.uint8)  # Image en couleur (noir)
    
    # pixels visités
    visite = [[False for _ in range(largeur)] for _ in range(hauteur)]
    
    # Directions des voisins (4-adjacence)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # étiquette actuelle
    current_label = 1
    
    # couleur aléatoire
    def generer_couleur_aleatoire():
        return [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
    
    # parcours pour étiqueter les pixels d'une composante
    def parcoursCC(p, couleur):
        s = [p]
        visite[p[0]][p[1]] = True
        image_color[p[0], p[1]] = couleur
        
        while s:
            r = s.pop()
            
            for direction in directions:
                v = (r[0] + direction[0], r[1] + direction[1])
                
                if 0 <= v[0] < hauteur and 0 <= v[1] < largeur and not visite[v[0]][v[1]] and im[v[0]][v[1]] == 1:
                    visite[v[0]][v[1]] = True
                    image_color[v[0], v[1]] = couleur  # Appliquer la couleur à l'image
                    s.append(v)
    
    # parcourt de l'image et étiquetage des composantes connexes avec des couleurs aléatoires
    for i in range(hauteur):
        for j in range(largeur):
            if im[i][j] == 1 and not visite[i][j]:
                couleur_aleatoire = generer_couleur_aleatoire()
                parcoursCC((i, j), couleur_aleatoire)
                current_label += 1
    
    return image_color


image_path = "binary.png"
if os.path.exists(image_path):
    # Charger l'image en niveaux de gris
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Erreur lors du chargement de l'image")
    else:
        # conversion en binaire
        _, image_bin = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)

        # Appliquer l'étiquetage des composantes connexes avec des couleurs aléatoires
        image_coloree = etiquetage_composantes_connexes(image_bin)

        # Afficher l'image colorée
        cv2.imshow("Image etiquetage", image_coloree)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
else:
    print(f"Le fichier {image_path} n'existe pas.")
