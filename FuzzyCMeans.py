import numpy as np
import cv2
import matplotlib.pyplot as plt


#  initialisation de la matrice d'appartenance de depart
def matrice_appartenance(n_samples, n_clusters):
    U = np.random.rand(n_samples, n_clusters)
    U /= np.sum(U, axis=1, keepdims=True) # on normalise our que la somme des appartenances = 1
    return U


# mise à jour des centres
def maj_centres(X, U, m):
    um = U ** m # on met chaque élément de la matrice d'appartenance à la puissance m
    numerateur = um.T.dot(X) #somme pondérée des pixels
    denominateur = np.sum(um, axis=0)[:, None] #somme des poids par cluster
    centres = numerateur / denominateur  #on obtient les centre on faisant la moyenne
    return centres


# mise à jour des degres d'appartance
def maj_appartenance(X, centers, m):
    #on calcule la distance entre chaque pixel et chaque centre
    dist = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2) + 1e-8
    puissance = 2 / (m - 1)
    # pour chaque i,j on calcule sum_k ( (d(i,j) / d(i,k))^2/(m-1) )
    denom = (dist[:, :, None] / dist[:, None, :]) ** puissance
    denom = np.sum(denom, axis=2) #somme sur k -> on obtient une matrice (N x C)
    U_new = 1.0 / denom          #on inverse pour obtenir la nouvelle matrice d'appartenance comme dans la formule
    return U_new



def fuzzy_c_means(X, n_clusters=2, m=2, epsilon=0.01, max_iter=100):
    n_samples = X.shape[0]
    U = matrice_appartenance(n_samples, n_clusters) #initialisation matrice

    for iteration in range(max_iter):
        centres = maj_centres(X, U, m)
        U_new = maj_appartenance(X, centres, m)

        if np.linalg.norm(U_new - U) < epsilon: #critere d'arret sinon on retourne à la step 2 de l'algo
            print(f" Convergence à l’itération {iteration + 1}")
            break
        U = U_new

    return centres, U


# on charge des images
image_path = 'milky-way-nvg.jpg'
image = cv2.imread(image_path)

if image is None:
    raise ValueError("Erreur chemin.")

if len(image.shape) == 3:
    rows, cols, ch = image.shape
    X = image.reshape(-1, ch).astype(np.float32)



print(f"dim de l'image : {rows}x{cols} ")

# parametre du fuzzy c
n_clusters = 5
m = 3
epsilon = 0.01

centers, U = fuzzy_c_means(X, n_clusters=n_clusters, m=m, epsilon=epsilon) #on execute l'algo


cluster_labels = np.argmax(U, axis=1) #Pour chaque pixel on chosit le cluster de plus forte appartenance
segmented_image_heatmap = cluster_labels.reshape(rows, cols) #carte labels

# chaque pixel prend la couleur de son centre
segmented = centers[np.argmax(U, axis=1)].reshape(rows, cols, ch).astype(np.uint8)

plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
plt.title('Image originale')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Fuzzy C ')
plt.imshow(segmented_image_heatmap, cmap='hot')
plt.axis('off')


plt.subplot(1, 3, 3)
plt.title('Fuzzy C')
plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()
