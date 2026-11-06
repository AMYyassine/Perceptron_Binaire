import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data[:, 2:4]  # Longueur et largeur des pétales
y = iris.target

'''
on d'abords essayer sans normaliser ou standardiser X
puis on va essayer d'encoder y pour qu'il passe de {0,1,2} à un vecteur onehot [1,0,0]
'''
Y=list()
for i in range(len(y)):
    if y[i]==0:
        Y.append([1,0,0])
    if y[i]==1:
        Y.append([0,1,0])
    if y[i]==2:
        Y.append([0,0,1])

#on va prendre uniquement par exemple 120 exemples pour entrainement et 30 pour le test
from sklearn.model_selection import train_test_split
Y = np.array(Y)
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    train_size=120,
    test_size=30,
    random_state=42,  # Reproduire la même séparation
    stratify=y,      # Garder les proportions des classes
)

print("X_train :", X_train.shape)  # (120, 2)
print("Y_train :", Y_train.shape)  # (120, 3)
print("X_test  :", X_test.shape)   # (30, 2)
print("Y_test  :", Y_test.shape)   # (30, 3)




# X_train : (120, 2)
# Y_train : (120, 3), avec des cibles one-hot

n, D = X_train.shape
K = Y_train.shape[1]

# Ajouter une colonne de 1 pour intégrer les biais
X_train_biais = np.column_stack((X_train, np.ones(n)))

# Une ligne par classe : D poids + 1 biais
W = np.zeros((K, D + 1))

eta = 1.0
max_epoques = 55
epoque = 0
modification = True

while modification and epoque < max_epoques:
    modification = False
    epoque += 1
    nombre_maj = 0

    for i in range(n):
        xi = X_train_biais[i]

        # Calculer les K scores avec les poids actuels
        scores = W @ xi

        classe_predite = np.argmax(scores)
        classe_vraie = np.argmax(Y_train[i])

        if classe_predite != classe_vraie:
            W[classe_vraie] += eta * xi
            W[classe_predite] -= eta * xi

            modification = True
            nombre_maj += 1

    print(f"Époque {epoque} : {nombre_maj} mises à jour")

if not modification:
    print("Arrêt : une époque complète sans mise à jour.")
else:
    print("Arrêt : nombre maximal d'époques atteint.")

print("\nPoids par classe :")
print(W[:, :-1])

print("\nBiais par classe :")
print(W[:, -1])


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 1. Prédictions sur les 30 exemples de test
X_test_biais = np.column_stack((X_test, np.ones(len(X_test))))

scores_test = X_test_biais @ W.T
y_pred = np.argmax(scores_test, axis=1)
y_vrai = np.argmax(Y_test, axis=1)

# Évaluation : les poids W restent fixes
erreurs = y_pred != y_vrai
accuracy = np.mean(y_pred == y_vrai)

print(f"Exactitude sur le test : {accuracy:.2%}")
print(f"Prédictions correctes : {np.sum(~erreurs)}/{len(y_vrai)}")
print(f"Nombre d'erreurs : {np.sum(erreurs)}")

# 2. Grille de points pour afficher les régions de décision
X_tous = np.vstack((X_train, X_test))

xx, yy = np.meshgrid(
    np.linspace(X_tous[:, 0].min() - 0.5,
                X_tous[:, 0].max() + 0.5, 400),
    np.linspace(X_tous[:, 1].min() - 0.5,
                X_tous[:, 1].max() + 0.5, 400),
)

grille = np.column_stack((xx.ravel(), yy.ravel()))
grille_biais = np.column_stack((grille, np.ones(len(grille))))

classes_grille = np.argmax(grille_biais @ W.T, axis=1)
classes_grille = classes_grille.reshape(xx.shape)

# 3. Affichage des régions et des observations de test
couleurs = ["royalblue", "darkorange", "seagreen"]
cmap = ListedColormap(couleurs)

fig, ax = plt.subplots(figsize=(9, 6))

ax.contourf(
    xx, yy, classes_grille,
    levels=np.arange(len(couleurs) + 1) - 0.5,
    cmap=cmap,
    alpha=0.20,
)

for k, couleur in enumerate(couleurs):
    masque = y_vrai == k

    ax.scatter(
        X_test[masque, 0],
        X_test[masque, 1],
        color=couleur,
        edgecolors="black",
        s=80,
        label=iris.target_names[k],
    )

# Entourer les observations mal classées
if np.any(erreurs):
    ax.scatter(
        X_test[erreurs, 0],
        X_test[erreurs, 1],
        facecolors="none",
        edgecolors="red",
        linewidths=2,
        s=220,
        label="Mal classé",
    )

ax.set_xlabel("Longueur des pétales (cm)")
ax.set_ylabel("Largeur des pétales (cm)")
ax.set_title(f"Perceptron — résultats sur le test : {accuracy:.1%}")
ax.legend(title="Vraie classe")
ax.grid(alpha=0.20)

plt.tight_layout()
plt.show()