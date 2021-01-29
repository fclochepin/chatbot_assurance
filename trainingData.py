
from sklearn import metrics
from time import time
from sklearn.model_selection import KFold
import warnings
import pickle
import numpy as np
import sys
import pandas as pd

#Imports pour le machine learning
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

#Imports pour le deep-learning
from tensorflow.keras import backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

#On supprime les warnings inutiles provoqués par le Perceptron
warnings.filterwarnings('ignore') 

#récupération du training pattern
training_pattern = pickle.load(open('temp/training_pattern.pkl','rb'))
training_pattern = np.array(training_pattern, dtype=object)

#création échantillons tests et apprentissage
X = np.array(list(training_pattern[:,0]))
Y = np.array(list(training_pattern[:,1]))

#A ce niveau, on va devoir transformer Y : 
# -> Le ML nécessite un entier Classe 0, 1, 2 ...
# -> Le DL nécessite un vecteur : Classe [1,0,0...], [0,1,0...], [0,0,1...]
# A ce stade, Y est en vecteur, nous allons créer un vecteur Y_transformed pour contenir un entier

Y_transformed = []
for eachVector in training_pattern[:,1]:
    value = eachVector.index(1)
    Y_transformed.append(value)
df = pd.DataFrame(Y_transformed)
Y_transformed = np.array(df)

print("Training data and Test data are created !")

#on va créer des modèles à partir de différents classifieurs
nameClf = ["SVM","Classificateur naif bayésien", "K-plus-proches-voisins", "RandomForest", "Perceptron multi-couche", "Decision tree"]
listClf = [SVC(kernel='linear'), GaussianNB(), KNeighborsClassifier(), RandomForestClassifier(n_jobs=2, random_state=0, n_estimators=100), MLPClassifier(random_state=0, hidden_layer_sizes=10), DecisionTreeClassifier()]

#Création d'un KFOLD a 10 splits
kf = KFold(n_splits=10, shuffle=True)

print("********************************************************")
print("Vérification des modèles de ML par la méthode des KFOLDS")
print("********************************************************\n")

#on entraine et teste les différents modèles
for (index, eachClf) in enumerate(listClf):

    #Création de la liste des résultats
    accuracys = []

    #On créé 10 échantillons d'apprentissage et de test
    for train_index, test_index in kf.split(list(training_pattern[:,0])):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y_transformed[train_index], Y_transformed[test_index]

        #on entraîne le modèle grâce à notre échantillon d'apprentissage
        eachClf.fit(X_train, y_train)

        #pour chaque classifieur, on prédit les classes pour l'échantillon de test
        y_pred = eachClf.predict(X_test)

        #on enregistre la métrique de précision
        accuracys.append(metrics.accuracy_score(y_test, y_pred))

    #on calcule le taux de réussite
    print("Accuracy for", nameClf[index] ,":" ,round(np.mean(accuracys)*100,2), " %")

print("*************************")
print("Création d'un modèle SVM")
print("*************************\n")

# A ce stade, on détermine que le modèle SVM est le meilleur, nous allons donc créer un modèle avec un échantillon d'apprentissage à 80 %
X_train, X_test, y_train, y_test = train_test_split(X, Y_transformed, test_size=0.25)
SVMClassifier = SVC(kernel='linear')
SVMClassifier.fit(X_train, y_train)
y_pred = SVMClassifier.predict(X_test)
print("Accuracy for SVM trained model : " + str(round(metrics.accuracy_score(y_test, y_pred)*100,2)))


print("*************************************")
print("Création d'un modèle de deep learning")
print("*************************************\n")

#Le modèle suivant est trouvé sur internet
#L'idée est de créer un algorithme qui fait varier ces paramètres
#A faire : se documenter sur Keras, comprendre le fonctionnement général des réseaux de neurones
#Faire varier les paramètres pour étudier les changements observés
#Proposer un modèle optimal

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
# deep neural networds model
model = Sequential()
model.add(Dense(128, input_shape=(len(X_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax'))
# Compiling model. SGD with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#Training and saving the model 
hist = model.fit(np.array(X_train), np.array(y_train), epochs=400, batch_size=5, verbose=0)
print("model is created")
y_pred = model.predict(X_test)
result = []
for index, eachVector in enumerate(y_pred):
    result.append(np.where(eachVector==eachVector.max())[0] == np.where(y_test[index]==y_test[index].max())[0])

print("Accuracy for DL trained model : " + str(round((sum(result)/len(result))[0]*100,2)))

#On sauvegarde ce modèle
pickle.dump(SVMClassifier,open('temp/model.pkl','wb'))  
model.save('temp/model_dl.h5', hist)

