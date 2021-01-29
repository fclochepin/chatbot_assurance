# Readme

Voici quelques indications pour lancer le code.

### Installation des librairies
```bash
pip install numpy
pip install sklearn
pip install pandas
pip install pickle
pip install spaCy
pip install tensorflow
python -m spacy download fr_core_news_md
```


## Liste des fichiers

#### importDataset

Ce fichier permet d'importer le dataset et de créer les structures de données adéquates pour entraîner le modèle. **Il n'est pas nécessaire de le lancer pour interagir avec le robot.** 
> Le robot utilise actuellement la méthode de pondération Binaire. Pour utiliser une pondération TF_IDF, changer la valeur de la ligne 10 à True puis lancer l'import. 

#### trainingData

Ce fichier permet d'entraîner le Chatbot. On peut comparer les classifieurs les plus performants à l'aide d'un K-Fold. Le modèle le plus performant étant SVM, on entraîne un modèle (75% apprentissage) et on l'enregistre pour l'utiliser par la suite en conversation. Dans un second temps, on créé également un réseaux de neurones pour effectuer un apprentissage profond. Ce modèle est également enregistré. **Il n'est pas nécessaire de le lancer pour interagir avec le robot.** 

> Le robot utilise actuellement la méthode de pondération Binaire. Pour utiliser une pondération TF-IDF, il faut relancer ce script après avoir effectuer l'import. 

#### main

Ce fichier permet d'interagir avec le chatbot. En l'exécutant, le Chatbot attendra une saisie de votre part : 

> Vous : (Saisir le texte ici)

Pour arrêter la conversation avec le Chatbot, saisir QUIT

> Le robot utilise actuellement la méthode de pondération Binaire. Pour utiliser une pondération TF-IDF, il faut modifier la ligne 11 : TF_IDF = False. 

> Le robot utilise actuellement le modèle Deep Learning. Pour utiliser le modèle Machine Learning, il faut modifier la ligne 12 : TF_IDF = True. 
