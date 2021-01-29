import json
import re
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import pandas as pd

TF_IDF = False

#Chargement des stopwords et de la lemmatisation Spacy
nlp = spacy.load('fr_core_news_md')
french_stopwords = stopwords.words('french')

#création du dictionnaire de mots et du dictionnaire de classes
words_dictionnary = []
categories = []
answers = []
listSentences = []

#création de la liste associant chaque groupe de mots à une catégorie
pattern_category = []

#récupération des données dans un objet dictionnaire
with open('data/dataset-assurance.json', encoding='utf-8') as f:
    data = json.load(f)

#Cette fonction permet de mettre en minuscule le mot, retirer le stopwords et lemmatiser
def stopWordLemma(sentence):
    newSentence = ""
    #Pour chaque mot dans la phrase
    for eachWord in sentence.split(" "):
        #On met en minuscule
        eachWord = eachWord.lower()
        #Si ce n'est pas un stopwords
        if(eachWord not in french_stopwords):
            #on lemmatise le mot
            if(nlp(eachWord)):
                eachWord = nlp(eachWord)[0].lemma_
                newSentence += eachWord + " "
    return newSentence[:-1]

#### METHODE BAG OF WORDS (Pondération binaire) ####

#alimentation du word_dictionnary, de categories et de pattern_category
for eachElem in data["intents"]:
    #alimentation de la liste des catégories
    categories.append(eachElem["tag"])
    answers.append(eachElem["responses"])
    for eachSentence in eachElem["patterns"]:
        #suppression des ponctuations
        eachSentence = re.sub(r'[^\w\s]',' ', eachSentence)
        #suppression des doubles espaces
        eachSentence = re.sub(' +', ' ', eachSentence)
        #Suppression des espaces de fin de ligne
        eachSentence = re.sub(r'[ \t]+$', '', eachSentence)
        #ajout d'un tuple dans pattern_category
        pattern_category.append((eachSentence, eachElem["tag"]),)
        #ajout de la phrase dans la liste des phrases
        listSentences.append(eachSentence)
        for eachWord in eachSentence.split():
            #on mets tous les mots en minuscule
            eachWord = eachWord.lower()
            #on vérifie que ce n'est pas un stopWords
            if(eachWord not in french_stopwords):
                #on lemmatise le mot
                eachWord = nlp(eachWord)[0].lemma_
                #ajout du mot
                if(eachWord not in words_dictionnary):
                    words_dictionnary.append(eachWord)

#Cette méthode convertit une phrase en un vecteur pondéré binairement
def toBagOfWords(listSentences, words):
    denseList = list()
    for eachSentence in listSentences:
        #initialisation de la liste contenant 0 et 1
        listWord = [0] * len(words)
        for eachWord in stopWordLemma(eachSentence).split(" "):
            #Vérification que le mot existe bien dans le dico global
            if(eachWord in words):
                #on ajoute un 1 à la position du mot
                position = words.index(eachWord)
                listWord[position] = 1
        denseList.append(listWord)
    return(pd.DataFrame(denseList, columns=words))

#### METHODE TF IDF ####

def toTFIDF(listSentences, keep=True):
    if(keep):
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(listSentences)
    else:
        vectorizer = pickle.load(open('temp/vectorizer.pkl','rb'))
        vectors = vectorizer.transform(listSentences)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()

    #Ce tableau pandas contient pour chaque ligne i le vecteur pondéré par TF-IDF de la phrase i du Dataset
    df = pd.DataFrame(denselist, columns=feature_names)
    if(keep):
        pickle.dump(vectorizer, open('temp/vectorizer.pkl','wb'))

    return df

#On convertit l'ensemble des phrases 
for eachSentence in listSentences:
    eachSentence = stopWordLemma(eachSentence)

if TF_IDF:
    df = toTFIDF(listSentences)
else:
    df = toBagOfWords(listSentences, words_dictionnary)

#Cette fonction permet de récupérer l'index de la catégorie
def categorize(categorie, categories):
    result = [0] * len(categories)
    result[categories.index(categorie)] = 1
    return result
    
#on remplit la liste qui associe vecteur de catégorie (0 et 1) à une phrase vectorisée
training_pattern = []
for index, eachTuple in enumerate(pattern_category):

    #2 méthodes proposées, binaire ou TF-IDF
    if(TF_IDF):
        training_pattern.append((df.iloc[index],categorize(eachTuple[1], categories)),) #Phrase véctorisée par TF-IDF
    else:
        #Phrase véctorisée binaire (0 = le mot n'est pas présent dans la phrase, 1 = le mot est présent)
        training_pattern.append((df.iloc[index],categorize(eachTuple[1], categories)),)
     
#On enregistre tous les fichiers dans un répertoire temporaire
pickle.dump(training_pattern,open('temp/training_pattern.pkl','wb'))  
pickle.dump(words_dictionnary,open('temp/words_dictionnary.pkl','wb'))  
pickle.dump(french_stopwords,open('temp/french_stopwords.pkl','wb'))  
pickle.dump(answers, open('temp/answers.pkl','wb'))
