import pickle
import numpy as np
import random
import importDataset
from tensorflow.keras.models import load_model

answers = pickle.load(open('temp/answers.pkl','rb'))
words_dictionnary = pickle.load(open('temp/words_dictionnary.pkl','rb'))

#Ces 2 booléens permettent de choisir la méthode de pondération et l'utilisation du Deep Learning
TF_IDF = False
DL = True

if DL:
    model = load_model('temp/model_dl.h5')
else:
    model = pickle.load(open('temp/model.pkl','rb'))

# Fonction permettant de choisir une réponse adapatée en fonction de la classe prédite
def answerChat(number):
    if(DL):
        vectorAnswers = answers[np.where(number==number.max())[1][0]]
    else:
        vectorAnswers = answers[number[0]]
    entierAlea = random.randint(0, len(vectorAnswers)-1)
    return(vectorAnswers[entierAlea])

#Demande à l'utilisateur de saisir quelque chose
question = ""
while(question != "quit"):
    question = input("Vous : ")
    if TF_IDF:
        vectorSentence = importDataset.toTFIDF([importDataset.stopWordLemma(question)], False)
    else:
        vectorSentence = importDataset.toBagOfWords([question], words_dictionnary)

    #La fonction model.predict prédit : 
    #La classe 0 dans le cadre du ML
    #Un vecteur [1,0,0] dans le cadre du DL
    print("\n Chatbot : ", answerChat(model.predict(vectorSentence)), "\n")

