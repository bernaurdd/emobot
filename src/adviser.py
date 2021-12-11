import os
import pandas as pd
import nltk

class Adviser():

    def __init__(self):
        path = os.getcwd() + '/ressources/wisdom.txt'
        self.advices = pd.read_csv(path, sep='\t', header=None)
        self.advices.columns = ['emotion', 'advice']
        self.seed = 0

    def getAdvice(self, emotion='neutral'):
        self.seed = self.seed + 1000
        return self.advices[self.advices['emotion'] == emotion].sample(random_state=self.seed , replace=True).iloc[0,1]

    def predictTopic(self, sentence):
        tokens = set(nltk.word_tokenize(sentence))
        keywords = []
        names = []
        names.append('relationship')
        keywords.append({'wife', 'relationship', 'love', 'girlfriend'})
        names.append('Carrer')
        keywords.append({'boss', 'job', 'money'})
        max = 0
        pred_topic = 'idk'
        for name, topic in zip(names, keywords):
            num = len(tokens.intersection(topic))
            if num >max:
                max = num
                pred_topic = name
        return pred_topic
