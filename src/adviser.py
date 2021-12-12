import os
import pandas as pd
import nltk

class Adviser():

    def __init__(self):
        path1 = os.getcwd() + '/ressources/prompts.txt'
        path2 = os.getcwd() + '/ressources/wisdom.txt'
        self.prompts = pd.read_csv(path1, sep=';', header=None)
        self.prompts.columns = ['emotion', 'prompts']
        self.advices = pd.read_csv(path2, sep=';', header=None)
        self.advices.columns = ['emotion', 'topic', 'advice']
        self.seed = 0

    def getPrompt(self, emotion):
        self.seed = self.seed + 1000
        return self.prompts[self.prompts['emotion'] == emotion].iloc[0, 1]

    def getAdvice(self, emotion='neutral', topic='relationship'):
        self.seed = self.seed + 1000
        applicableAdvice = self.advices[(self.advices['emotion'] == emotion) & (self.advices['topic'] == topic)]
        return applicableAdvice.sample(random_state=self.seed , replace=True).iloc[0,2]

    def predictTopic(self, sentence):
        tokens = set(nltk.word_tokenize(sentence))
        keywords = []
        names = []
        names.append('relationship')
        keywords.append({'wife', 'relationship', 'love', 'girlfriend'})
        names.append('career')
        keywords.append({'boss', 'job', 'money'})
        max = 0
        pred_topic = 'idk'
        for name, topic in zip(names, keywords):
            num = len(tokens.intersection(topic))
            if num >max:
                max = num
                pred_topic = name
        return pred_topic

