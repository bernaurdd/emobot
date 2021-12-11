import os
import pandas as pd

class Adviser():

    def __init__(self):
        path = os.getcwd() + '/ressources/wisdom.txt'
        self.advices = pd.read_csv(path, sep='\t', header=None)
        self.advices.columns = ['emotion', 'advice']

    def getAdvice(self, emotion='neutral'):
        return self.advices[self.advices['emotion'] == emotion].sample().iloc[0,1]
