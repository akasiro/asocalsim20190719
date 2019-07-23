from gensim.similarities import SparseMatrixSimilarity
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
import numpy as np
import jieba
class tfidfmodel_cal():
    def __init__(self,listoftextlist):
        self.build_model(listoftextlist)
    def build_model(self,listoftextlist):
        self.dictionary = Dictionary(listoftextlist)
        self.num_features = len(self.dictionary.token2id)
        self.corpus = [self.dictionary.doc2bow(text) for text in listoftextlist]
        self.tfidf = TfidfModel(self.corpus)
        self.index = SparseMatrixSimilarity(self.tfidf[self.corpus], self.num_features)

    def cal_tfidf(self, textlist, typecode =None):
        new_vec = self.dictionary.doc2bow(textlist)
        sim = self.index[self.tfidf[new_vec]]
        if typecode == None:
            simlist = [i for i in sim if i < 0.99]
            if len(simlist) == 0:
                return np.nan
            else:
                return max(simlist)
        else:
            return sim[typecode]
            

if __name__ == '__main__':
    text1 = '无痛人流并非无痛'
    text2 = '北方人流浪到南方'
    text2 = ''
    texts = [text1, text2]
    texts = [text1,'']
    keyword = '无痛人流并非无痛'
    texts = [jieba.lcut(text) for text in texts]
    texts = [jieba.lcut(text1),[]]

    new_vec = jieba.lcut(keyword)

    m = tfidfmodel_cal(texts)
    print(m.cal_tfidf(new_vec, 0))
    print(m.cal_tfidf(new_vec,  1))
    print(m.cal_tfidf(new_vec))

