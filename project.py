
from data_utils import readfile
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer , HashingVectorizer , CountVectorizer
from scipy import sparse
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import KFold , cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import recall_score ,f1_score , precision_score ,accuracy_score
from sklearn.metrics import classification_report
import pandas as pd
import sys


class Model:
    def __init__(self,train,test):
        self.test=test
        self.train=train
        
    def tfidfvec(self , ngram =(1,3)):
        self.count_vec1 = TfidfVectorizer(ngram) 
        self.count_vec2 = TfidfVectorizer(ngram) 
        
    def countvec(self , ngram =(1,3)):
        self.count_vec1 = TfidfVectorizer(ngram) 
        self.count_vec2 = TfidfVectorizer(ngram) 
    
    def train_model(self):
        train_count_L=self.count_vec1.fit_transform(self.train[0])
        train_count_R=self.count_vec2.fit_transform(self.train[1])
        self.train_count=sparse.hstack((train_count_L,train_count_R))
        
        test_count_L=self.count_vec1.transform(self.test[0])
        test_count_R=self.count_vec2.transform(self.test[1])
        self.test_count=sparse.hstack((test_count_L,test_count_R))

# feature_selection: -----------------------

    def mutualinfo(self , top_values=100):
        train_feature= mutual_info_classif(self.train_count, self.train[2],n_neighbors=3)

        # train_feature.tofile('train_feature.dat')
        # train_feature = np.fromfile('train_feature.dat')
        # Only use the top 100 values:
        flat=train_feature.flatten()
        flat.sort()
        top_100=flat[-top_values]

        train_selection = sparse.coo_matrix((self.train_count.shape[0],0), dtype=np.int8)
        test_selection = sparse.coo_matrix((self.test_count.shape[0],0), dtype=np.int8)
        for i , value in enumerate(train_feature):
            if value >= top_100:
                train_sub=self.train_count.todense()[:,i]
                train_sub=sparse.coo_matrix(train_sub)
                train_selection=sparse.hstack((train_selection,train_sub))
                
                test_sub = self.test_count.todense()[:,i]
                test_sub= sparse.coo_matrix(test_sub)
                test_selection=sparse.hstack((test_selection,test_sub))
        
        print('train_count.shape',self.train_count.shape )
        self.train_count=train_selection
        print('train_count.shape',self.train_count.shape)
        self.test_count=test_selection
        
# # feature_selection: ----------------------- Frequency
    def frequencyfs(self, top_values=100):

        dic_freq=self.train_count.sum(axis=0)
        dic_freq=np.squeeze(np.asarray(dic_freq)) #to array
        sort_dic_freq=dic_freq
        sort_dic_freq.sort()
        top_100=sort_dic_freq[-top_values]

        train_selection = sparse.coo_matrix((self.train_count.shape[0],0), dtype=np.int8)
        test_selection = sparse.coo_matrix((self.test_count.shape[0],0), dtype=np.int8)
        for i , value in enumerate(dic_freq):
            if value >= top_100:
                train_sub=self.train_count.todense()[:,i]
                train_sub=sparse.coo_matrix(train_sub)
                train_selection=sparse.hstack((train_selection,train_sub))
                
                test_sub = self.test_count.todense()[:,i]
                test_sub= sparse.coo_matrix(test_sub)
                test_selection=sparse.hstack((test_selection,test_sub))
        print('train_count.shape',self.train_count.shape )
        self.train_count=train_selection
        print('train_count.shape',self.train_count.shape)
        self.test_count=test_selection

        
        
        
### -----------------------------------------
    def learn(self):
        # from sklearn.cross_validation import KFold , cross_val_score
        
        clf = MultinomialNB()
        self.scores=cross_val_score(clf,self.train_count,self.train[2],cv=5,scoring='accuracy')
        print("train accuracy" , self.scores.mean())
        
        import pandas as pd
        clf1 = MultinomialNB().fit(self.train_count,self.train[2])
        self.y_pred = clf1.predict(self.test_count)

    def evaluate(self , outputname = 'out.txt'):

        self.y_pred = pd.DataFrame(self.y_pred,columns=['label'])
        self.y_value = pd.DataFrame(self.test[2],columns=['Gold_label'])

        self.recall = recall_score(self.y_value, self.y_pred , average='weighted')
        self.precision_score = recall_score(self.y_value, self.y_pred , average='weighted')
        self.f1_score = f1_score(self.y_value, self.y_pred , average='weighted')
        self.accuracy=accuracy_score(self.y_value, self.y_pred )


        print('Recall score: {0:0.2f}'.format(self.recall))
        print('precision score: {0:0.2f}'.format(self.precision_score))
        print('f1 score: {0:0.2f}'.format(self.f1_score))
        print('accuracy : {0:0.2f}'.format(self.accuracy))
        target_names = ['contradiction', 'neutral' ,'entailment']
        print(classification_report(self.y_value, self.y_pred, target_names=target_names))
           
        with open(outputname, 'w') as f:
            print("train accuracy" , self.scores.mean())
            print('Recall score: {0:0.2f}'.format(self.recall),file=f)
            print('precision score: {0:0.2f}'.format(self.precision_score),file=f)
            print('f1 score: {0:0.2f}'.format(self.f1_score),file=f)
            print('accuracy : {0:0.2f}'.format(self.accuracy),file=f)
            print(classification_report(self.y_value, self.y_pred, target_names=target_names),file=f)
        return(self.accuracy)


def main():
    
    if sys.argv[1] == 'test':
        test=readfile("test_stem")
        print("test mode")
    else: 
        test=readfile("dev_stem")
        
    train=readfile("train_stem")
    Model_Class = Model(train,test)
    Model_Class.tfidfvec( ngram =(2,2))
    Model_Class.train_model()
    Model_Class.learn()
    acc=Model_Class.evaluate()
    

    

if __name__ == '__main__':
    main()

