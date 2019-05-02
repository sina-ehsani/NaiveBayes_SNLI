
from data_utils import readfile
import numpy as np 

test=readfile("dev")
train=readfile("train")

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer , HashingVectorizer , CountVectorizer
# count_vec1 = TfidfVectorizer() #ngram_range=(1,5))
# count_vec2 = TfidfVectorizer()# ngram_range=(1,5))

count_vec1= CountVectorizer()#stop_words='english', ngram_range=(1,3) )
count_vec2= CountVectorizer()#stop_words='english' , ngram_range=(1,3))


from scipy import sparse
train_count_L=count_vec1.fit_transform(train[0])
train_count_R=count_vec2.fit_transform(train[1])
train_count=sparse.hstack((train_count_L,train_count_R))

test_count_L=count_vec1.transform(test[0])
test_count_R=count_vec2.transform(test[1])
test_count=sparse.hstack((test_count_L,test_count_R))

# # feature_selection: ----------------------- MI
# from sklearn.feature_selection import mutual_info_classif
# # train_feature= mutual_info_classif(train_count, train[2],n_neighbors=3)

# # train_feature.tofile('train_feature.dat')
# train_feature = np.fromfile('train_feature.dat')

# # Only use the top 100 values:
# flat=train_feature.flatten()
# flat.sort()
# top_100=flat[-500]

# train_selection = sparse.coo_matrix((train_count.shape[0],0), dtype=np.int8)
# test_selection = sparse.coo_matrix((test_count.shape[0],0), dtype=np.int8)
# for i , value in enumerate(train_feature):
#     if value >= top_100:
#         train_sub=train_count.todense()[:,i]
#         train_sub=sparse.coo_matrix(train_sub)
#         train_selection=sparse.hstack((train_selection,train_sub))
        
#         test_sub = test_count.todense()[:,i]
#         test_sub= sparse.coo_matrix(test_sub)
#         test_selection=sparse.hstack((test_selection,test_sub))
        
# print('train_count.shape',train_count.shape )
# train_count=train_selection
# print('train_count.shape',train_count.shape)
# test_count=test_selection
### -----------------------------------------


# # feature_selection: ----------------------- Frequency

dic_freq=train_count.sum(axis=0)
dic_freq=np.squeeze(np.asarray(dic_freq)) #to array
sort_dic_freq=dic_freq
sort_dic_freq.sort()
top_100=sort_dic_freq[-100]

train_selection = sparse.coo_matrix((train_count.shape[0],0), dtype=np.int8)
test_selection = sparse.coo_matrix((test_count.shape[0],0), dtype=np.int8)
for i , value in enumerate(dic_freq):
    if value >= top_100:
        train_sub=train_count.todense()[:,i]
        train_sub=sparse.coo_matrix(train_sub)
        train_selection=sparse.hstack((train_selection,train_sub))
        
        test_sub = test_count.todense()[:,i]
        test_sub= sparse.coo_matrix(test_sub)
        test_selection=sparse.hstack((test_selection,test_sub))
print('train_count.shape',train_count.shape )
train_count=train_selection
print('train_count.shape',train_count.shape)
test_count=test_selection

### -----------------------------------------



# from sklearn.cross_validation import KFold , cross_val_score
from sklearn.model_selection import KFold , cross_val_score
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

## Naive Bayees
clf = MultinomialNB()
scores=cross_val_score(clf,train_count,train[2],cv=5,scoring='accuracy')
print("score_mean" , scores.mean())

clf1 = MultinomialNB().fit(train_count,train[2])
y_pred = clf1.predict(test_count)
y_pred = pd.DataFrame(y_pred,columns=['label'])
y_value = pd.DataFrame(test[2],columns=['Gold_label'])

# ## KNN
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=50)
# scores=cross_val_score(knn,train_count,train[2],cv=5,scoring='accuracy')
# print("score_mean" , scores.mean())

# knn1=KNeighborsClassifier(n_neighbors=20).fit(train_count,train[2])
# y_pred = knn1.predict(test_count)
# y_pred = pd.DataFrame(y_pred,columns=['label'])
# y_value = pd.DataFrame(test[2],columns=['Gold_label'])



from sklearn.metrics import recall_score ,f1_score , precision_score ,accuracy_score
recall = recall_score(y_value, y_pred , average='weighted')
precision_score = recall_score(y_value, y_pred , average='weighted')
f1_score = f1_score(y_value, y_pred , average='weighted')
accuracy=accuracy_score(y_value, y_pred )


print('Recall score: {0:0.2f}'.format(recall))
print('precision score: {0:0.2f}'.format(precision_score))
print('f1 score: {0:0.2f}'.format(f1_score))
print('accuracy : {0:0.2f}'.format(accuracy))

from sklearn.metrics import classification_report
target_names = ['contradiction', 'neutral' ,'entailment']
print(classification_report(y_value, y_pred, target_names=target_names))


with open('out_dev_stem_frequency.txt', 'w') as f:
    print("train_accu" , scores.mean(),file=f)
    print('Recall score: {0:0.2f}'.format(recall),file=f)
    print('precision score: {0:0.2f}'.format(precision_score),file=f)
    print('f1 score: {0:0.2f}'.format(f1_score),file=f)
    print('accuracy : {0:0.2f}'.format(accuracy),file=f)
    print(classification_report(y_value, y_pred, target_names=target_names),file=f)



frames=[y_value,y_pred]
result=pd.concat(frames,axis=1)

# result.to_csv("snli-test.csv", sep='\t')
result['compare'] = np.where(result['Gold_label'] == result['label'], 0 , 1)

acc=(len(result)-sum(result.compare))/len(result)

print("acc" , acc)

