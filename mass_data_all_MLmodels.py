# import dependencies
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# drop ID col
df = pd.read_csv('mass_data_cleanup.csv')
df = df.drop(['Unnamed: 0'],axis=1)
df.head()

X=df[['Age','Shape','Margin',"Density"]].values
y=df[['Severity']].values

#we have dependent variable y,and it is binary categorical var, let's try some supervised ML classification modles
#try Decision tree,
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state = 0)
clf.fit(X_train, y_train)
print(f'Decision Tree Score:',clf.score(X_test, y_test))  # 0.72



'''
#display the decsion tree,the complexity is a nature way to tell that decison tree is not a best solution
#the feature might not split easily
from IPython.display import Image  
from sklearn.externals.six import StringIO  
import pydotplus
import Graphviz 
features =['Age','Shape','Margin',"Density"]
dot_data = StringIO()  
clf=clf.fit(X_test, y_test)
tree.export_graphviz(clf, out_file=dot_data,  
                         feature_names=features)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
'''

#instead of simple train/test set,use  k-folds cross validation to get a better measure of our model 
 
from sklearn.model_selection import cross_val_score 
dt_scores = cross_val_score(clf, X,y, cv=10)
DT_score=dt_scores.mean()  # 0.73
# create a dictionary to plot all models's score
plot_xy={}
plot_xy.update({'DecisionTree':DT_score})


#try RandomForestClassifier 
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200)
rf = rf.fit(X_train, y_train.ravel())
print(f'RandomForest Classfier Score:',rf.score(X_test, y_test.ravel())) #0.75
rf_scores = cross_val_score(rf, X,y, cv=10)
RF_score=rf_scores.mean() #0.77
plot_xy.update({'RF':RF_score})

# Support vector machine linear classifier
from sklearn.svm import SVC 
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
all_features_scaled = scaler.fit_transform(X)
all_features_scaled
svc_rbf_model = SVC(kernel='rbf')
svc_rbf_model.fit(X_train, y_train.ravel())
sv_rbf_scores = cross_val_score(svc_rbf_model, all_features_scaled,y, cv=10)
sv_rbf_score=sv_rbf_scores.mean()  #0.80
plot_xy.update({'svc_rbf':sv_rbf_score})

model = SVC(kernel='linear')
model.fit(X_train, y_train.ravel())
sv_l_scores = cross_val_score(model,all_features_scaled,y, cv=10)
sv_l_score=sv_l_scores.mean() #0.80
plot_xy.update({'svc_linear':sv_l_score})



'''
model = SVC(kernel='poly')
model.fit(X_train, y_train.ravel())
sv_p_scores = cross_val_score(model,all_features_scaled,y, cv=10)
sv_p_scores.mean()


model = SVC(kernel='sigmoid')
model.fit(X_train, y_train.ravel())
sv_s_scores = cross_val_score(model, X,y, cv=10)
sv_s_scores.mean()
'''


from sklearn.neighbors import KNeighborsClassifier

# KNN is all about k
# Loop through different k values to see which has the highest accuracy
# Note: We only use odd numbers because we don't want any ties
train_scores = []
test_scores = []
for k in range(1, 20, 2):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train.ravel())
    
    train_score = knn.score(X_train, y_train.ravel())
    test_score = knn.score(X_test, y_test.ravel())
    
    train_scores.append(train_score)
    test_scores.append(test_score)
    
    print(f"k: {k}, Train/Test Score: {train_score:.3f}/{test_score:.3f}")
    
import matplotlib.pyplot as plt    
plt.plot(range(1, 20, 2), train_scores, marker='o')
plt.plot(range(1, 20, 2), test_scores, marker="x")
plt.xlabel("k neighbors")
plt.ylabel("Testing accuracy Score")
plt.show()


knn = KNeighborsClassifier(n_neighbors=10)  #we choose k=10
knn_scores= cross_val_score(knn, X,y, cv=10)
knn_score=knn_scores.mean()  #0.78
plot_xy.update({'KNN':knn_score})

from sklearn.naive_bayes import MultinomialNB
NB= MultinomialNB()
from sklearn.preprocessing import MinMaxScaler
#Naive Bayes doesn't accept negative data,it can't handle negative inputs. 0-1 yields the best results.
scaler=MinMaxScaler()
data_scalered=scaler.fit_transform(X)
NB_scores=cross_val_score(NB,data_scalered,y,cv=10)
NB_score=NB_scores.mean()  #0.78
plot_xy.update({'MultinomialNB':NB_score})

# svc_kernal rbf is the winner in those above complex models. We will also try simple logistical regression
from sklearn.linear_model import LogisticRegression
lg_clf = LogisticRegression(random_state=0)
lg_scores=cross_val_score(lg_clf,X,y,cv=10)
lg_scores.mean() #0.8

from sklearn.linear_model import LogisticRegression
lg_clf = LogisticRegression(random_state=0)
lg_scores=cross_val_score(lg_clf,data_scalered,y,cv=10)
lg_score=lg_scores.mean() #0.8  simple is better
plot_xy.update({'LR':lg_score})
 


#finally we will try  build  neural networ,Now set up an actual MLP model using Keras:
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.model_selection import cross_val_score

def create_model():
    model = Sequential()
    #4feature inputs going into an 8-unit layer 
#     model.add(Dense(8, input_dim=4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(8, input_dim=4, kernel_initializer='normal', activation='relu'))
    # Another hidden layer of 4 units
    model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    # Output layer with a binary classification (malignant or begnial)
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Wrap our Keras model in an estimator compatible with scikit_learn
estimator = KerasClassifier(build_fn=create_model, epochs=10, verbose=0)
# Now we can use scikit_learn's cross_val_score to evaluate this model identically to the others
ANN_cv_scores = cross_val_score(estimator, all_features_scaled, y, cv=10)
ANN_cv_score=ANN_cv_scores.mean()  #0.80 try different topology,  choices of hyperparameters
plot_xy.update({'ANN':ANN_cv_score})

## Plot validation,compare all models, the simple the better,so we'll save .h5

import matplotlib.pyplot as plt
plot_xy_sorted= sorted(plot_xy.items(), key=lambda kv: kv[1])
fig,_=plt.subplots()
plt.bar(range(len(plot_xy)), list(plot_xy.values()), align='center')
plt.xticks(range(len(plot_xy)), list(plot_xy.keys()),rotation='vertical')
plt.title('Model cross_val_score')
plt.show()
fig.savefig('all_ML_models_validation.png', dpi=100, bbox_inches="tight")




