import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pydotplus

def encoder(X, y, train_feature, train=True):

    le = LabelEncoder()
    for i in train_feature:
        X[i] = le.fit_transform(X[i])
    if not train:
        return X
    y = le.fit_transform(y)

    return X, y

def graphic(clf, name):
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True,feature_names = train_feature, class_names=['0','1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png(name)

if __name__ == "__main__":
    
    
    # Task 4.1
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    
    train_feature = ['home', 'preseason', 'media']
    X = train_df[train_feature]
    X_test = test_df[train_feature]
    y = train_df['label']
    X, y = encoder(X, y, train_feature)
    X_test, y_true = encoder(X_test, test_df['label'], train_feature)
    

    
    # Task 4.2
    train_df = pd.read_csv("train_4-2.csv")
    
    train_feature = ['Outlook', 'Temperature', 'Humidity', 'Windy']
    X = train_df[train_feature]
    y = train_df['label']
    X, y = encoder(X, y, train_feature)    
    

    
    # Task 5
    train_df = pd.read_csv("train_5.csv")
    test_df = pd.read_csv("test_5.csv")
    
    train_feature = ['home', 'preseason', 'media']
    X = train_df[train_feature]
    y = train_df['label']
    X_test = test_df[train_feature]
    X, y = encoder(X, y, train_feature)    
    X_test, y_true = encoder(X_test, test_df['label'], train_feature)
    
    
    clf = DecisionTreeClassifier() # criterion = 'entropy'
    clf = clf.fit(X, y)

    # graphic(clf, 'task5_entropy.png')
    y_pred = clf.predict(X_test)
    print(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    result = precision_recall_fscore_support(y_true, y_pred)
    print(result)
    