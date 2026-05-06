import numpy as np
from pyparsing import alphas
import scipy 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import graphviz
import dtreeviz
import optuna
import shap

print(f'Package versions: \n')

print(f'numpy {np.__version__}')
print(f'scipy {scipy.__version__}')
print(f'pandas {pd.__version__}')
print(f'seaborn {sns.__version__}')
print(f'scikit-learn {sklearn.__version__}')
print(f'graphviz {graphviz.__version__}')
print(f'dtreeviz {dtreeviz.__version__}')
print(f'optuna {optuna.__version__}')
print(f'shap {shap.__version__}')


from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
# Model selection and evaluation tools


from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_score, average_precision_score, log_loss

from IPython.display import display

# Optional configuration

# This is to clear the warnings from the notebook, usually we should leave this on


from sklearn.tree import DecisionTreeClassifier

#X_train is predictors/inputs and y_train is the target churn variable/binary labels
def train_decision_tree(X_train, y_train, criteria='entropy', leaf_nodes = 5):

    clf = DecisionTreeClassifier(criterion=criteria, max_leaf_nodes=leaf_nodes)
    clf.fit(X_train, y_train)
    return clf

import graphviz
from sklearn.tree import export_graphviz
#features is a list of the predictor names that we use  to split the data in the decision tree
def visualize_tree(clf, features): 
    import os
    # Add Graphviz bin directory to PATH so dot executable can be found
    graphviz_path = r"C:\Program Files\Graphviz\bin"
    if graphviz_path not in os.environ['PATH']:
        os.environ['PATH'] = graphviz_path + os.pathsep + os.environ['PATH']
    
    dot_data = export_graphviz(clf, out_file=None, impurity=False, feature_names=features, 
                               class_names=['No-churn', 'Churn'], rounded=True)
    graph = graphviz.Source(dot_data)
    return graph

def sophisticated_tree_viz(clf, X_train, y_train, features):
   
    target = 'Churn'
    viz = dtreeviz.model(clf, X_train, y_train, target_name=target, feature_names = features,
    class_names=['Stay', 'Churn'])  
              
    viz.view(scale=2)


from sklearn.tree import plot_tree

def plot_tree(clf, features): 

    plot_tree(clf, feature_names = features, class_names=['No-churn','Churn'], impurity=False,
              rounded=True, filled = True)
    plt.show()


def cost_complexity_pruning(X_train, y_train, features):

    model = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=5)
    path = model.cost_complexity_pruning_path(X_train, y_train)
    alphas = path.ccp_alphas

    model = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=5)

    search_space = {
        'ccp_alpha': alphas,
    }

    # Select alpha by grid search
    tree_search = GridSearchCV(model, search_space, cv = 5 , scoring='neg_log_loss')
    tree_search.fit(X_train, y_train)
    tree = tree_search.best_estimator_

    print('Best parameters found by grid search:', tree_search.best_params_, '\n')

    dot_data = export_graphviz(tree, out_file=None , impurity=False, feature_names = features,
                           class_names=['No-churn','Churn'], rounded=True) 
    graph = graphviz.Source(dot_data)
    graph.render('tree02') # saves tree to a file
    graph