# This is a sample Python script.
import statistics
import colorama
from colorama import Fore
import pandas as pd
import numpy as np
import seaborn as sea
import warnings
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

def print_hi():

    print(Fore.RED + 'Hi, Here Dermatology Data Set')
    print(Fore.RESET)

def main():

    print(Fore.BLUE + '-> Capitolo1 | Problem Statement & Data Visualization <-')
    print(Fore.RESET)

    #Reading
    df = pd.read_csv(
        'Dataset/dermatology.data',
        delimiter=',',
        names=[
            'erythema', 'scaling', 'definite borders', 'itching', 'koebner phenomenon', 'polygonal papules', 'follicular papules',
            'oral mucosal involvement', 'knee and elbow involvement', 'scalp involvement', 'family history', 'melanin incontinence',
            'eosinophils in the infiltrate', 'PNL infiltrate', 'fibrosis of the papillary dermis', 'exocytosis', 'acanthosis',
            'hyperkeratosis', 'parakeratosis', 'clubbing of the rete ridges', 'elongation of the rete ridges', 'thinning of the suprapapillary epidermis',
            'spongiform pustule', 'munro microabcess', 'focal hypergranulosis', 'disappearance of the granular layer', 'vacuolisation and damage of basal layer',
            'spongiosis', 'saw-tooth appearance of retes', 'follicular horn plug', 'perifollicular parakeratosis', 'inflammatory monoluclear inflitrate', 'band-like infiltrate', 'age', 'class'
        ]
    )
    print(df.head(10).to_string())


    print(Fore.BLUE + '\n-> Capitolo2 | Exploratory Data Analysis <-')
    print(Fore.RESET)

    #General Information
    print('Shape: ', df.shape)
    print('Types:\n', df.dtypes)
    print('Missing Value: ', df.notna().values.any())

    #Missing Values Trick
    values = []
    for i in df['age']:
        if i != '?':
            values.append(i)
    values = list(map(int, values))
    mean = statistics.mean(values)
    df.replace('?', round(mean), inplace = True)

    #Discretization
    df['age'], _ = pd.factorize(df['age'], sort=True)

    #Target Vector & Design Matrix
    y = df['class']
    X = df.drop(['class'], axis=1)

    print(Fore.BLUE + '\n-> Capitolo3 | Feature Selection & Data Preparation <-')
    print(Fore.RESET)

    #Correlation Matrix
    plt.figure(figsize=(12,10), dpi=70)
    data = pd.DataFrame(df)
    corr = data.corr()
    matrix = np.triu(corr, k=1)
    sea.heatmap(corr, mask=matrix, cmap='YlGnBu', annot=False, square=True, linewidths=.5)
    plt.title('Correlation Heatmap', fontsize=20)
    plt.show()

    #Splitting: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    #Scaling
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)


    print(Fore.BLUE + '\n-> Capitolo4 | Modeling & Training <-')
    print(Fore.RESET)

    #Modeling
    models = [
        KNeighborsClassifier(weights='distance'),
        SVC(class_weight='balanced'),
        DecisionTreeClassifier(class_weight='balanced'),
        LogisticRegression(class_weight='balanced', multi_class='multinomial', solver='saga')
    ]

    models_names = [
        'K-NN',
        'SVM',
        'DT',
        'Softmax Regression'
    ]

    models_hparameters = [
        {'n_neighbors': list(range(3, 13, 2))},
        {'C': [1e-4, 1e-2, 3, 1e1, 1e2], 'kernel': ['linear', 'rbf'], 'gamma': [0.001, 0.0001]},
        {'criterion': ['gini', 'entropy']},
        {'penalty': ['l1', 'l2'], 'C': [1e-5, 5e-5, 1e-5, 5e-4, 1]}
    ]

    choosen_hparameters = []
    estimators = []

    for model, model_name, hparameters in zip(models, models_names, models_hparameters):
        print('\n', model_name)
        clf = GridSearchCV(estimator=model, param_grid=hparameters, scoring='accuracy', cv=5)
        clf.fit(X_train, y_train)
        choosen_hparameters.append(clf.best_params_)
        estimators.append((model_name, clf))
        print('Accuracy: ', clf.best_score_)
        for hparam in hparameters:
            print(f'\t The best choice for parameter {hparam}: ', clf.best_params_.get(hparam))


    print(Fore.BLUE + '\n-> Capitolo5 | Cross Validation <-')
    print(Fore.RESET)

    #clf_stack = StackingClassifier(estimators=estimators, final_estimator=DecisionTreeClassifier())
    final_model = models[1]

    #CrossValidation
    scores = cross_validate(final_model, X_train, y_train, cv=5, scoring=('f1_weighted', 'accuracy'))
    print('The cross-validated weighted F1-score of the model is ', np.mean(scores['test_f1_weighted']))
    print('The cross-validated Accuracy of the model is ', np.mean(scores['test_accuracy']))
    print('\n')

    #Wrapper
    sfs = SequentialFeatureSelector(final_model, cv=2)
    sfs.fit(X_train, y_train)
    #print('Feature selezionate: ', sfs.get_support())
    print('N. Feature selezionate: ', sfs.get_support()[sfs.get_support()].size)

    X_train = sfs.transform(X_train)

    print(Fore.BLUE + '\n-> Capitolo6 | Testing & Evaluation <-')
    print(Fore.RESET)

    final_model.fit(X_train, y_train)

    X_test = scaler.transform(X_test)
    X_test = sfs.transform(X_test)

    y_pred = final_model.predict(X_test)

    target_names = ['Psoriasi', 'Dermatite Seberroica', 'Lichen Planus', 'Pitiriasi Rosea', 'Dermatite Cronica','Pitiriasi Rubra Pilaris']
    print("\nClassification Report:\n", metrics.classification_report(y_test, y_pred, target_names=target_names))
    print('Accuracy is ', accuracy_score(y_test, y_pred))
    print('Precision is ', precision_score(y_test, y_pred, average='weighted'))
    print('Recall is ', recall_score(y_test, y_pred, average='weighted'))
    print('F1-Score is: ', f1_score(y_test, y_pred, average='weighted'))

if __name__ == '__main__':
    print_hi()
    main()

