from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from io import BytesIO
import base64
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    classifier_name = request.form['classifier']
    param1 = request.form['param1']
    param2 = request.form['param2']
    # Adicione mais parâmetros conforme necessário

    # Carregue sua base de dados e realize o pré-processamento
    # Substitua os dados abaixo pelos seus próprios
    X, y = load_data()

    # Divida os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Escolha e configure o classificador
    if classifier_name == 'KNN':
        classifier = KNeighborsClassifier(n_neighbors=int(param1))
    elif classifier_name == 'SVM':
        classifier = SVC(C=float(param1), kernel=param2)
    elif classifier_name == 'MLP':
        classifier = MLPClassifier(hidden_layer_sizes=(int(param1),), max_iter=1000)
    elif classifier_name == 'DT':
        classifier = DecisionTreeClassifier(max_depth=int(param1))
    elif classifier_name == 'RF':
        classifier = RandomForestClassifier(n_estimators=int(param1))

    # Treine o classificador
    classifier.fit(X_train, y_train)

    # Faça previsões no conjunto de teste
    y_pred = classifier.predict(X_test)

    # Calcule as métricas
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average='macro')
    recall = metrics.recall_score(y_test, y_pred, average='macro')
    f1_score = metrics.f1_score(y_test, y_pred, average='macro')

    # Crie e salve a matriz de confusão como uma imagem
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks(range(len(set(y))), set(y))
    plt.yticks(range(len(set(y))), set(y))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Salve a imagem como bytes
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.read()).decode('utf-8')

    plt.close()

    return render_template('results.html', accuracy=accuracy, precision=precision, recall=recall, f1_score=f1_score, confusion_matrix=img_str)

def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y

if __name__ == '__main__':
    app.run(debug=True)
