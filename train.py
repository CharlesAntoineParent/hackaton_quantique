import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from qiskit import QuantumCircuit
from qiskit_algorithms.optimizers import COBYLA
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.utils import algorithm_globals

from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN


CDIR = os.path.dirname(os.path.abspath(__file__))

# Global variable to store iteration number
iterationIdx = 0

def getQuantumClassifier(seed):
    algorithm_globals.random_seed = seed

    ndim = 9

    # Construct QNN
    qc = QuantumCircuit(ndim)
    feature_map = ZZFeatureMap(ndim)
    ansatz = RealAmplitudes(ndim)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    # Construct estimator for measurement
    estimator_qnn = EstimatorQNN(
        circuit=qc, input_params=feature_map.parameters, weight_params=ansatz.parameters
    )

    # Callback to monitor progress during training
    def callback_graph(weights, loss):
        global iterationIdx
        print('Iteration %d: loss = %f' % (iterationIdx, loss))
        iterationIdx += 1

    # Construct neural network classifier
    model = NeuralNetworkClassifier(
        estimator_qnn, optimizer=COBYLA(maxiter=4), callback=callback_graph
    )

    return model

def getModels(seed):
    return {
            # Baseline classifiers
            'linear': LogisticRegression(),
            'svm': SVC(kernel='rbf', gamma='auto'),
            'decision-tree': DecisionTreeClassifier(),
            'mlp': MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(16, 2), random_state=seed),
            # Quantum classifier
            'qml': getQuantumClassifier(seed),
            }

def train():
    # Set seed for reproducibility
    seed = 42

    # Load data
    data = pd.read_csv(os.path.join(CDIR, "dataset", "challenge_dataset.csv"))
    x = data.drop("label", axis=1).values
    y = data["label"].values

    # Feature scaling and normalization of input features
    scaler = preprocessing.MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    x_normalized = preprocessing.normalize(x_scaled, norm='l2')

    # Model selection
    models = getModels(seed)
    for name, model in models.items():

        # Perform stratified k-fold cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        accuracies = []
        for train_index, test_index in skf.split(x, y):
            global iterationIdx
            iterationIdx = 0
            x_train_fold, x_test_fold = x_normalized[train_index], x_normalized[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]
            model.fit(x_train_fold, y_train_fold)
            accuracies.append(model.score(x_test_fold, y_test_fold))

        # Print statistics for model
        meanAccuracy = np.mean(accuracies)
        stdAccuracy = np.std(accuracies)
        print("Overall Accuracy (%s): %0.1f %% (std. %0.2f %%)" % (name, meanAccuracy * 100, stdAccuracy * 100))

if __name__ == "__main__":
    train()
    print("All done.")
