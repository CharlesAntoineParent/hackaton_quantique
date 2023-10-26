import os
import logging
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

from qiskit import Aer, QuantumCircuit, transpile, assemble, Aer, execute
from qiskit_algorithms.optimizers import COBYLA
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.utils import algorithm_globals
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import Sampler, Estimator
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.utils.loss_functions import L2Loss
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B, SPSA
from qiskit.quantum_info import SparsePauliOp

from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN


CDIR = os.path.dirname(os.path.abspath(__file__))

# Global variable to store iteration number
iterationIdx = 0

def getQuantumClassifier(ndim, estimator=None, seed=0):
    algorithm_globals.random_seed = seed

    sampler = Sampler()

    l2_loss = L2Loss()

    nb_qubits = ndim
    nb_layers = 2

    ansatz = RealAmplitudes(nb_qubits, reps=nb_layers)
    feature_map = ZZFeatureMap(nb_qubits)
    qc = feature_map.compose(ansatz)

    x0 = np.random.random(ansatz.num_parameters)

    objective_func_vals = []
    def callback_graph(weights, obj_func_eval):
        logging.info("Loss value: %f" % obj_func_eval)
        objective_func_vals.append(obj_func_eval)

    def parite(x):
        return "{:b}".format(x).count("1") % 2

    sampler = Sampler()
    qnn = SamplerQNN(
        circuit=qc,
        sampler=sampler,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        interpret=parite,
        output_shape=2
    )

    model = NeuralNetworkClassifier(
        qnn,
        optimizer=COBYLA(maxiter=50),
        loss=l2_loss,
        initial_point=x0,
        callback=callback_graph)

    return model

def getModels(ndim, seed):
    return {
            # Baseline classifiers
            'linear': LogisticRegression(),
            'svm': SVC(kernel='rbf', gamma='auto'),
            'decision-tree': DecisionTreeClassifier(),
            'mlp': MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(16, 2), random_state=seed),
            # Quantum classifier
            'qml': getQuantumClassifier(ndim, seed=seed),
            }

def preprocessData(x, ndim=None):
    if ndim is not None:
        # Reduce dimensionality
        pca = PCA(n_components=3)
        x = pca.fit_transform(x)
    else:
        scaler = preprocessing.MinMaxScaler()
        x = scaler.fit_transform(x)

    x = preprocessing.normalize(x, norm='l2')
    return x

def train():
    # Set seed for reproducibility
    seed = 42

    # Load data
    data = pd.read_csv(os.path.join(CDIR, "dataset", "challenge_dataset.csv"))
    x = data.drop("label", axis=1).values
    y = data["label"].values

    # Data preprocessing
    x = preprocessData(x, ndim=3)
    
    # Model selection
    ndim = x.shape[-1]
    models = getModels(ndim, seed)
    for name, model in models.items():

        # Perform stratified k-fold cross-validation
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        accuracies = []
        for train_index, test_index in skf.split(x, y):
            global iterationIdx
            iterationIdx = 0
            x_train_fold, x_test_fold = x[train_index], x[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]
            model.fit(x_train_fold, y_train_fold)
            accuracies.append(model.score(x_test_fold, y_test_fold))

        # Print statistics for model
        meanAccuracy = np.mean(accuracies)
        stdAccuracy = np.std(accuracies)
        logging.info("Overall Accuracy (%s): %0.1f %% (std. %0.2f %%)" % (name, meanAccuracy * 100, stdAccuracy * 100))

        if name == 'qml':
            model.save(os.path.join(CDIR, 'qml.model'))

if __name__ == "__main__":
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)
    train()
    logging.info("All done.")
