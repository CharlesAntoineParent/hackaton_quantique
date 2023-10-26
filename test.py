import os
import logging
import numpy as np
import pandas as pd
import qiskit_ibm_runtime

from qiskit_ibm_runtime import Session
from qiskit_ibm_provider import IBMProvider
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold

from train import preprocessData

CDIR = os.path.dirname(os.path.abspath(__file__))

ressources = {
    'simulator': {
        'provider': 'ibm-q/open/main',
        'backend': 'ibmq_qasm_simulator'
    },
    'ibm_quebec': {
        'provider': 'pinq-quebec-hub/iq-quantum/hackathon',
        'backend': 'ibm_quebec'
    },
}

IBMProvider.save_account(token="26957a0c470abaf3caf7b7dd37b6f662a1f168095308eae2ff6c490dd51c1ed5fb875bb39d034e6c02a11384d24d4d61fb64d2f5de5179978599957579b2adca", overwrite=True)

def test():
    # Set seed for reproducibility
    seed = 42

    # Load data
    data = pd.read_csv(os.path.join(CDIR, "dataset", "challenge_dataset.csv"))
    x = data.drop("label", axis=1).values
    y = data["label"].values

    # Data preprocessing
    x = preprocessData(x, ndim=3)

    for ressource in ressources.values():
        logging.info('Testing on %s' % (ressource['backend']))
        service = QiskitRuntimeService(instance=ressource['provider'])
        with Session(service=service, backend=ressource['backend']) as session:
            
            # Load model from file
            estimator = qiskit_ibm_runtime.Estimator(session=session)
            model = NeuralNetworkClassifier.load(os.path.join(CDIR, 'qml.model'))
            model.warm_start = True
            model.neural_network.estimator = estimator

            # Perform stratified k-fold cross-validation
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            accuracies = []
            for train_index, test_index in skf.split(x, y):
                _, x_test_fold = x[train_index], x[test_index]
                _, y_test_fold = y[train_index], y[test_index]
                accuracies.append(model.score(x_test_fold, y_test_fold))

            # Print statistics for model
            meanAccuracy = np.mean(accuracies)
            stdAccuracy = np.std(accuracies)
            logging.info("Overall Accuracy (qml, backend=%s): %0.1f %% (std. %0.2f %%)" % (ressource['backend'], meanAccuracy * 100, stdAccuracy * 100))

if __name__ == "__main__":
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)
    test()
    logging.info("All done.")
