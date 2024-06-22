import numpy as np
import json
import uuid
from tqdm import tqdm

from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor, VQR
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit_aer import AerSimulator
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN

from skimage.transform import resize



from utils import plot_grids

class VQCSolver:

    def __init__(self, inps, outp, grid_size=4):
        self.inps = inps
        self.outp = outp
        self.grid_size = grid_size
        self.num_qubits = int(np.ceil(np.log2(grid_size * grid_size)))  # Number of qubits equals grid size squared\
        self.max_iter = 100
        
    def resize_grid(self, grid):
        resized_grid = resize(grid, (self.grid_size, self.grid_size), order=1, preserve_range=True, anti_aliasing=False)
        return resized_grid.astype(grid.dtype)
    
    def encode_grid(self, grid):
        flat_grid = grid.flatten()
        return flat_grid

    def preprocess(self, inps, outp):
        # convert to numpy
        inps = np.array([self.encode_grid(self.resize_grid(np.array(inp))) for inp in inps])
        outp = np.array([self.encode_grid(self.resize_grid(np.array(out))) for out in outp])
        return inps, outp

    def solve(self, grid):
        X_train, y_train = self.preprocess(self.inps, self.outp)
        
        y_min, y_max = y_train.min(), y_train.max()
        y_train_normalized = 2 * (y_train - y_min) / (y_max - y_min) - 1
        
        # Initialize a list to store the regressors
        regressors = []
        
        for i in range(16):
            # construct feature map with 16 parameters
            params_x = [Parameter(f"x_{j}") for j in range(self.grid_size**2)]
            feature_map = QuantumCircuit(10, name=f"fm_{i}")
            for j in range(self.grid_size**2):
                feature_map.ry(params_x[j], 0)

            # construct ansatz with 16 parameters
            params_y = [Parameter(f"y_{j}") for j in range(self.grid_size**2)]
            ansatz = QuantumCircuit(10, name=f"vf_{i}")
            for j in range(self.grid_size**2):
                ansatz.ry(params_y[j], 0)

            # qc = QNNCircuit(feature_map=feature_map, ansatz=ansatz)
            # regression_estimator_qnn = EstimatorQNN(circuit=qc)
            
            vqr = VQR(
                feature_map=feature_map,
                ansatz=ansatz,
                optimizer=L_BFGS_B(maxiter=5)
            )
            print(f"Training regressor {i+1}/16")
            vqr.fit(X_train, y_train[:, i])
            regressors.append(vqr)
        
        # Predict on the input grid using all regressors
        # Predict on the input grid using all regressors
        grid_preprocessed = self.preprocess([grid], [grid])[0]
        predictions_normalized = [regressor.predict(grid_preprocessed) for regressor in regressors]
        
        # Denormalize predictions
        predictions = np.array([((pred + 1) / 2 * (y_max - y_min) + y_min).item() for pred in predictions_normalized]).reshape(self.grid_size, self.grid_size).tolist()
        
        return predictions

if __name__ == "__main__":
    base_path = '../arc-prize-2024'
    testing_data = json.load(open(f'{base_path}/arc-agi_test_challenges.json', 'r'))
    for key, task in testing_data.items():
        train_data = task['train']
        test_data = task['test'][0]['input']
        inps, outs = [], []
        for pair in train_data:
            inps.append(pair['input'])
            outs.append(pair['output'])
        plot_grids(inps+outs, ['1' for _ in inps]+['2' for _ in outs]).savefig(f'pair_{key}.png')
        # train
        print(len(inps[0]))
        solver = VQCSolver(inps, outs)
        # solve
        pred = solver.solve(test_data)
        print(pred)
        plot_grids([test_data, pred], ['1', '2']).savefig(f'pred_{key}.png')
        # break