from sklearn.datasets import make_blobs

# example dataset
# https://qiskit-community.github.io/qiskit-machine-learning/tutorials/07_pegasos_qsvc.html
features, labels = make_blobs(n_samples=20, n_features=10, centers=2, random_state=3, shuffle=True)

print(features[0])