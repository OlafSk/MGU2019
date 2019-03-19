import numpy as np
import matplotlib.pyplot as plt

def plot_decision_surface(nn, X, n_samples = 50, proba=True, many_classes=False, **kwargs):
    OX = np.linspace(X['x'].min(), X['x'].max(), n_samples)
    OY = np.linspace(X['x'].min(), X['y'].max(), n_samples)
    xx, yy = np.meshgrid(OX, OY)
    flat_x, flat_y = xx.flatten().tolist(), yy.flatten().tolist()
    z = np.zeros_like(flat_x)
    for i in range(len(flat_x)):
        z[i] = nn.forward_pass(np.array([[flat_x[i], flat_y[i]]])) if not many_classes else nn.forward_pass(np.array([[flat_x[i], flat_y[i]]])).argmax(axis=-1)
    if not proba:
        z = z > 0.5
    plt.contourf(xx,yy, z.reshape(xx.shape[0],xx.shape[1]),
    **kwargs)
    return z

def one_hot_encode(Y, k):
    Y2 = np.zeros((Y.shape[0], k))
    Y2[np.arange(Y.shape[0]), Y.reshape(-1, )-1] = 1
    return Y2