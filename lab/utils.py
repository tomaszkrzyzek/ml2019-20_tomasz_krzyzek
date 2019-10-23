import numpy as np
import utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker, cm

import matplotlib.animation as animation
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.preprocessing import StandardScaler


def get_fn_values(points, fn, X_vals):
    return np.array([fn(points, v) for v in X_vals])


def plot_1d_set(dataset, ax, loss_fns, show_title=False):
    linspace = np.linspace(dataset.min(), dataset.max(), num=200)
    ax.set_xlabel("v")
    ax.set_ylabel("Loss val")
    ax.scatter(dataset, [0] * len(dataset))
    for idx, loss_fn in enumerate(loss_fns):
        y_vals = get_fn_values(dataset, loss_fn, linspace)
        if show_title:
            ax.set_title(loss_fn.__name__)
        ax.plot(linspace, y_vals, label=loss_fn.__name__)

        
def plot_2d_set(dataset, ax, loss_fn):
    dataset_mins = dataset.min(0)
    dataset_maxs = dataset.max(0)
    first_linspace = np.linspace(dataset_mins[0], dataset_maxs[0], num=40)
    second_linspace = np.linspace(dataset_mins[1], dataset_maxs[1], num=40)
    X, Y = np.meshgrid(first_linspace, second_linspace)
    Z = np.empty_like(X)

    for row_idx, first_coord in enumerate(first_linspace):
        for col_idx, second_coord in enumerate(second_linspace):
            Z[row_idx][col_idx] = loss_fn(dataset, np.array([first_coord, second_coord]))
    ax.plot_surface(X, Y, Z)

    ax.scatter(dataset[:, 0], dataset[:, 1], np.zeros((dataset.shape[0],)))

    
def contour_2d_set(dataset, ax, loss_fn, linspaces=None):
    dataset_mins = dataset.min(0)
    dataset_maxs = dataset.max(0)
    if linspaces is None:
        first_linspace = np.linspace(dataset_mins[0], dataset_maxs[0], num=25)
        second_linspace = np.linspace(dataset_mins[1], dataset_maxs[1], num=25)
    else:
        first_linspace, second_linspace = linspaces
    X, Y = np.meshgrid(first_linspace, second_linspace, indexing="xy")
    Z = np.empty_like(X)

    for row_idx, first_coord in enumerate(first_linspace):
        for col_idx, second_coord in enumerate(second_linspace):
            Z[col_idx][row_idx] = loss_fn(dataset, np.array([first_coord, second_coord]))
    
    ax.contour(X, Y, Z, levels=20)
    if linspaces is None:
        ax.scatter(dataset[:, 0], dataset[:, 1])
    else:
        ax.contourf(first_linspace, second_linspace, Z, levels=300, cmap=cm.PuBu_r)
    #    plt.colorbar()
        

def plot_2d_loss_fn(loss_fn, title, dataset):
    fig = plt.figure(figsize=(10, 4))
    fig.suptitle(title)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    plot_2d_set(dataset, ax, loss_fn)
    ax = fig.add_subplot(1, 2, 2)
    contour_2d_set(dataset, ax, loss_fn)
    plt.show(fig)
    plt.close(fig)


def plot_minimums(dataset, loss_fns, loss_fns_mins, title):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title)

    min_vals = []
    for (loss_fn, loss_fn_min, ax) in zip(loss_fns, loss_fns_mins, axes):
        min_val = loss_fn_min(dataset)
        min_vals += [min_val]
        ax.scatter(
            min_val,
            loss_fn(dataset, min_val),
            color="black"
        )
        plot_1d_set(dataset, ax, [loss_fn], show_title=True)

    plt.show(fig)
    plt.close(fig)
    print(
        "ME minimum: {:.2f} MSE minimum: {:.2f} Max Error minimum: {:.2f}".format(
            *min_vals)
    )


def plot_gradient_steps_1d(ax, dataset, gradient_descent_fn, grad_fn, loss_fn, num_steps=100, learning_rate=1e-1):
    final_v, final_grad, all_v = gradient_descent_fn(
        grad_fn, dataset, num_steps=num_steps, learning_rate=learning_rate)
    plot_1d_set(dataset, ax, [loss_fn])
    y_vals = get_fn_values(dataset, loss_fn, all_v)
    ax.scatter(all_v, y_vals, c=np.arange(len(all_v)), cmap=plt.cm.viridis)
    return final_v


def plot_gradient_steps_2d(ax, dataset, gradient_descent_fn, grad_fn, loss_fn, num_steps=100, learning_rate=1e-2, linspaces=None):
    final_v, final_grad, all_v = gradient_descent_fn(
        grad_fn, dataset, num_steps=num_steps, learning_rate=learning_rate)
    contour_2d_set(dataset, ax, loss_fn, linspaces)
    ax.scatter(all_v[:, 0], all_v[:, 1], c=np.arange(len(all_v)), cmap=plt.cm.viridis)

    print("Final grad value for {}: {}".format(loss_fn.__name__, final_grad))
    return final_v


def visualize_normal_dist(X, loc, scale):
    peak = 1 / np.sqrt(2 * np.pi * (scale ** 2))
    plt.hist(X, bins=50, density=True)
    plt.plot([loc - scale, loc - scale], [0, peak], color="r", label="1 sigma")
    plt.plot([loc + scale, loc + scale], [0, peak], color="r")

    plt.plot([loc - 2 * scale, loc - 2 * scale], [0, peak], color="b", label="2 sigma")
    plt.plot([loc + 2 * scale, loc + 2 * scale], [0, peak], color="b")

    plt.plot([loc - 3 * scale, loc - 3 * scale], [0, peak], color="g", label="3 sigma")
    plt.plot([loc + 3 * scale, loc + 3 * scale], [0, peak], color="g")
    plt.legend()

    
def scatter_with_whiten(X, whiten, name, standarize=False):
    plt.title(name)
    plt.scatter(X[:, 0], X[:, 1], label="Before whitening")
    white_X = whiten(X)
    plt.axis("equal")
    plt.scatter(white_X[:, 0], white_X[:, 1], label="After whitening")
    
    
    if standarize:
        X_standarized = (X - X.mean(axis=0)) / X.std(axis=0)
        plt.scatter(X_standarized[:, 0], X_standarized[:, 1], label="Standarized")
        
    plt.legend()
    plt.show()

    
def generate_and_fit(mu, sigma, samples_num, grad_fn):
    dataset = np.random.normal(mu, sigma, size=(samples_num, 1))
    (final_mu, final_sigma), _, _ = gradient_descent(
        grad_fn,
        dataset,
        learning_rate=5e-2 / dataset.shape[0],
        num_steps=20000
    )

    print("Final mu: {:.2f}. Final sigma: {:.2f}".format(final_mu, final_sigma))
    print("True mu: {:.2f}. True sigma: {:.2f}".format(mu, sigma))

    plt.scatter(dataset, np.zeros_like(dataset), color="red", s=3.)
    X = np.linspace(-5, 5, num=1000)
    grad_Y = norm.pdf(X, loc=final_mu, scale=final_sigma)
    plt.plot(X, grad_Y, label="Found distribution")
    true_Y = norm.pdf(X, loc=mu, scale=sigma)
    plt.plot(X, true_Y, label="True distribution")
    plt.legend()
    plt.show()


def plot_clustering(X, y, k=3):
    
    assert X.shape[0] == y.shape[0]

    f = plt.figure(figsize=(8, 8))
    ax = f.add_subplot(111)
    ax.axis('equal')
    
    for i in range(k):
        ax.scatter(X[y == i, 0], X[y == i, 1])
        
        
def animate_clustering(X, ys):

    def update_colors(i, ys, scat):
        scat.set_array(ys[i]) 
        return scat,

    n_frames = len(ys)

    colors = ys[0]

    fig = plt.figure(figsize=(8, 8))
    scat = plt.scatter(X[:, 0], X[:, 1], c=colors)

    ani = animation.FuncAnimation(fig, update_colors, frames=range(n_frames),
                                  fargs=(ys, scat))
    return ani


def plot_cluster_comparison(datasets, results):
    
    assert len(results) == len(datasets),  "`results` list length does not match the dataset length!"

    n_rows = len(results)
    n_cols = len(results[0])

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4 * n_rows, 4 * n_cols))

    for ax, col in zip(axes[0], ['K-Means', 'DBSCAN', 'Agglomerative', 'GMM']):
        ax.set_title(col, size=24)

    for row, X, y_row in zip(axes, datasets, results):
        for ax, y in zip(row, y_row):

            ax.scatter(X[:,0], X[:,1], c=y.astype(np.int64))
            

def get_clustering_data():
    
    def standarize(X):
        return StandardScaler().fit_transform(X)

    n_samples = 1500
    noisy_circles = make_circles(n_samples=n_samples, factor=.5, noise=.05)

    noisy_moons = make_moons(n_samples=n_samples, noise=.05)
    # Anisotropicly distributed data
    random_state = 170
    X, y = make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = make_blobs(n_samples=n_samples,
                                 cluster_std=[1.0, 2.5, 0.5],
                                 random_state=random_state)

    datasets = [noisy_circles[0],
          noisy_moons[0],
          X_aniso,
          varied[0]]

    datasets = [standarize(X) for X in datasets]

    return datasets