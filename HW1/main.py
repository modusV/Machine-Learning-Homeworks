
from PIL import Image
import matplotlib.pyplot as pyplot
import os
import time
import numpy
from functools import wraps
from sklearn import preprocessing
from sklearn.decomposition import PCA
import sklearn.model_selection
from sklearn import naive_bayes as nb
from matplotlib.colors import ListedColormap
import plotly.plotly as py
import plotly.graph_objs as go


def classic_pca(X, size, nComp=2):
    # Cool and thrilling, but occupies too much RAM....dho

    # mean for rows
    M = numpy.mean(X)

    # centered matrix
    centered = X - M

    # calculate standard dev
    std = numpy.std(centered)

    # scale matrix, divide for standard dev
    scaled = centered / std

    # perchè ho sulle righe i vari samples e selle slides è il contrario
    sigma = numpy.dot(scaled.T, scaled)

    # eigenvalues and eigenvectors
    e, ev = numpy.linalg.eig(sigma)

    # sort eigenvalues and eigenvectors
    idx = e.argsort()[::-1]
    eigenValues = e[idx]
    eigenVectors = ev[:, idx]

    # faccio il transform
    # matrice proiettata nSamples * nComp

    eigenVectorLimit = eigenVectors[:nComp]
    projected = numpy.dot(scaled, eigenVectorLimit)

    # riproietto nelle dimensioni originali
    reversed = numpy.dot(projected, eigenVectorLimit.T)
    reversed = reversed * std + M

    return


def total_variance(X):
    row_variance = 0
    tot_var = 0
    for i in X:
        for j in i:
            row_variance += pow(j, 2)
        row_variance = row_variance/len(i)
        tot_var += row_variance
        row_variance = 0
    return tot_var


def partial_variance(X):
    var = []
    comp_variance = 0

    for i in X:
        for j in i:
            comp_variance += pow(j, 2)
        comp_variance = comp_variance/len(i)
        var.append(comp_variance)
        comp_variance = 0
    return var


def plot_images(images_data, imgn, size, titles):

    fig = pyplot.figure(figsize=(30, 20))
    i = 1
    for im, tit in zip(images_data, titles):
        im = numpy.reshape(im[imgn]/255, size)
        ax = fig.add_subplot(2, 3, i)
        i += 1
        ax.imshow(im)
        ax.axis("off")
        ax.set_title(tit)
    fig.show()
    fig.savefig("decomposed.jpg")


def plot_scatter(labels, colors, X):

    fig = pyplot.figure(figsize=(30, 10))
    ax = fig.add_subplot(1, 3, 1)

    ax.scatter(X[:, 0], X[:, 1], c=colors)
    ax.set_xlabel("Component 0")
    ax.set_ylabel("Component 1")

    ax = fig.add_subplot(1, 3, 2)

    ax.scatter(X[:, 2], X[:, 3], c=colors)
    ax.set_xlabel("Component 3")
    ax.set_ylabel("Component 4")


    ax = fig.add_subplot(1, 3, 3)

    ax.scatter(X[:, 10], X[:, 11], c=colors)
    ax.set_xlabel("Component 11")
    ax.set_ylabel("Component 12")

    fig.show()
    fig.savefig("scatter.jpg")


def plot_cum_var(var, n):

    fig = pyplot.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_ylabel('% Variance Explained')
    ax.set_xlabel('# of Features')
    ax.set_title('PCA Analysis')
    ax.set_ylim(30, 100)
    ax.plot(var[:n])
    fig.show()
    fig.savefig("cumulativeV.jpg")


def make_meshgrid(x, y, h=1):

    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """

    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h),
                            numpy.arange(y_min, y_max, h))

    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

        Parameters
        ----------
        ax: matplotlib axes object
        clf: a classifier
        xx: meshgrid ndarray
        yy: meshgrid ndarray
        params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(numpy.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    out = pyplot.contourf(xx, yy, Z, **params)
    return out


def plot_boundaries(X, Y, x_t, y_t, model, title, labels):

    # leggi meshgrid
    # counturf
    # scatterplot

    markers = ('s', 'D', 'o', '^')
    colors = ('#7EBC89', '#AF1B3F', '#473144', '#DF9B6D')
    label = numpy.unique(labels)

    cmap = ListedColormap(colors[:len(numpy.unique(Y)) + 1])

    fig = pyplot.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)

    xx, yy = make_meshgrid(X[:, 0], X[:, 1])
    plot_contours(ax, model, xx, yy, cmap=cmap, alpha=0.8)

    pyplot.xlim(xx.min(), xx.max())
    pyplot.ylim(yy.min(), yy.max())

    # plot class samples
    i = 0
    for idx, cl in enumerate(numpy.unique(Y)):
        pyplot.scatter(x=x_t[y_t == cl, 0], y=x_t[y_t == cl, 1],
                    alpha=0.8, c=colors[cl], label=label[i],
                       marker=markers[cl], edgecolor='k')
        i += 1

    pyplot.xlabel("Less important component")
    pyplot.ylabel("Most important component")
    pyplot.title(title)
    pyplot.legend()
    fig.show()


def watcher(func):
    """
    Decorator for dumpers.
    Shows how much time it
    takes to create/retrieve
    the blob.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f" ===> took {end-start} seconds")
        return result
    return wrapper


def menu():
    choice = int(input(f"Please insert:"
                       f"\n1- Display Image"
                       f"\n2- Plot scatterplots"
                       f"\n3- Plot cumulative variance"
                       f"\n4- Exit"))
    return choice


@watcher
def improved_pca(X, size, labels, colors):

    mean = numpy.mean(X)
    std = numpy.std(X)
    x_scaled = preprocessing.scale(X, with_mean=True, with_std=True)

    pcaMax = PCA(.999, svd_solver='full')
    pcaMax.fit(x_scaled)

    variance = pcaMax.explained_variance_
    var = numpy.cumsum(numpy.round(pcaMax.explained_variance_ratio_, decimals=3) * 100)

    tot_comp = pcaMax.components_
    comp2 = pcaMax.components_[:2]
    comp6 = pcaMax.components_[:6]
    comp60 = pcaMax.components_[:60]
    comp_last6 = pcaMax.components_[-6:]

    pcaMax.components_ = comp_last6
    proj_last6 = pcaMax.transform(x_scaled)
    inverse_last6 = pcaMax.inverse_transform(proj_last6)

    pcaMax.components_ = comp2
    fitted2 = pcaMax.transform(x_scaled)
    inverse2 = pcaMax.inverse_transform(fitted2)

    pcaMax.components_ = comp6
    fitted6 = pcaMax.transform(x_scaled)
    inverse6 = pcaMax.inverse_transform(fitted6)

    pcaMax.components_ = comp60
    fitted60 = pcaMax.transform(x_scaled)
    inverse60 = pcaMax.inverse_transform(fitted60)

    pcaMax.components_ = tot_comp
    fittedComp = pcaMax.transform(x_scaled)
    inverseComp = pcaMax.inverse_transform(fittedComp)

    images = []
    titles = []

    images.append(inverse_last6)
    titles.append("Last 6 components")
    images.append(inverse2)
    titles.append("First two components")
    images.append(inverse6)
    titles.append("First six components")
    images.append(inverse60)
    titles.append("First 60 components")
    images.append(inverseComp)
    titles.append("All available components")

    for im in images:
        im *= std
        im += mean

    images.append(X)
    titles.append("Original image")

    choice = menu()

    while choice != 4:

        if choice == 1:
            print(f"Insert image number (0-{len(X)})")
            imn = int(input())
            plot_images(images, imn, size, titles)

        elif choice == 2:
            plot_scatter(labels, colors, fitted60)

        elif choice == 3:
            print(f"Insert requested percentage number (0-100)")
            perc = int(input())
            plot_cum_var(var, perc)
        else:
            print("Invalid value")

        choice = menu()


def generate_pca():

    images_data = []
    colors_dict = {"dog": "red", "guitar": "orange", "house": "green", "person": "blue"}
    color_labels = []
    root_path = "PACS_homework/"
    new_path = ""
    name = ""
    labels = []
    folders = []

    for name in os.listdir(root_path):
        if os.path.isdir(os.path.join(os.path.abspath(root_path), name)):
            folders.append(name)

    folders = sorted(folders)

    for dir in folders:
        new_path = root_path + dir + "/"
        for r, d, f in os.walk(new_path, topdown=False):
            for name in f:
                x = numpy.asarray(Image.open(new_path + name))
                x = x.ravel()
                labels.append(dir)
                images_data.append(x)
                color_labels.append(colors_dict[dir])

    im = numpy.asarray(Image.open(new_path + name))  # open one image to get size
    m, n = im.shape[0:2]  # get the size of the images
    size = (m, n, 3)
    choice = int(input("Select one option: "
                       "\n1- PCA decomposition"
                       "\n2- Naive Bayes Classifier"
                       "\n3- Exit"))

    if choice == 1:
        improved_pca(images_data, size, labels, color_labels)
    elif choice == 2:
        naive_bayes(images_data, size, labels, color_labels)
    elif choice == 3:
        print("Exiting, bye")
    else:
        print("Invalid value, exiting")
    return



@watcher
def naive_bayes(images, size, labels, color_labels):

    mean = numpy.mean(images)
    std = numpy.std(images)

    scaled = preprocessing.scale(images, with_mean=True, with_std=True)

    pca = PCA(4)
    pca.fit(scaled)

    one_two = pca.components_[:2]
    three_four = pca.components_[2:4]

    lab = numpy.copy(labels)

    for i, elem in enumerate(labels):
        if elem == 'dog':
            labels[i] = 0
        if elem == 'house':
            labels[i] = 1
        if elem == 'guitar':
            labels[i] = 2
        if elem == 'person':
            labels[i] = 3

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(scaled, labels, train_size=0.7, random_state=1)

    model = nb.GaussianNB()
    model.fit(X_train, y_train)
    score_original = model.score(X_test, y_test)
    print(f"Score with without pca decomposition is: {score_original}")

    pca.components_ = one_two
    X_train_transf = pca.transform(X_train)
    X_test_transf = pca.transform(X_test)
    model.fit(X_train_transf, y_train)
    score_first = model.score(X_test_transf, y_test)
    print(f"Score with first two pcs is: {score_first}")
    title = "Decision boundaries for NB with first two pca components"

    plot_boundaries(X_train_transf, y_train, X_test_transf, y_test, model, title, lab)

    pca.components_ = three_four
    X_train_transf = pca.transform(X_train)
    X_test_transf = pca.transform(X_test)
    model.fit(X_train_transf, y_train)
    score_second = model.score(X_test_transf, y_test)
    print(f"Score with third and fourth pcs is: {score_second}")
    title = "Decision boundaries for NB with 3rd and 4th pca components"


    plot_boundaries(X_train_transf, y_train, X_test_transf, y_test, model, title, lab)
    return


if __name__ == '__main__':
    generate_pca()
