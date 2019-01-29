
import numpy as np
import sklearn
from matplotlib import pyplot as plt
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import GridSearchCV, KFold
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
import time
from functools import wraps
from sklearn import preprocessing
from prettytable import PrettyTable
import math

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def watcher(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f" ===> took {end-start} seconds")
        return result
    return wrapper


def make_meshgrid(x, y, h=0.04):

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
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))

    return xx, yy


def plot_contourf(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

        Parameters
        ----------
        ax: matplotlib axes object
        clf: a classifier
        xx: meshgrid ndarray
        yy: meshgrid ndarray
        params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    out = plt.contourf(xx, yy, Z, **params)
    return out


def plot_boundaries(X, Y, x_t, y_t, model, title, labels, ax):

    # leggi meshgrid
    # counturf
    # scatterplot

    markers = ('s', 'D', 'o')
    colors = ('#7EBC89', '#AF1B3F', '#DF9B6D')
    label = np.unique(labels)

    cmap = ListedColormap(colors[:len(np.unique(Y)) + 1])

    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(1, 1, 1)

    xx, yy = make_meshgrid(X[:, 0], X[:, 1])
    plot_contourf(ax, model, xx, yy, cmap=cmap, alpha=0.8)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    # plot class samples
    i = 0
    for idx, cl in enumerate(np.unique(Y)):
        plt.scatter(x=x_t[y_t == cl, 0], y=x_t[y_t == cl, 1],
                    alpha=0.8, c=colors[cl], label=label[i],
                       marker=markers[cl], edgecolor='k')
        i += 1

    # plt.xlabel("Sepal Length")
    # plt.ylabel("Sepal Width")
    # plt.title(title)
    # plt.legend()
    # fig.show()


def plot_single_boundary(X, Y, model):

    ax = plt.figure(figsize=(10, 10))
    plt.subplot(1, 1, 1)
    plot_decision_regions(X=X,
                          y=Y,
                          clf=model,
                          legend=2)

    plt.xlabel('Sepal Length', size=14)
    plt.ylabel('Sepal width', size=14)
    plt.title('Decision boundaries', size=16)
    plt.show()


def plot_colormap(scores, C_range, gamma_range):

    fig = plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=cm.get_cmap('summer'),
               norm=MidpointNormalize(vmin=0.2, midpoint=0.6))
    plt.xlabel('Gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy')
    plt.show()
    fig.savefig("colormap_grid" + str(int(np.ceil(time.time()))) + ".png")


def print_table(scores, c_values, gamma_values, title):

    t = PrettyTable()
    scores = scores.T
    t.field_names = [title] + ["{:.1e}".format(c) for c in c_values]
    max_overall = np.max(scores)
    min_overall = np.min(scores)

    for index, row in enumerate(scores):

        r = []
        if type(gamma_values[index]) == str:
            r.append(gamma_values[index])
        else:
            r.append("{:.1e}".format(gamma_values[index]))

        for value in row:
            if value == max_overall:
                r.append("\033[94m {:.2f}%\033[00m".format(value * 100))
            elif value == min_overall:
                r.append("\033[91m {:.2f}%\033[00m".format(value * 100))
            else:
                r.append("{:.2f}%".format(value * 100))

        t.add_row(r)

    print(t)


@watcher
def grid_search(X, y, X_train, y_train, X_test, y_test, X_val, y_val):

    C_range = np.logspace(-3, 3, 7)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)

    # Execute grid search on the combinations of C and gamma

    scores = np.empty((len(gamma_range), len(C_range)))
    for i, g in enumerate(gamma_range):
        for j, c in enumerate(C_range):

            rbf = svm.SVC(kernel='rbf', C=c, gamma=g).fit(X_train, y_train)
            scores[i][j] = rbf.score(X_val, y_val)
            if scores[i][j] >= np.amax(scores):
                best_rbf = rbf
                best_c = c
                best_gamma = g

    '''
    grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid=param_grid, cv=2)
    grid.fit(X_train, y_train)
    '''
    best_achieved = best_rbf.score(X_test, y_test)
    print("Best C = \033[94m {}\033[00m".format(best_c) +
          "\nBest Gamma = \033[94m {}\033[00m".format(best_gamma))

    '''
    max = np.max(scores)
    flat_index = np.argmax(scores)
    i = int(flat_index / scores.shape[0])
    j = flat_index % scores.shape[1]
    best = dict(best_c=C_range[i], best_gamma=gamma_range[j])
    print(best)
    '''
    plot_colormap(scores.T, C_range, gamma_range)
    print("Grid search best achieved: \033[94m {:.2f}%\033[00m".format(best_achieved * 100))
    print_table(scores, gamma_range, C_range, title="Gamma/C")



    # si nota che il classifier si avvicina molto ad un linear -> il linear è un rbf degenerato
    # plot_single_boundary(X, y, grid)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    plot_boundaries(X, y, X_train, y_train, linear, None, Y, ax)
    ax.set_title('Grid search rbf')
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    fig.show()
    fig.savefig("cv_best_rbf.png")


def best_linear(linear_models, X_test, y_test, linear_scores, c_values):

    best_linear = max(linear_scores)
    best_linear_index = linear_scores.index(best_linear)
    m_best = linear_models[best_linear_index]
    best_linear_score = m_best.score(X_test, y_test)
    print("Best linear score achieved: " + "\033[94m {:.2f}%\033[00m".format(best_linear_score * 100))


def plot_accuracy(linear_scores, rbf_scores, c_values):

    y_pos = np.arange(len(linear_scores))
    bar_width = 0.35
    values = ['Linear', 'Rbf']

    linear_scores = [i * 100 for i in linear_scores]
    performance = linear_scores

    rbf_scores = [i * 100 for i in rbf_scores]

    plt.bar(y_pos, performance, bar_width,
            align='center',
            color='r',
            alpha=0.5)

    plt.bar(y_pos + bar_width, rbf_scores, bar_width,
            align='center',
            color='b',
            alpha=0.5)

    plt.xticks(y_pos + bar_width/2, c_values)
    axes = plt.gca()
    axes.set_ylim([0, 100])
    plt.legend(values, loc=2)
    plt.ylabel('Accuracy rate')
    plt.xlabel('Values for C')
    plt.title('Accuracy of model')
    plt.savefig("accuracy.png")
    plt.show()



def plot_boundaries_lib(X, Y, models, titles):

    ax = plt.figure(figsize=(30, 10))
    plt.subplot(1, 3, 1)
    plot_decision_regions(X=X,
                      y=Y,
                      clf=models['linear'],
                      legend=2)

    plt.xlabel('Sepal Length', size=14)
    plt.ylabel('Sepal width', size=14)
    plt.title(titles['linear'], size=16)

    plt.subplot(1, 3, 2)
    plot_decision_regions(X=X,
                      y=Y,
                      clf=models['rbf'],
                      legend=2)

    plt.xlabel('Sepal Length', size=14)
    plt.ylabel('Sepal width', size=14)
    plt.title(titles['rbf'], size=16)
    plt.show()


@watcher
def nfold_cv(X, Y, n_splits):

    C_range = np.logspace(-3, 3, 7)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)

    kf = KFold(n_splits=n_splits, shuffle=True)
    kf.get_n_splits(X)

    scores = np.empty((len(gamma_range), len(C_range), n_splits))
    for i, gamma in enumerate(gamma_range):
        for j, C in enumerate(C_range):
            k = 0
            for X_train_index, X_test_index in kf.split(X):

                X_train = X[X_train_index]
                X_test = X[X_test_index]
                y_train = Y[X_train_index]
                y_test = Y[X_test_index]
                model = svm.SVC(kernel='rbf', C=C, gamma=gamma)
                model.fit(X_train, y_train)
                scores[i, j, k] = model.score(X_test, y_test)
                k += 1

    scores_avg = np.mean(scores, axis=2)
    max = np.max(scores_avg)
    print("Manual rbf k-fold cross validation best achieved: \033[94m {:.2f}%\033[00m".format(max*100))
    flat_index = np.argmax(scores_avg)
    i = int(flat_index / scores_avg.shape[0])
    j = flat_index % scores_avg.shape[1]
    print("Best C = \033[94m {}\033[00m".format(C_range[j]) +
          "\nBest Gamma = \033[94m {}\033[00m".format(gamma_range[i]))
    plot_colormap(scores_avg.T, C_range, gamma_range)
    print_table(scores_avg, gamma_range, c_values, "C/Gamma CV")


def menu(text):
    pick = input(text)
    if (pick != str(0)) and (pick != str(1)):
        print("Invalid value")
        exit()
    return int(pick)


if __name__ == '__main__':

    iris = datasets.load_iris()
    X = iris.data[:, :2]
    Y = iris.target

    # we don't really need to scale, because iris is already pretty balanced
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)

    choice = menu("Do you want to print all boundaries? (0/1)")

    X = preprocessing.scale(X, with_mean=True, with_std=True)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.5, shuffle=True)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, train_size=0.4, shuffle=True)

    titles = {"linear": "Linear SVM Decision Region Boundary",
              "rbf": "RBF SVM Decision Region Boundary"}
    c_values = np.logspace(-3, 4, 8)

    linear_scores = []
    rbf_scores = []

    v = len(c_values) * 2
    rows = columns = math.ceil(math.sqrt(v))
    rows = rows + 1

    if choice:

        fig = plt.figure(figsize=(rows * 10, columns * 10))
        i = 1

    linear_models = []
    for C in c_values:
        linear = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
        rbf = svm.SVC(kernel='rbf', C=C, gamma='scale').fit(X_train, y_train)

        if choice:

            ax = fig.add_subplot(rows, columns, i)
            plot_boundaries(X, Y, X_train, y_train, linear, titles, Y, ax)
            ax = fig.add_subplot(rows, columns, i + columns - (len(c_values) % columns) + len(c_values))
            plot_boundaries(X, Y, X_train, y_train, rbf, titles, Y, ax)
            i += 1

        s = linear.score(X_val, y_val)
        linear_scores.append(s)
        s = rbf.score(X_val, y_val)
        rbf_scores.append(s)

        linear_models.append(linear)

    names = ["Linear", "Rbf"]
    title = "C values"
    print_table(np.reshape(linear_scores + rbf_scores, (len(linear_scores), 2)), c_values, names, title)

    if choice:
        fig.show()
        fig.savefig("boundaries.png")

    plot_accuracy(linear_scores, rbf_scores, c_values)
    best_linear(linear_models, X_test, y_test, linear_scores, c_values)
    grid_search(X, Y, X_train, y_train, X_test, y_test, X_val, y_val)
    nfold_cv(X, Y, 10)
