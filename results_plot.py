import matplotlib.pyplot as plt


def plot_as_matrix(V, title, figsize=(10,10), annotated=False, save_path=None):

    figure = plt.figure(figsize=figsize)
    axes = figure.add_subplot(111)
    axes.set_title(f'{title}')
    caxes = axes.matshow(V, interpolation='nearest')

    #axes.plot([-0.5, 1.5], [4.5, 4.5], color='black', linewidth='3')
    #axes.plot([4.5, 4.5], [-0.5, 4.5], color='black', linewidth='3')

    #cbar = figure.colorbar(caxes, [V.min(), V.max()])
    #cbar.ax.set_xticklabels([str(V.min()), str(V.max())])

    if annotated:
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                text = axes.text(j, i, f'{V[i, j]:.2f}', ha="center", va="center", color="w")

    if save_path is not None:
        plt.savefig(f'{save_path}.png', bbox_inches='tight')

    return figure