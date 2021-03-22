import matplotlib.pyplot as plt


def draw_loss_accuracy_plot(curves: dict) -> None:
    """ The function creates a plot of 4 curves and displays it"""
    colors = "bgrcmyk"
    color_index = 0
    epochs = range(1, len(next(iter(curves.values()))) + 1)

    for label, value in curves.items():
        plt.plot(epochs, value, c=colors[color_index], label=label)
        color_index += 1

    plt.xticks(epochs)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()
