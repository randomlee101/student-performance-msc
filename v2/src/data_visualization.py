import matplotlib.pyplot as plt


def plot_training_history(history):
    """
    Plots training vs validation history for accuracy, loss, precision, and recall.

    Parameters:
        history (History object): The History object returned by model.fit().
    """
    metrics = ['accuracy', 'loss', 'precision', 'recall']
    plt.figure(figsize=(14, 10))

    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 1)
        train_metric = history.history.get(metric)
        val_metric = history.history.get(f'val_{metric}')

        if train_metric is not None and val_metric is not None:
            plt.plot(train_metric, label=f'Training {metric}', marker='o')
            plt.plot(val_metric, label=f'Validation {metric}', marker='x')
            plt.title(f'{metric.capitalize()} Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, f'{metric} not found in history.',
                     horizontalalignment='center', verticalalignment='center',
                     transform=plt.gca().transAxes)
            plt.title(f'{metric.capitalize()} Not Available')

    plt.tight_layout()
    plt.show()
