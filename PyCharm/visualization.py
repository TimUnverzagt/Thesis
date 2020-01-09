# General modules
import matplotlib.pyplot as plt


def plot_measure_over_n_trainings(histories, history_names, measure_key,
                                  variable_name='epoch'):
    epoch_count = range(1, len(histories[0][measure_key]) + 1)
    for idx, history in enumerate(histories):
        measure = history[measure_key]
        plt.plot(epoch_count, measure)

    plt.legend(history_names)
    plt.xlabel(variable_name)
    plt.ylabel(measure_key)
    plt.show()

    return


def plot_mean_and_var_over_n_trainings(histories, history_names, measure_key,
                                       variable_name='epoch'):

    return


def _calculate_mean_and_var_for_histories(histories):
    return
