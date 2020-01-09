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


def plot_averaged_experiments(experiments, measure_key,
                              variable_name='epoch'):
    combined_histories = _calculate_point_wise_average_over_experiments(experiments, measure_key)

    return


def _calculate_point_wise_average_over_experiments(experiment_results, measure_key):
    # While the input consists of experiments containing multiple histories
    # we need the histories containing multiples values at each point.
    # Histories can't hold ore than one value by point but by combining
    # the restructuring with the readout of one specific measure they never need to.
    histories_with_colleted_measures = []
    for experiment_result in enumerate(experiment_results):
        print()

    return

