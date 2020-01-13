# General modules
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def plot_measure_over_n_trainings(histories, history_names, measure_key,
                                  variable_name='epoch'):
    epoch_count = range(1, len(histories[0][measure_key]) + 1)

    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(histories))))
    for idx, history in enumerate(histories):
        c = next(color)
        measure = history[measure_key]
        plt.plot(epoch_count, measure, color=c)

    plt.legend(history_names)
    plt.xlabel(variable_name)
    plt.ylabel(measure_key)
    plt.show()

    return


def plot_measure_with_confidence_over_n_trainings(measure_with_confidence, history_names, measure_key,
                                                  variable_name='epoch'):
    # epoch_count = range(1, len(measure_with_confidence[0]['means']))
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(measure_with_confidence))))
    for idx, measure_with_confidence in enumerate(measure_with_confidence):
        c = next(color)
        plot_mean_and_CI(
            measure_with_confidence['means'],
            measure_with_confidence['upper_bounds'],
            measure_with_confidence['lower_bounds'],
            color_mean=c,
            color_shading=c)

    plt.legend(history_names, bbox_to_anchor=(1.05, 1.05))
    plt.xlabel(variable_name)
    plt.ylabel(measure_key)
    plt.savefig("../LaTeX/gfx/Experiments/test.png",
                bbox_inches='tight')
    plt.show()
    return


def plot_averaged_experiments(experiment_results, measure_key,
                              variable_name='epoch'):
    combined_histories = _calculate_point_wise_average_over_experiments(experiment_results, measure_key)

    plot_measure_with_confidence_over_n_trainings(
        measure_with_confidence=combined_histories,
        history_names=experiment_results[0]['network_names'],
        measure_key=measure_key
    )

    return


def plot_average_measure_over_different_pruning_depths(pruning_results, pruning_names, measure_key,
                                                       variable_name='pruning epoch'):
    developments = []
    for pruning_result in pruning_results:
        development_of_average_measure = []
        for history in pruning_result:
            measure_at_epochs = history[measure_key]
            development_of_average_measure.append(
                np.average(measure_at_epochs)
            )
        developments.append(
            development_of_average_measure
        )

    epoch_count = range(1, len(developments[0]) + 1)
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(developments))))
    for development in developments:
        c = next(color)
        plt.plot(epoch_count, development, color=c)

    plt.legend(pruning_names)
    plt.xlabel(variable_name)
    plt.ylabel(measure_key)
    plt.show()
    return


def plot_averaged_early_tickets(experiment_results, measure_key,
                                variable_name='pruning epoch'):
    processed_experiment_results = []
    for experiment_result in experiment_results:
        developments = []
        for pruning_result in experiment_result['pruning_results']:
            development_of_average_measure = []
            for history in pruning_result:
                measure_at_epochs = history[measure_key]
                development_of_average_measure.append(
                    np.average(measure_at_epochs)
                )
            developments.append(
                development_of_average_measure
            )
        processed_experiment_results.append(
            developments
        )

    processed_array = np.array(processed_experiment_results)
    bundled_developments = np.moveaxis(processed_array, 0, -1)
    print("Plotting the average of multiple early ticket searches is not fully implemented or funtioncal.")
    print("Unless the code now produces different results when run multiple times there is no point in using this function tho.")
    return


def _calculate_point_wise_average_over_experiments(experiment_results, measure_key):
    # While the input consists of experiments containing multiple histories
    # we need the histories containing multiples means at each point.
    # Histories can't hold ore than one value by point but by combining
    # the restructuring with the readout of one specific measure they never need to.
    histories_of_bundled_datapoints = []
    # Prepare structure of the correct size
    for i in range(len(experiment_results[0]['network_histories'])):
        histories_of_bundled_datapoints.append([])
    for i in range(len(experiment_results[0]['network_histories'])):
        for j in range(len(experiment_results[0]['network_histories'][0][measure_key])):
            histories_of_bundled_datapoints[i].append([])
    # Bundle datapoints
    for result in experiment_results:
        network_histories = result['network_histories']
        combined_history = []
        for idx, history in enumerate(network_histories):
            datapoints = history[measure_key]
            for j in range(len(datapoints)):
              histories_of_bundled_datapoints[idx][j].append(
                  datapoints[j]
              )
    #  Prepare another structure
    histories_with_values_and_bounds = []
    # Prepare structure of the correct size
    for history in histories_of_bundled_datapoints:
        means = []
        lower_bounds = []
        upper_bounds = []
        total = 0
        nans = 0
        for bundled_datapoint in history:
            bundled_stats = stats.bayes_mvs(bundled_datapoint)
            mean = np.mean(bundled_datapoint)
            means.append(mean)
            if np.isnan(stats.bayes_mvs(bundled_datapoint)[0][1][0]):
                lower_bounds.append(mean)
                upper_bounds.append(mean)
                # nans += 1
            else:
                lower_bounds.append(bundled_stats[0][1][0])
                upper_bounds.append(bundled_stats[0][1][0])
            # total += 1

        # print("Percentage of Nan-values: " + str(nans/total))
        histories_with_values_and_bounds.append(
            {'means': means,
             'lower_bounds': lower_bounds,
             'upper_bounds': upper_bounds}
        )

    return histories_with_values_and_bounds


# credit to Studywolf
# https://studywolf.wordpress.com/2017/11/21/matplotlib-legends-for-mean-and-confidence-interval-plots/
def plot_mean_and_CI(mean, lb, ub, color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(range(len(mean)), ub, lb,
                     color=color_shading, alpha=.5)
    # plot the mean on top
    plt.plot(mean, color=color_mean)
