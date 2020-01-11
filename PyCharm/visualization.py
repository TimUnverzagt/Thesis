# General modules
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


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

    plt.legend(history_names)
    plt.xlabel(variable_name)
    plt.ylabel(measure_key)
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
