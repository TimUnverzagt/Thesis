# General modules
import matplotlib.pyplot as plt


def plot_measure_comparision_over_training(history1, history_name1,
                                           history2, history_name2,
                                           measure_name):
    measure1 = history1.history[measure_name]
    measure2 = history2.history[measure_name]

    epoch_count = range(1, len(measure1) + 1)

    plt.plot(epoch_count, measure1, 'r--')
    plt.plot(epoch_count, measure2, 'b-')
    plt.legend([history_name1, history_name2])
    plt.xlabel('epoch')
    plt.ylabel(measure_name)
    plt.show()

    return
