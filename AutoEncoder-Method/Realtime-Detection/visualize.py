import numpy as np
import matplotlib.pyplot as plt


def fft_data(data):
    """
    This function takes the data and returns the fft of the data
    :param data:
    :return: fft of the data (x+iy) and absolute value of the fft (magnitude of the fft)
    """
    fft_result = np.fft.fft(data)
    return fft_result.tolist(), np.abs(fft_result).tolist()


def visualize_data(x, y, z, sps, plot_type, fig, axs):
    """
    This function visualizes the data in x, y, z directions with respect to time
    :param x:
    :param y:
    :param z:
    :param sps samples per second:
    :return: plots the data
    """

    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    names = ["x - axis", "y - axis", "z - axis"]

    if plot_type == "time":
        for i, ax in enumerate(axs[0]):
            ax.clear()
            data = [x, y, z][i]
            time = len(data) / sps  # time = number of samples / samples per second
            horizontal_axis = np.linspace(0, time, len(data))
            ax.plot(horizontal_axis, data)

            ax.set_title(names[i])

            ax.set_xlabel("Time (s)")
            ax.set_ylabel('Magnitude')
    elif plot_type == "frequency":
        for i, ax in enumerate(axs[1]):
            ax.clear()
            data = [x, y, z][i]
            horizontal_axis = np.linspace(int(-sps / 2), int(sps / 2), len(data))
            ax.stem(horizontal_axis, data[int(np.ceil(len(data) / 2)) + 1:] + data[:int(np.ceil(len(data) / 2)) + 1])

            ax.set_title(names[i])

            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel('Magnitude')

    plt.tight_layout()

    plt.pause(0.01)
    fig.canvas.flush_events()

    # plt.show()