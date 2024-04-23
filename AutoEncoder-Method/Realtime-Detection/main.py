from multiprocessing import Pool

from data_collect import *
from visualize import *
from ml_model import *


def model(data_buffer):
    seq_size = 30

    # print("Dataset received.")
    Xtrain, Xtest = split_dataset(data_buffer)
    # print("Dataset splitted.")
    scaler = get_fitted_scalar(Xtrain)
    Xtrain = scale(Xtrain, scaler)
    # print("Dataset scaled.")
    trainX, trainY = to_sequences(Xtrain, Xtrain, seq_size)
    # print("Splited to sequences.")
    model = get_model(trainX)
    # print("Model acquired.")

    model.fit(trainX, trainY, epochs=10, batch_size=32, validation_split=0.1, verbose=1)
    print("Fitting done.")

    return model


if __name__ == "__main__":
    num_samples = 512  # This should match with the number of samples taken by the MCU.
    sampling_frequency = 250
    seq_size = 30

    model_trained = True
    trained_models = None

    x_data = [0.0] * num_samples
    y_data = [0.0] * num_samples
    z_data = [0.0] * num_samples

    fig, axs = plt.subplots(2, 3, figsize=(15, 5))

    port = find_arduino()
    ser = get_serial_port(port, 115200)

    while True:
        if not model_trained:
            x_data_buffer, y_data_buffer, z_data_buffer = collect_dataset(2000, 10, num_samples, ser)
            print(x_data_buffer)
            print(y_data_buffer)
            print(z_data_buffer)

            with Pool() as pool:
                trained_models = pool.map(model, (x_data_buffer, y_data_buffer, z_data_buffer))

            model_trained = True

        received_data = str(ser.readline())[2:-5].casefold()
        print(received_data)

        if received_data == "x":
            x_data = fill_buffer(num_samples, ser)
        elif received_data == "y":
            y_data = fill_buffer( num_samples, ser)
        elif received_data == "z":
            z_data = fill_buffer(num_samples, ser)

        # x_data, y_data, z_data = get_new_dataset(ser, num_samples)
        print(x_data)
        print(y_data)
        print(z_data)

        visualize_data(x_data, y_data, z_data, sampling_frequency, "time", fig, axs)
