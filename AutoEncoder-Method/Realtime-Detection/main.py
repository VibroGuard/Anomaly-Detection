import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

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

    max_MAE = get_max_MAE(model, trainX)

    print("Training done.")

    return model, max_MAE, scaler


def predict(model, max_mae, scaler, xdata):
    seq_size = 30
    # print("Before scaling: ", xdata.shape)
    xdata = scale(xdata, scaler)
    # print("After scaling: ", xdata.shape)
    xdata, ydata = to_sequences(xdata, xdata, seq_size)
    # print("Sequenced: ", xdata.shape)
    mae = get_mae(model, xdata)
    # print("MAE acquired.")
    anomaly_indices = np.asarray(mae > max_mae).nonzero()[0]

    print("Anomaly indices found.")

    return anomaly_indices


if __name__ == "__main__":
    num_samples = 512  # This should match with the number of samples taken by the MCU.
    sampling_frequency = 250
    seq_size = 30

    model_trained = False
    trained_models = None
    anomaly_indices = None

    x_data = [0.0] * num_samples
    y_data = [0.0] * num_samples
    z_data = [0.0] * num_samples

    # fig, axs = plt.subplots(2, 3, figsize=(15, 5))
    fig, axs = plt.subplots(1, 3, figsize=(5, 5))

    port = find_arduino()
    ser = get_serial_port(port, 115200)

    while True:
        if not model_trained:
            print("Starting collecting data...")
            x_data_buffer, y_data_buffer, z_data_buffer = collect_dataset(2000, 10, num_samples, ser)
            print("Data collection done.")
            print(x_data_buffer)
            print(y_data_buffer)
            print(z_data_buffer)

            print("Start training model...")
            with Pool() as pool:
                trained_models = pool.map(model, (x_data_buffer, y_data_buffer, z_data_buffer))

            print("Model training done.")
            model_trained = True

        received_data = str(ser.readline())[2:-5].casefold()
        print(received_data)

        if received_data == "x":
            x_data = fill_buffer(num_samples, ser)
            continue
        elif received_data == "y":
            y_data = fill_buffer(num_samples, ser)
            continue
        elif received_data == "z":
            z_data = fill_buffer(num_samples, ser)

            # Start doing only when data of  all three axis are available.

            print("Getting predictions...")
            with Pool() as pool:
                anomaly_indices = pool.starmap(predict, (
                    (trained_models[0][0], trained_models[0][1], trained_models[0][2], np.array(x_data).reshape(-1, 1)),
                    (trained_models[1][0], trained_models[1][1], trained_models[1][2], np.array(y_data).reshape(-1, 1)),
                    (
                        trained_models[2][0], trained_models[2][1], trained_models[2][2],
                        np.array(z_data).reshape(-1, 1))))
        else:
            continue

        print(x_data)
        print(y_data)
        print(z_data)



        # visualize_data(x_data, y_data, z_data, sampling_frequency, "time", fig, axs)
        visualize_data_time_only(x_data, y_data, z_data, sampling_frequency, fig, axs)
        visualize_anomalies(x_data, y_data, z_data, anomaly_indices[0], anomaly_indices[1], anomaly_indices[2], sampling_frequency, fig, axs)