from data_collect import *
from visualize import *
from ml_model import *

num_samples = 512  # This should match with the number of samples taken by the MCU.
sampling_frequency = 250

model_trained = False

x_data = [0.0] * num_samples
y_data = [0.0] * num_samples
z_data = [0.0] * num_samples

fig, axs = plt.subplots(2, 3, figsize=(15, 5))

port = find_arduino()
ser = get_serial_port(port, 115200)

while True:
    if not model_trained:
        x_data_buffer, y_data_buffer, z_data_buffer = collect_dataset(2000, 60, num_samples, ser)
        print(x_data_buffer)
        print(y_data_buffer)
        print(z_data_buffer)

        xXtrain, xXtest = split_dataset(x_data_buffer)
        yXtrain, yXtest = split_dataset(y_data_buffer)
        zXtrain, zXtest = split_dataset(z_data_buffer)

        xScaler = get_fitted_scalar(xXtrain)
        yScaler = get_fitted_scalar(yXtrain)
        zScaler = get_fitted_scalar(zXtrain)

        xXtrain = scale(xXtrain, xScaler)
        yXtrain = scale(yXtrain, yScaler)
        zXtrain = scale(zXtrain, zScaler)



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

    if model_trained:
        pass
        # Only run when
