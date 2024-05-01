import serial
import serial.tools.list_ports
import time

from visualize import *


def find_arduino(port=None):
    """Get the name of the port that is connected to Arduino."""
    if port is None:
        ports = serial.tools.list_ports.comports()
        for p in ports:
            print(p.manufacturer)
            if p.manufacturer is not None and ("Arduino" in p.manufacturer or "FTDI" in p.manufacturer):
                port = p.device
    return port


def get_serial_port(port=None, baudrate=115200):
    ser = None

    if port is None:
        port = find_arduino()

    if (port is not None):
        ser = serial.Serial()
        ser.baudrate = baudrate
        ser.port = port
        ser.open()
        if ser.is_open:
            print("Serial port: " + port + " is opened.")
        else:
            print("Serial port: " + port + " cannot be opened.")

    return ser


def fill_buffer(num_samples, ser):
    i = 0
    temp_buffer = []
    temp_buffer.clear()

    while i < num_samples:
        try:
            value = float(ser.readline())
        except:
            value = 0.0
        # print("value: ", value)

        temp_buffer.append(value)
        i += 1

    # return np.array(temp_buffer).reshape(-1, 1)
    return temp_buffer


def collect_dataset(samples_amount, time_amount, num_samples, ser):
    start_time = time.time()
    samples = 0

    x_data_buffer = []
    y_data_buffer = []
    z_data_buffer = []

    x_data_buffer.clear()
    y_data_buffer.clear()
    z_data_buffer.clear()

    with open("x_data.txt", "wt"):
        pass
    with open("y_data.txt", "wt"):
        pass
    with open("z_data.txt", "wt"):
        pass

    while (time.time() - start_time <= time_amount) and (samples < samples_amount):
        print(time.time() - start_time)
        print(samples)

        received_data = str(ser.readline())[2:-5].casefold()

        if received_data == "x":
            x_data = fill_buffer(num_samples, ser)
            x_data_buffer.extend(x_data)
            with open("x_data.txt", "at") as x_data_file:
                for x in x_data:
                    x_data_file.write(str(x))
                    x_data_file.write(" ")

            # Increment the samples count.
            samples += num_samples
        elif received_data == "y" and samples != 0:
            y_data = fill_buffer(num_samples, ser)
            y_data_buffer.extend(y_data)
            with open("y_data.txt", "at") as y_data_file:
                for y in y_data:
                    y_data_file.write(str(y))
                    y_data_file.write(" ")
        elif received_data == "z" and samples != 0:
            z_data = fill_buffer(num_samples, ser)
            z_data_buffer.extend(z_data)
            with open("z_data.txt", "at") as z_data_file:
                for z in z_data:
                    z_data_file.write(str(z))
                    z_data_file.write(" ")

    return np.array(x_data_buffer).reshape(-1, 1), np.array(y_data_buffer).reshape(-1, 1), np.array(z_data_buffer).reshape(-1, 1)



#     while True:
#
#         # with open("x_data.txt", "at") as x_data_file:
#         #     for x in x_data:
#         #         x_data_file.write(str(x))
#         #         x_data_file.write(" ")
#         #
#         # with open("y_data.txt", "at") as y_data_file:
#         #     for y in y_data:
#         #         y_data_file.write(str(y))
#         #         y_data_file.write(" ")
#         #
#         # with open("z_data.txt", "at") as z_data_file:
#         #     for z in z_data:
#         #         z_data_file.write(str(z))
#         #         z_data_file.write(" ")
#
#         print(x_data)
#         print(y_data)
#         print(z_data)
#         #
#         # fft_ij_x, fft_mag_x = fft_data(x_data)
#         # fft_ij_y, fft_mag_y = fft_data(y_data)
#         # fft_ij_z, fft_mag_z = fft_data(z_data)
#         # #     print(fft_mag_x)
#         # #     print(fft_mag_y)
#         # #     print(fft_mag_z)
#         #
#
#         # if received_data is not None:
#         visualize_data(x_data, y_data, z_data, sampling_frequency, "time", fig, axs)
#         # #
#         # visualize_data(fft_mag_x, fft_mag_y, fft_mag_z, sampling_frequency, "frequency", fig, axs)
#
#             # received_data = None
# else:
#     print("Port not found.")