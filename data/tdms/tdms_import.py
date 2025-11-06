"""
TDMS file import and processing utilities.

This module provides functions for reading and processing TDMS (Technical Data
Management Streaming) files from measurement systems. It handles:
- Reading motor control TDMS files
- Reading measurement result TDMS files
- Time synchronization and interpolation
- Data alignment and conversion to CSV format
- Time jump detection and correction

The module is designed to process testbench measurement data where motor
control signals and sensor measurements are stored in separate TDMS files
with potentially different sampling rates.
"""

from nptdms import TdmsFile

import csv
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
 
 
def motor_read_tdms(tdms_file_path):
    """
    Read motor control TDMS file and extract control signals.
    
    Reads a TDMS file containing motor control data (torque and velocity
    setpoints and measurements). The data is stored in a 9-column format
    representing time and various motor/propeller control signals.
 
    Parameters:
    -----------
    tdms_file_path : str
        Path to the input motor control .tdms file.
        
    Returns:
    --------
    headers : list
        List of column headers: ["Time", "MotorTorqueSet", "MotorTorque",
        "MotorVelocitySet", "MotorVelocity", "PropellerTorqueSet",
        "PropellerTorque", "PropellerVelocitySet", "PropellerVelocity"]
    channel_data : ndarray
        Reshaped data array (N x 9) containing all motor control signals
    """
 
    # Ensure the TDMS file data is written in a tabular format
    try:
        # Read the TDMS file
        tdms_file = TdmsFile.read(tdms_file_path)
        
        headers = ["Time", "MotorTorqueSet", "MotorTorque", "MotorVelocitySet", "MotorVelocity", "PropellerTorqueSet", "PropellerTorque", "PropellerVelocitySet", "PropellerVelocity"]

        group = tdms_file.groups()[0]
        channel = group.channels()[0]
        
        channel_data = channel.data.tolist()
        if not channel_data:
            print(f"Channel {channel.name} has no data.")
        else:
            print(f"Channel {channel.name} has data.")
        
        channel_data = np.reshape(np.array(channel_data), (-1, 9))

        print(f"TDMS file successfully read")
    except Exception as e:
        print(f"Error converting TDMS: {e}")
 
    return headers, channel_data

def result_read_tdms(tdms_file_path):
    """
    Read measurement result TDMS file and extract sensor data.
    
    Reads a TDMS file containing measurement data from various sensors
    (encoders, torque sensors, etc.). All channels from all groups are
    collected and returned as a transposed array for easy column access.
 
    Parameters:
    -----------
    tdms_file_path : str
        Path to the input measurement result .tdms file.
        
    Returns:
    --------
    header : list
        List of channel names from all groups in the TDMS file
    data : ndarray
        Transposed data array (N x n_channels) containing all sensor
        measurements with each column corresponding to a channel
    """
 
    # Ensure the TDMS file data is written in a tabular format
    try:
        # Read the TDMS file
        tdms_file = TdmsFile.read(tdms_file_path)

        # Collect all channel names for the header row
        header = []
        data = []
        for group in tdms_file.groups():
            for channel in group.channels():
                header.append(channel.name)
                channel_data = channel.data.tolist()
                if not channel_data:
                    print(f"Channel {channel.name} has no data.")
                else:
                    print(f"Channel {channel.name} has data.")
                    data.append(channel_data)
            
        print(f"TDMS file successfully read")
    except Exception as e:
        print(f"Error converting TDMS: {e}")
 
    return header, np.array(data).T
 
def remove_time_jumps(time_vector, threshold=0.05):
    """
    Removes large time jumps from a time vector by adjusting subsequent values.
    
    Args:
        time_vector (np.array): Input time vector
        threshold (float): Threshold for detecting large time jumps (default: 0.05)
    
    Returns:
        np.array: Time vector with jumps removed
    """
    time_diff = np.diff(time_vector)
    large_diff_indices = np.where(np.abs(time_diff) > threshold)[0]
    
    corrected_time = time_vector.copy()
    for idx in large_diff_indices:
        corrected_time[idx+1:] -= time_diff[idx]
    
    return corrected_time

def process_measurement_data(result_path, motor_path, output_file_path):
    """
    Process and synchronize measurement data from TDMS files.
    
    Combines motor control data and sensor measurement data from separate
    TDMS files, synchronizes them to a common time base, applies unit
    conversions, and writes the result to a CSV file.
    
    Processing steps:
    1. Reads motor control and measurement result TDMS files
    2. Selects relevant columns from each dataset
    3. Converts time stamps to seconds and removes time jumps
    4. Interpolates motor data to measurement time base
    5. Converts encoder angles to radians
    6. Scales torque values by calibration factors
    7. Combines and writes to CSV
    
    Parameters:
    -----------
    result_path : str
        Path to measurement result TDMS file
    motor_path : str
        Path to motor control TDMS file
    output_file_path : str
        Path to output CSV file for processed data
    """
    try:
        motor_headers, motor_data = motor_read_tdms(motor_path)
        result_headers, result_data = result_read_tdms(result_path)

        # Select the required columns
        selected_columns_motor = [0, 2, 4, 6, 8]  # Indices of the required columns
        selected_headers_motor = [motor_headers[i] for i in selected_columns_motor]
        motor_data = motor_data[:, selected_columns_motor]

        selected_columns_result = [0, 1, 2, 3, 4, 15, 16]
        selected_headers_result = [result_headers[i] for i in selected_columns_result]
        result_data = result_data[:, selected_columns_result]
        
        motor_data[:,0] = (motor_data[:,0] - motor_data[0,0])
        motor_data[:,0] = remove_time_jumps(motor_data[:,0])

        result_data[:,0] = (result_data[:,0] - result_data[0,0]) * 25e-9
        result_data[:,0] = remove_time_jumps(result_data[:,0])

        
        # Interpolate motor data
        motor_data_interp = np.zeros((len(result_data[:,0]), motor_data.shape[1]))
        for i in range(motor_data.shape[1]):
            interpolation_func = interp1d(motor_data[:,0], motor_data[:,i], kind="linear", fill_value="extrapolate")
            motor_data_interp[:,i] = interpolation_func(result_data[:,0])
        
        motor_data = motor_data_interp

        # Combine data
        data = np.hstack((result_data, motor_data[:,1:]))
        data = data[3000:,:]

        headers = selected_headers_result + selected_headers_motor[1:]

        # Open the output file for writing
        with open(output_file_path, mode='w', newline='') as output_file:
            csv_writer = csv.writer(output_file)

            # Write the selected headers
            csv_writer.writerow(headers)

            # Calculate encoder time
            for i in [1, 3]:
                data[:, i] = (data[:, i] - data[0, i]) * 25e-9
                data[:, i] = remove_time_jumps(data[:, i])
            
            # Convert encoder angle readout to radians
            for i in [2, 4]:
                data[:, i] = data[:, i] * (2 * np.pi) / 20e3

            # Get proper torque values
            data[:, 5] *= 10
            data[:, 6] *= 4

            # Write the processed data to CSV
            for row in data:
                csv_writer.writerow(row)
 
        print(f"Processed data written to: {output_file_path}")
    except Exception as e:
        print(f"Error processing measurement data: {e}")

if __name__ == "__main__":
    process_measurement_data("data/IceExcitation_3200rpm_0.tdms", "data/IceExcitation_3200rpm_0_motor.tdms","data/IceExcitation_3200rpm_0.csv")