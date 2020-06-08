# Deep Learning for Time Series Classification
# Using CNNs, by Dustin Harris

# This project takes time series data, converts it to images, and
# uses image classification from the Fast AI library.

import matplotlib.pyplot as plt
import numpy as np 
np.seterr(divide='warn', invalid='warn')
import scipy as sp
from scipy import signal
import random
import os
import math
import itertools
import sys

# Fast AI Libraries
from fastai.vision import *

# Read serial data
import serial

# Gesture Libraries from https://github.com/makeabilitylab
import gesturerec.utility as grutils
import gesturerec.data as grdata
import gesturerec.vis as grvis

from gesturerec.data import SensorData
from gesturerec.data import GestureSet

# Time Series Libraries
from pyts.image import RecurrencePlot
from pyts.datasets import load_gunpoint
from pyts.image import MarkovTransitionField

def preprocess_signal(s):
    '''Preprocesses the signal'''
    
    processed_signal = s
    
    # Resampling
    processed_signal = signal.resample(processed_signal, 100)
    
    # -1<x<1
    processed_signal = 2.*(processed_signal - np.min(processed_signal))/np.ptp(processed_signal)-1
    
    return processed_signal

path = Path('/home/dustin/Documents/FinalProject/Processed-GestureImages')
learn = load_learner(path)

# Serial reader python code found here:
# https://makersportal.com/blog/2018/2/25/python-datalogger-reading-the-serial-output-from-arduino-to-analyze-data-using-pyserial

ser = serial.Serial('/dev/ttyUSB0')

list_of_gestures = ['Shake', 'Baseball Throw', 'Midair \'S\'', 'At Rest', 'Forehand Tennis']

score = 0

ser = serial.Serial('/dev/ttyUSB0', 115200)

for gesture in list_of_gestures:
    ser.flushInput()

    stored_signals = []
    not_started = True
    finished = False
    
    print("Try for gesture: " + gesture)

    while (finished != True):
        line = ser.readline()   # read a '\n' terminated line
        all_sensors = line.decode("utf-8")
        all_sensors = all_sensors[:-2]
        all_sensors = all_sensors.split(', ')
        if (all_sensors[4] == '1'):
            not_started = False
            stored_signals.append(all_sensors)
        if (all_sensors[4] == '0' and not_started == False):
            finished = True

    sensor_values = dict()
    sensor_values["timestamps"] = []
    sensor_values["x"] = []
    sensor_values["y"] = []
    sensor_values["z"] = []

    for reading in stored_signals:
        sensor_values["timestamps"].append(reading[0])
        sensor_values["x"].append(int(reading[1]))
        sensor_values["y"].append(int(reading[2]))
        sensor_values["z"].append(int(reading[3]))

    # Create image
    
    sensor_values["x_p"] = preprocess_signal(sensor_values["x"])
    sensor_values["y_p"] = preprocess_signal(sensor_values["y"])
    sensor_values["z_p"] = preprocess_signal(sensor_values["z"])

    final_stack = np.concatenate((sensor_values["x"], sensor_values["y"]), axis=None)
    final_stack = np.concatenate((final_stack, sensor_values["z"]), axis=None)

    Images = [final_stack, final_stack]
    rp = RecurrencePlot(threshold='point', percentage=50)
    X_rp = rp.fit_transform(Images)

    plt.figure(figsize=(5, 5))
    plt.imshow(X_rp[0], cmap='binary', origin='lower')
    plt.axis('off')
    plt.tight_layout()

    directory = './Prediction/'
    filename = 'prediction.png'
    path_n_file = directory + filename
    if os.path.exists(directory):
        plt.savefig(path_n_file, bbox_inches='tight')
    else:
        print("No such file '{}'".format(path), file=sys.stderr)

    img = open_image(directory + filename)

    pred_class,pred_idx,outputs = learn.predict(img)
    print("Prediction: ", pred_class, "\n")
    if (gesture == str(pred_class)):
        score = score + 1
        
print("Accuracy: ", (score/len(list_of_gestures)))