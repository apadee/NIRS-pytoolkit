# -*- coding: utf-8 -*-

"""Filtering tools"""

# Authors: Anna Padée <anna.padee@unifr.ch>
#
# License: BSD-3-Clause


import numpy as np
from scipy.signal import butter, freqz, filtfilt
import matplotlib.pyplot as plt


def butter_lowpass(cutoff: float, fs: float, order: int = 5):
    """
    Calculate parameters of a butterworth lowpass filter.
    Args:
        cutoff: cuttoff frequency in Hz
        fs:     sample frequency in Hz
        order:  order of the filter
    Returns:
        a, b (float): Numerator (b) and denominator (a) polynomial coefficients of the filter
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 5):
    """
    Filter the data with lowpass butterworth filter
    Args:
        data: Array of data values; Two rows (O2Hb and HHb) for each channel
        cutoff: cuttoff frequency in Hz
        fs: sample frequency in Hz
        order: order of the filter
    Returns:
        result (numpy.ndarray): 2D array of filtered data. If the input was 1D, the new shape will be (1, n_of_samples)
    """
    data = np.atleast_2d(data)
    result = np.empty_like(data)

    b, a = butter_lowpass(cutoff, fs, order=order)
    for i in range(0,len(data[:, 0])):
        #filtfilt applies the same filter twice: forward and backwards, which cancels out phase distortion
        # and doesn't cause delay in the signal, unlike lfilter(), which filters the same way twice
        result[i, :] = filtfilt(b, a, data[i,:])
    return result


def plot_lowpass_response(cutOff: float, fs: float, order: int):
    """
    Plot the lowpass filter response in frequency domain
    Args:
        cutoff: cuttoff frequency in Hz
        fs: sample frequency in Hz
        order: order of the filter
    Returns:
        None
    """
    b, a = butter_lowpass(cutOff, fs, order)
    w, h = freqz(b, a)
    fig = plt.figure()
    fig.canvas.set_window_title('Lowpass filter response')
    plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
    plt.plot(cutOff, 0.5 * np.sqrt(2), 'ko')
    plt.axvline(cutOff, color='k')
    plt.xlim(0, 0.5 * fs)
    plt.title("Filter response")
    plt.xlabel('Freq[Hz]')
    plt.grid()
    plt.show()


def butter_bandpass(cut_low: float, cut_high: float, fs: float, order: int = 5):
    """
        Calculate parameters of a butterworth bandpass filter.
        Args:
            cut_low: lower cuttoff frequency in Hz
            cut_high: upper cuttoff frequency in Hz
            fs:     sample frequency in Hz
            order:  order of the filter
        Returns:
            a, b (float): Numerator (b) and denominator (a) polynomial coefficients of the filter
        """
    nyq = 0.5 * fs
    b, a = butter(order, [cut_low / nyq, cut_high / nyq], btype='band')
    return b, a


def butter_bandpass_filter(data: np.ndarray, cut_low: float, cut_high: float, fs: float, order: int = 5):
    """
        Filter the data with lowpass butterworth filter
        Args:
            data: Array of data values; Two rows (O2Hb and HHb) for each channel
            cut_low: lower cuttoff frequency in Hz
            cut_high: upper cuttoff frequency in Hz
            fs: sample frequency in Hz
            order: order of the filter
        Returns:
            result (numpy.ndarray): 2D array of filtered data. If the input was 1D, the new shape will be (1, n_of_samples)
        """
    data = np.atleast_2d(data)

    result = np.empty_like(data)

    b, a = butter_bandpass(cut_low, cut_high, fs, order=order)
    for i in range(0, len(data[:, 0])):
        # filtfilt applies the same filter twice: forward and backwards, which cancels out phase distortion
        # and doesn't cause delay in the signal, unlike lfilter(), which filters the same way twice
        result[i, :] = filtfilt(b, a, data[i, :])
    return result


def plot_bandpass_response(cut_low: float, cut_high: float, fs: float, filterOrder: int):
    """
    Plot the bandpass filter response in frequency domain
    Args:
        cut_low: lower cuttoff frequency in Hz
        cut_high: upper cuttoff frequency in Hz
        fs: sample frequency in Hz
        filterOrder: order of the filter
    Returns:
        None
    """
    b, a = butter_bandpass(cut_low, cut_high, fs, filterOrder)
    w, h = freqz(b, a)
    fig = plt.figure()
    fig.canvas.set_window_title('Bandpass filter response')
    plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
    plt.plot(cut_low, 0.5 * np.sqrt(2), 'ko')
    plt.axvline(cut_low, color='k')
    plt.plot(cut_high, 0.5 * np.sqrt(2), 'ko')
    plt.axvline(cut_high, color='k')
    plt.xlim(0, 0.5 * fs)
    plt.title("Filter response")
    plt.xlabel('Freq[Hz]')
    plt.grid()
    plt.show()

def butter_highpass(cutoff : float, fs : float, order : int = 5):
    """
    Calculate parameters of a butterworth highpass filter.
    Args:
        cutoff: cuttoff frequency in Hz
        fs:     sample frequency in Hz
        order:  order of the filter
    Returns:
        a, b (float): Numerator (b) and denominator (a) polynomial coefficients of the filter
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data : np.ndarray, cutoff : float, fs : float, order : int = 5):
    """
    Filter the data with highpass butterworth filter
    Args:
        data: Array of data values; Two rows (O2Hb and HHb) for each channel
        cutoff: cuttoff frequency in Hz
        fs: sample frequency in Hz
        order: order of the filter
    Returns:
        result (numpy.ndarray): 2D array of filtered data. If the input was 1D, the new shape will be (1, n_of_samples)
    """
    data = np.atleast_2d(data)
    result = np.empty_like(data)

    b, a = butter_highpass(cutoff, fs, order=order)
    for i in range(0,len(data[:, 0])):
        #filtfilt applies the same filter twice: forward and backwards, which cancels out phase distortion
        # and doesn't cause delay in the signal, unlike lfilter(), which filters the same way twice
        result[i, :] = filtfilt(b, a, data[i, :])
    return result


def detrend(data: np.ndarray, polydeg: int = 3, downsample: int = 1):
    """
    Removes trend from the data by approximating it with a polynomial.
    Args:
        data: Data array (channels, timepoints)
        polydeg: Degree of the polynomial used for approximation
        downsample: If set to n, takes only every n-th datapoint for approximation. By default, takes all datapoints.
    :return: Data with the trend removed for every channel
    """
    if not ("numpy.ndarray" in str(type(data))):
        raise TypeError('Data must be numpy ndarray, not %s' % type(data))

    nsamples = len(data[0, :])
    nchannels = len(data[:, 0])
    xaxis = np.array(range(0, nsamples))
    result = np.zeros((nchannels, nsamples))
    for i in range(0, nchannels):
        coeffs = np.polyfit(xaxis[::downsample], data[i, ::downsample], deg=polydeg)
        fitted = np.polyval(coeffs, xaxis)
        result[i, :] = data[i, :] - fitted

    return result
