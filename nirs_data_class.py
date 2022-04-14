# -*- coding: utf-8 -*-

"""General linear model for NIRS"""

# Authors: Anna Padée <anna.padee@unifr.ch>
#
# License: BSD-3-Clause

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import re
import filter as ft
from scipy.io import loadmat
from sklearn.decomposition import PCA
import logging
from matplotlib import ticker

log = logging.getLogger('nirs_data')
logging.basicConfig(level=logging.INFO)


def arg_array(fn):
    def decorated_function(self, data):
        array = np.asarray(data)
        return fn(self, array)
    return decorated_function


class NIRSData:
    """
    NIRS data class.

    Attributes:
        oxyChannels : oxyhemoglobin data. axes: 0-channels, 1-time points
        deoxyChannels : deoxyhemoglobin data. axes: 0-channels, 1-time points
        chLabels : Channels' names
        wavelengths : wavelengts used for recording in nm
        n_ch : number of channels
        trigger : paradigm-related events
        trigger_block : Block paradigm, 1-for condition present, 0 for baseline
        channel_distance : source-detector distance for each channel
        default_labels : Default labels for oxyhemoglobin, deoxyhemoglobin,
                         source and detector. User for plotting.
        datafile : Path to the datafile
        sources : Source number for each channel
        detectors : Detector number for each channel
        srcPos : xyz positions of sources
        detPos : xyz positions of detectors
        xyz : xyz position of channels
        region_labels : brain region names for every channel
        short_channels_ind : short channels
        channel_mapping : dictionary to get channel index from it's name

        raw_data : Raw (light intensities) data.

        self.events : events in the format of timestamps
        self.params : parameters read from nirs file
        self.fs : sampling frequency
        self.bad_channels : indices of bad channels
        self.bad_segments : bad segments, list of tuples [begin, end]
        self.m_artifacts : motion artifacts indices for every channel

        self.averaged_oxy : averaged oxyhemoglobin
        self.averaged_deoxy : averaged deoxyhemoglobin
    """
    def __init__(self):
        self.oxyChannels = np.empty(0)
        self.deoxyChannels = np.empty(0)
        self.chLabels = []
        self.wavelengths = []
        self.n_ch = None
        self.trigger = None
        self.trigger_block = None
        self.channel_distance = None
        self.default_labels = {'oxy': ' O2Hb', 'deoxy': ' HHb',
                               'src': 'Source ', 'det': 'Detector '}
        self.datafile = ""
        self.sources = []
        self.detectors = []
        self.srcPos = None
        self.detPos = None
        self.xyz = None
        self.region_labels = np.empty(0, dtype=str)
        self.short_channels_ind = []
        self.channel_mapping = {}

        self.raw_data = np.empty(0)

        self.events = []
        self.params = {}
        self.fs = 1
        self.bad_channels = []
        self.bad_segments = []
        self.m_artifacts = []

        self.averaged_oxy = None
        self.averaged_deoxy = None

        self._plot_no = 0

    def read_homer2(self, filepath: str = None,
                    ext_coeff_file="extinction_coeff.txt"):
        """
        Load homer2 *.nirs file.
        Args:
            filepath: File location. If none, searches location
                      set in nirs_data.datafile
            ext_coeff_file: A file with extinction coefficients
                            in [cm-1/(moles/liter)] for each wavelength.
                            The file has to contain columns:
                            wavelength, e for HbO, e for Hb
        Returns: None
        """
        if filepath is None:
            filepath = self.datafile
        else:
            self.datafile = filepath

        if not os.path.isfile(filepath):
            log.error("File path {} does not exist. "
                      "Exiting...".format(filepath))
            sys.exit()

        data_file = loadmat(filepath, appendmat=False)
        log.info("Processing nirs file: {}".format(filepath))
        channel_info = np.array(data_file['SD'][0][0][0]).astype(int)
        self.sources = channel_info[:, 0] - 1
        self.detectors = channel_info[:, 1] - 1
        channel_wav_lens = channel_info[:, 3] - 1
        log.info("{ns} sources, {nd} detectors found".format(
            ns=np.unique(self.sources).shape[0],
            nd=np.unique(self.detectors).shape[0]))
        self.wavelengths = data_file['SD'][0][0][1].flatten()

        self.srcPos = data_file['SD'][0][0][2]
        self.detPos = data_file['SD'][0][0][3]
        spatial_units = data_file['SD'][0][0][6][0]
        self.xyz = self.srcPos[self.sources] - 0.5 * \
                   (self.srcPos[self.sources] - self.detPos[self.detectors])
        channel_dist = self.srcPos[self.sources] - self.detPos[self.detectors]
        self.channel_distance = np.linalg.norm(channel_dist, axis=1)

        if re.match("mm", spatial_units):
            self.channel_distance *= 10
        elif not re.match("cm", spatial_units):
            raise ValueError("Unrecognized spatial units in the datafile")

        self.fs = len(data_file['t'][0]) / data_file['t'][0][-1]
        log.info("Sampling frequency: {freq:.2f} Hz".format(freq=self.fs))
        self.trigger = data_file['s']
        self.raw_data = data_file['d']

        self.n_ch = int(self.raw_data.shape[1] / self.wavelengths.shape[0])
        e_coeffs = np.loadtxt(ext_coeff_file)
        log.info("Absorption coefficients loaded "
                 "from: {}".format(ext_coeff_file))

        e = np.empty((self.wavelengths.shape[0], 2))
        for i in range(self.wavelengths.shape[0]):
            e[i, 0] = e_coeffs[np.where(e_coeffs[:, 0] ==
                                        self.wavelengths[i])[0], 1] #HbO2
            e[i, 1] = e_coeffs[np.where(e_coeffs[:, 0] ==
                                        self.wavelengths[i])[0], 2] #Hb

        e = e * 2.303 * 150 / 66.500

        self.oxyChannels = np.empty((self.n_ch, self.raw_data.shape[0]))
        self.deoxyChannels = np.empty((self.n_ch, self.raw_data.shape[0]))
        for i in range(self.n_ch):
            OD = np.vstack((self.raw_data[:, i],
                            self.raw_data[:, self.n_ch + i]))
            x = np.linalg.solve(e*self.channel_distance[i], OD)
            self.oxyChannels[i] = np.copy(x[0, :])
            self.deoxyChannels[i] = np.copy(x[1, :])
            self.chLabels.append(self.default_labels['src'] +
                                 str(self.sources[i] + 1) + "-" +
                                 self.default_labels['det'] +
                                 str(self.detectors[i] + 1))
            self.channel_mapping[self.chLabels[i]] = i

        self.sources = self.sources[:self.n_ch]
        self.detectors = self.detectors[:self.n_ch]
        self.channel_distance = self.channel_distance[:self.n_ch]
        self.xyz = self.xyz[:self.n_ch]
        self.short_channels_ind = np.where(self.channel_distance <
                                           np.mean(self.channel_distance) -
                                           2 * np.std(self.channel_distance))[0]

        self.region_labels = np.empty(self.n_ch, dtype=str)
        self.chLabels = np.array(self.chLabels).astype(str)

        log.info("Delta OD successfully converted to O2Hb and "
                 "HHb concentraion changes. {ch} channels, {tp} timepoints "
                 "loaded".format(ch=self.oxyChannels.shape[0],
                                 tp=self.oxyChannels.shape[1]))
        log.info("Recording time: "
                 "{} s".format('%.2f' % (self.oxyChannels.shape[1]/self.fs)))
        return

    def read_artinis_file(self, filepath: str=None, null_marker="NULL"):
        """
        Read standard artinis text export format.
        Args:
            filepath: Path to the data textfile
        Returns:
            channelData (numpy.ndarray): Array of data values; Two columns
                                         (O2Hb and HHb) for each channel
            params (dict): Metadata dictionary. Includes sampling rate in Hz,
                           datafile duration in seconds
                        and channel labels for each column in the data array.
            events (list): Events (keyboard interrupts) list. The format of one
                           entry is [int, str] with int being sample
                           number on which the event occurred and string
                           containing the event info (type and timestamp)
        """

        if filepath is None:
            filepath = self.datafile

        if not os.path.isfile(filepath):
            print("File path {} does not exist. Exiting...".format(filepath))
            sys.exit()

        self.datafile = filepath
        fd = open(filepath)

        columnLabels = []
        # Import metadata: sampling frequency, runtime and channel labels

        line = fd.readline()
        while line:
            line = fd.readline()
            if re.match("Datafile duration:", line):
                self.params['Datafile duration'] = re.findall("\d+\.\d+",
                                                              line)[0]
            if re.match("Export sample rate:", line):
                self.params['Export sample rate'] = re.findall("\d+\.\d+",
                                                               line)[0]
            if re.match("1\W+(Sample number)", line):
                break
        while line:
            line = fd.readline()
            if (re.match("^\d+\W+[A-Za-z\[\]]+", line) and not
                re.match(".*(Event)", line)):
                columnLabels.append(" ".join(line.split()[1:]))
                if " ".join(line.split()[1:-2]) not in self.chLabels:
                    self.chLabels.append(" ".join(line.split()[1:-2]))
            # Stops when data begins: on the line with just column numbers
            if re.match("(\d+\t*)+$", line):
                break
        self.params['Legend'] = columnLabels

        numberOfColumns = len(line.split())
        # one column is sample numbers, one is for events,
        # the rest are O2Hb and HHb values for each channel
        self.n_ch = (len(line.split()) - 2) / 2

        plainTextData = []
        # Read data, including keyboard interrupts
        line = fd.readline()
        while line:
            plainTextData.append(line.split())
            line = fd.readline()

        # Get numerical values for each channel, move event
        # data to the separate dictionary
        channelData = []
        for i in plainTextData:
            # in case of an event (keyboard interrupt) artinis appends date
            # and time to the line, making it longer.
            # This check is for this timestamp; Extra columns are then removed
            if (len(i) > numberOfColumns or i[-1] != null_marker):
                event = [int(i[0]), " ".join(i[numberOfColumns - 1:])]
                self.events.append(event)

                i = i[0:numberOfColumns]
            channelData.append(list(map(float, i[1:-1])))
        channelData = np.array(channelData)
        channelData = channelData.transpose()

        oxyind = []
        deoxyind = []
        for i in range(0, channelData.shape[0]):
            if re.match(".+O2Hb (.+)", columnLabels[i]):
                oxyind.append(i)
            elif re.match(".+HHb (.+)", columnLabels[i]):
                deoxyind.append(i)
            else:
                self.trigger = channelData[i, :]

        self.oxyChannels = channelData[oxyind, :]
        self.oxyLabels = [columnLabels[i] for i in oxyind]
        self.deoxyChannels = channelData[deoxyind, :]
        self.deoxyLabels = [columnLabels[i] for i in deoxyind]

        self.fs = float(self.params['Export sample rate'])

        log.info("Artinis text file successfully imported. "
                 "{ch} channels, {tp} timepoints "
                 "loaded".format(ch=self.oxyChannels.shape[0],
                                 tp=self.oxyChannels.shape[1]))
        log.info("Recording time: "
                 "{} s".format('%.2f' % (self.oxyChannels.shape[1] / self.fs)))

        return channelData

    def convert_events_to_trigger(self):
        """
        Convert artinis events into a time-axis trigger
        Returns:

        """
        self.trigger = np.empty(self.oxyChannels.shape[1])
        prev = [0]
        val = 0
        for event in self.events:
            self.trigger[prev[0]:event[0]] = val
            prev = event
            val = (val + 1) % 2
        return

    def paradigm_make_block_design(self, conditions='All', block_length=20):
        """
        Makes block design of the paradigm based on the trigger.
        Args:
            conditions: Which conditions to include. 'All' or a list
            block_length: Length of the block in seconds. A list (for each condition separately) or a number
            (for equal block in all conditions)

        Returns:

        """
        if conditions == 'All':
            conditions = [i for i in range(self.trigger.shape[1])]
        if type(block_length) is int or type(block_length) is float:
            block_length = [block_length] * len(conditions)

        if len(block_length) != len(conditions):
            raise ValueError("The number of block lengths: {block} \t must equal "
                             "the number of conditions {cond}".format(block=len(block_length), cond=len(conditions)))

        self.trigger_block = np.zeros((self.trigger.shape[0], len(conditions)))
        for i, cnd in enumerate(conditions):
            block_starts = np.where(self.trigger[:, cnd] > 0)[0]
            for start in block_starts:
                self.trigger_block[start: start+int(block_length[i]*self.fs), i] = 1

        return

    def read_region_labels_from_file(self, filename):
        """
        Matches region names from a textfile to channels
        :param filename: Textfile, first column: channel labels, next column(s): regions names
        :return:
        """
        #self.region_labels = [" "] * self.oxyChannels.shape[0]
        self.region_labels = np.empty([self.oxyChannels.shape[0], 2], dtype="<U10")
        channel_placement = np.loadtxt(filename, dtype=str, delimiter=',')
        for i in range(len(self.chLabels)):
            if self.chLabels[i] in channel_placement[:, 0]:
                self.region_labels[i, 0] = channel_placement[np.where(channel_placement[:, 0] == self.chLabels[i]), 1][0][0].split()[0]
                self.region_labels[i, 1] = \
                channel_placement[np.where(channel_placement[:, 0] == self.chLabels[i]), 1][0][0].split()[1]
        self.region_labels = np.array(self.region_labels).astype(str)
        return

    def plot_data(self, range_low: int = 0, range_high: int = 0,
                  channels=None,
                  n_of_plots_in_line: int = 3, max_rows_per_window: int = 5,
                  mode: str = "time", plotO2Hb: bool = True, plotHHb: bool = True, plotTrig: bool = True, condition=0):
        """
        Simple tool for displaying the data with matplotlib
        Args:
            range_low: Data subset selection: lower limit in seconds
            range_high: Data subset selection: upper limit in  seconds
            channels: list of channels to plot. Indices (int) or names (str)
            n_of_plots_in_line: Number of plots in one row
            max_rows_per_window: Maximum number of rows until new window is created
            mode:  "time", "average" or "FFT". Plots time series, averaged time series or Fourier transform of the signal
            plotO2Hb: show oxyhemoglobin
            plotHHb: showdeoxyhemoglobin
            plotTrig: show trigger
            condition: which trigger condition to show
        Returns:
            None
        """

        if not ("numpy.ndarray" in str(type(self.oxyChannels))):
            raise TypeError('Data must be numpy ndarray, not %s' % type(self.oxyChannels))

        range_low = int(range_low*self.fs)
        range_high = int(range_high*self.fs)

        if range_high <= 0:
            range_high = range_high + self.oxyChannels.shape[1]

        if range_low < 0:
            range_low = range_low + self.oxyChannels.shape[1]

        if range_low >= range_high:
            raise IndexError(
                'Requested range lower bound is higher than upper bound: (%d, %d)' % (range_low, range_high))
        if channels is None:
            channels = [i for i in range(self.n_ch)]

        if mode == "time" or mode == "FFT":
            nsamples = self.oxyChannels[:, range_low:range_high].shape[1]
        elif mode == "average":
            self.average_data(condition=condition)
            nsamples = self.averaged_oxy.shape[1]

        # formatting x axis in seconds or Hz
        if mode == "time":
            xaxis = np.linspace(range_low/self.fs, (nsamples+range_low) / self.fs, nsamples)
        elif mode == "average":
            xaxis = np.linspace(0, nsamples / self.fs, nsamples)
        elif mode == "FFT":
            xaxis = np.fft.rfftfreq(self.oxyChannels.shape[1]) * self.fs

        # number of plots
        if self.oxyChannels.shape[0] != self.deoxyChannels.shape[0]:
            raise ValueError("Oxy and deoxy channels do not match")
        #n_of_plots_in_window = min(len(channels), max_rows_per_window * n_of_plots_in_line)
        n_of_plots_in_window = max_rows_per_window * n_of_plots_in_line

        for i, ch in enumerate(channels):
            if isinstance(ch, str):
                ind = self.channel_mapping[ch]
            elif isinstance(ch, int):
                ind = ch
            else:
                raise ValueError("Wrong channel id: {}. Must be name or number".format(i))
            i_local = i % n_of_plots_in_window
            if i_local == 0:
                fig = plt.figure('NIRS O2Hb and HHb signals (' + str(self._plot_no) + ")",
                                 figsize=(32, 18))
                self._plot_no += 1
            ax = plt.subplot(int(n_of_plots_in_window / n_of_plots_in_line) + n_of_plots_in_window % n_of_plots_in_line,
                        n_of_plots_in_line, i_local + 1)
            if mode == "time":
                if plotO2Hb:
                    plt.plot(xaxis, self.oxyChannels[ind, range_low:range_high], "r", label='O2Hb')
                if plotHHb:
                    plt.plot(xaxis, self.deoxyChannels[ind, range_low:range_high], "b", label='HHb')
                if plotTrig:
                    tr_min, tr_max = ax.get_ylim()
                    if self.trigger_block is not None:
                        plt.plot(xaxis,
                                 self.trigger_block[range_low:range_high] * 0.1 * (tr_max-tr_min) + tr_min,
                                 "xkcd:dark blue", label='trigger', linewidth=1)
                    elif self.trigger is not None:
                        plt.plot(xaxis, self.trigger[range_low:range_high, condition] * 0.1 * (tr_max-tr_min) + tr_min,
                                 "xkcd:dark blue", label='trigger', linewidth=1)
                for event in self.events:
                    if range_low < event[0] < range_high:
                        plt.axvline((event[0] - range_low) / self.fs, color='k')
                if len(self.m_artifacts) > 0:
                    artifacts = np.array(self.m_artifacts[ind]).astype(int)
                    plt.plot((artifacts[(artifacts > range_low) & (artifacts < range_high)]-range_low)/self.fs,
                             self.oxyChannels[ind, artifacts[(artifacts > range_low) & (artifacts < range_high)]],
                             "*", color="xkcd:dark red", label="Motion artifacts")

            elif mode == "FFT":
                if plotO2Hb:
                    plt.plot(xaxis, np.abs(np.fft.rfft(self.oxyChannels[ind, range_low:range_high])), "r", label='O2Hb')
                if plotHHb:
                    plt.plot(xaxis, np.abs(np.fft.rfft(self.deoxyChannels[ind, range_low:range_high])), "b", label='HHb')
                plt.ylim((0, 1.05*max(np.abs(np.fft.rfft(self.oxyChannels[ind, range_low:range_high]))[int(self.fs*2):])))

            if mode == "average":
                adj_mean_o = 0
                adj_mean_d = 0
                if plotO2Hb and plotHHb:
                    adj_mean_o = np.mean(self.averaged_oxy[ind, :])
                    adj_mean_d = np.mean(self.averaged_deoxy[ind, :])
                if plotO2Hb:
                    plt.plot(xaxis, (self.averaged_oxy[ind, :] - adj_mean_o), "r", label='O2Hb')
                if plotHHb:
                    plt.plot(xaxis, (self.averaged_deoxy[ind, :] - adj_mean_d), "b", label='HHb')
            ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0e}"))
            plt.grid()
            plt.legend()
            title = ""
            if plotO2Hb:
                title = self.chLabels[ind] + self.default_labels['oxy']
                if plotHHb:
                    title += " / " + self.chLabels[ind] + self.default_labels['deoxy']
            elif plotHHb:
                title = self.chLabels[ind] + self.default_labels['deoxy']
            plt.title(title)
            if mode == "time" or mode == "average":
                if i_local >= n_of_plots_in_window - n_of_plots_in_line or i >= len(channels) - n_of_plots_in_line:
                    plt.xlabel("time (s)")
                if i_local % n_of_plots_in_line == 0:
                    plt.ylabel("Hemoglobin \n concentration changes")
                #ax.relim()
                #ax.autoscale_view()
            elif mode == "FFT":
                plt.xlabel("frequency (Hz)")
                plt.ylabel("Power spectrum of hemoglobin \n concentration changes")
            if ch in self.bad_channels:
                ax.set_facecolor('xkcd:pale peach')

            fig.tight_layout()
        return

    def mark_bad_channels(self, byHbeatPresence=True, byVariance=False, cutoff_var_factor=0.5,
                          hr_f_range=(0.7, 1.5), hr_cutoff_factor=0.5, doPlot=False):
        """
        Finding possibly noisy channels based either:
         - relative variance of oxy hemoglobin to deoxy hemoglobin changes
         - presence of heartbeat in oxyhemoglobin changes
         - or both

        :param byHbeatPresence: Marks channel as bad if it has lower percentage of frequency power spectrum in the
            requested possible heartbeat range.
        :param byVariance: Marks channel as bad if the variance of oxy changes is too close to the variance of deoxy changes.
        :param cutoff_var_factor: Percentage of the oxy changes std which the deoxy changes std has to be
        above for the channel to be marked as bad
        :param hr_f_range: Requested range for possible heartbeat frequency
        :param hr_cutoff_factor: The percentage of the maximum found heartrate contribution that a channel needs
            to be considered good
        :return: The list of bad channels, also written to self.bad_channels
        """
        self.bad_channels = []
        HRcontribution = np.empty(self.oxyChannels.shape[0])
        VARcontribution = np.empty(self.oxyChannels.shape[0])
        if byHbeatPresence:
            f = np.fft.rfftfreq(self.oxyChannels.shape[1]) * self.fs
            subset = np.where((f > hr_f_range[0]) & (f < hr_f_range[1]))[0]
        for i in range(0, int(self.oxyChannels.shape[0])):
            if byVariance:
                if np.std(self.deoxyChannels[i]) > cutoff_var_factor * np.std(self.oxyChannels[i]) and not byHbeatPresence:
                    self.bad_channels.append(i)
                VARcontribution[i] = np.std(self.deoxyChannels[i]) / np.std(self.oxyChannels[i])
            if byHbeatPresence:
                spec = np.abs(np.fft.rfft((self.oxyChannels[i, :] - np.mean(self.oxyChannels[i, :]))/np.var(self.oxyChannels[i, :])))
                HRcontribution[i] = np.sum(spec[subset]) / np.sum(spec[np.where(f > 0.2)[0]])
        if byHbeatPresence:
            log.info("Bad channel selection based on heartbeat presence: maximum heartrate contribution in "
                         "the requested range = {}%".format('%.2f' % (100*max(HRcontribution))))
            for i in range(0, int(self.oxyChannels.shape[0])):
                if not byVariance:
                    if HRcontribution[i] < (hr_cutoff_factor * HRcontribution.max()):
                        self.bad_channels.append(i)
                else:
                    if HRcontribution[i] < (hr_cutoff_factor * HRcontribution.max()) and \
                            VARcontribution[i] > cutoff_var_factor:
                        self.bad_channels.append(i)
        log.info("Bad channel selection finished. {} channels marked as bad.".format(len(self.bad_channels)))
        if doPlot:
            if byHbeatPresence:
                plt.figure()
                plt.plot(HRcontribution)
                plt.grid()
                plt.ylabel("% of spectral power in heartbeat range")
                plt.axhline(y=hr_cutoff_factor * HRcontribution.max(), color='r')
                plt.xlabel("Channels")
            if byVariance:
                plt.figure()
                plt.plot(VARcontribution)
                plt.grid()
                plt.ylabel("Deoxy/Oxy hemoglobin variance")
                plt.axhline(y=cutoff_var_factor, color='r')
                plt.xlabel("Channels")
        return self.bad_channels

    def remove_bad_channels(self):
        """
        Removal of the data marked in self.bad_channels
        :return:
        """
        self.remove_channels(self.bad_channels)
        log.info("Bad channel removal complete.")
        return

    @arg_array
    def remove_channels(self, ch_indices):
        """
        Removal of the data by given indices
        :return:
        """
        if ch_indices.size == 0:
            return
        idx = sorted(ch_indices)
        self.oxyChannels = np.delete(self.oxyChannels, idx, axis=0)
        self.deoxyChannels = np.delete(self.deoxyChannels, idx, axis=0)
        if self.averaged_oxy is not None:
            self.averaged_oxy = np.delete(self.averaged_oxy, idx, axis=0)
        if self.averaged_deoxy is not None:
            self.averaged_deoxy = np.delete(self.averaged_deoxy, idx, axis=0)
        self.sources = np.delete(self.sources, idx)
        self.detectors = np.delete(self.detectors, idx)
        self.channel_distance = np.delete(self.channel_distance, idx)
        self.region_labels = np.delete(self.region_labels, idx, axis=0)
        self.xyz = np.delete(self.xyz, idx, axis=0)
        self.chLabels = np.delete(self.chLabels, idx)
        self.n_ch = self.oxyChannels.shape[0]
        for i in idx[::-1]:
            if len(self.m_artifacts) > 0:
                self.m_artifacts.pop(i)
            if i in self.short_channels_ind:
                self.short_channels_ind = np.delete(self.short_channels_ind, np.where(self.short_channels_ind == i))
            if i in self.bad_channels:
                self.bad_channels = np.delete(self.bad_channels, np.where(self.bad_channels == i))
        for i, sch in enumerate(self.short_channels_ind):
            self.short_channels_ind[i] -= np.where(ch_indices < sch)[0].shape[0]
        for i, bch in enumerate(self.bad_channels):
            self.bad_channels[i] -= np.where(ch_indices < bch)[0].shape[0]
        self.channel_mapping = {}
        for i in range(self.n_ch):
            self.channel_mapping[self.chLabels[i]] = i

        log.info("Channel removal complete. {} channels removed.".format(len(ch_indices)))
        return

    def find_heartbeat_f(self, f_low=0.5, f_high=3, doPlot=False):
        """
        Needs denoising
        :param f_low:
        :param f_high:
        :param doPlot: plot the result?
        :return: dominant frequency in all data channels
                mean across all channels
                std from the mean
        """
        f = np.fft.rfftfreq(self.oxyChannels.shape[1]) * self.fs
        subset_f = f[np.where((f > f_low) & (f < f_high))[0]]
        peak_f = np.empty(self.oxyChannels.shape[0])
        if doPlot:
            plt.figure("Spectrum")

        max_y = 0.0
        for i in range(0, self.oxyChannels.shape[0]):
            spec = np.abs(np.fft.rfft(self.oxyChannels[i, :]))
            subset = spec[np.where((f > f_low) & (f < f_high))]
            peak_f[i] = subset_f[np.argmax(subset)]
            max_y = max(max_y, np.max(subset))
            if doPlot:
                plt.plot(f, spec, color="xkcd:steel blue")
                plt.plot(subset_f, subset, color="xkcd:clear blue")

        mean_f = np.mean(peak_f[np.where(
            (peak_f >= np.mean(peak_f) - np.std(peak_f)) & (peak_f <= np.mean(peak_f) + np.std(peak_f)))])
        std_f = np.std(
            peak_f[
                np.where((peak_f >= np.mean(peak_f) - np.std(peak_f)) & (peak_f <= np.mean(peak_f) + np.std(peak_f)))])
        if doPlot:
            plt.axvline(mean_f, color="xkcd:periwinkle",
                        label="Mean = " + '%.2f' % mean_f + ", St_dev = " + '%.2f' % std_f)
            plt.errorbar(mean_f, max_y, xerr=std_f, fmt='o', ecolor="xkcd:periwinkle")
            plt.legend()
            plt.grid()
            plt.ylim((0, max_y*1.1))
        return peak_f, mean_f, std_f

    def remove_bad_segments(self):
        ind = np.empty(0)
        for segment in self.bad_segments:
            if segment[0] < 0:
                segment[0] += self.oxyChannels.shape[1]
            if segment[1] <= 0:
                segment[1] += self.oxyChannels.shape[1]
            ind = np.concatenate((ind, np.arange(segment[0], segment[1], 1))).astype(int)
        ind = np.unique(ind)
        ind = np.sort(ind)
        self.oxyChannels = np.delete(self.oxyChannels, ind, axis=1)
        self.deoxyChannels = np.delete(self.deoxyChannels, ind, axis=1)
        if self.trigger is not None:
            self.trigger = np.delete(self.trigger, ind)
        if self.trigger_block is not None:
            self.trigger_block = np.delete(self.trigger_block, ind)
        log.info("{} bad segments removed.".format(len(self.bad_segments)))
        self.bad_segments = []
        return

    def remove_segment(self, seg):
        """
        Remove a segment from the data. Bad segments list has to be cleared first
        Args:
            seg: Segment to be removed. Array of indices or a slice

        Returns:

        """
        if len(self.bad_segments) > 0:
            log.info("Bad segment list is not empty, use remove_bad_segments() or clean it first, exiting...")
            return
        self.oxyChannels = np.delete(self.oxyChannels, seg, axis=1)
        self.deoxyChannels = np.delete(self.deoxyChannels, seg, axis=1)
        if self.trigger is not None:
            self.trigger = np.delete(self.trigger, seg)
        if self.trigger_block is not None:
            self.trigger_block = np.delete(self.trigger_block, seg)
        log.info("Segment removed.")
        self.bad_segments = []
        return

    def normalise(self, zero_mean=True, one_variance=False):
        """
        Brings all the time series to mean = 0 and variance = 1
        :return:
        """
        for i in range(0, self.oxyChannels.shape[0]):
            if zero_mean:
                self.oxyChannels[i, :] = self.oxyChannels[i, :] - np.mean(self.oxyChannels[i, :])
                self.deoxyChannels[i, :] = self.deoxyChannels[i, :] - np.mean(self.deoxyChannels[i, :])
            if one_variance:
                self.deoxyChannels[i, :] = self.deoxyChannels[i, :] / np.sqrt(np.var(self.oxyChannels[i, :]))
                self.oxyChannels[i, :] = self.oxyChannels[i, :] / np.sqrt(np.var(self.oxyChannels[i, :]))
            log.info("Normalisation complete.")
        return

    def find_motion_artifacts(self, mode="variance", discr_factor=5, time_window=1):
        """
        Automatic detection of motion artifacts.
        :param mode: either variance-shift based ("variance") or Sobel filter ("sobel"). Sobel filter approach is
                reimplemented from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5803523/#sec2.1.1 and at this point
                not recomended
        :param discr_factor: How many times std of 1s window variance must be exceeded to qualify as motion artifact
        :param time_window: Time window in seconds to search for artifacts
        :return: none, the result is saved as a list of artifacts indices for each channel in self.m_artifacts
        """
        self.m_artifacts = self.oxyChannels.shape[0] * [0]
        if mode == "sobel":
            cutOff_up = 2  # cutoff frequency of the filter in Hz
            filterOrder = 5
            temp_data = ft.butter_lowpass_filter(self.deoxyChannels, cutOff_up, fs=self.fs, order=filterOrder)
        for i in range(0, self.oxyChannels.shape[0]):
            self.m_artifacts[i] = []

            if mode == "sobel":
                temp_data[i, :] = np.convolve([-1, 0, 1], temp_data[i, :])[1:-1]
                Q1 = np.median(np.sort(temp_data[i, :])[:int(temp_data.shape[1]/2)])
                Q3 = np.median(np.sort(temp_data[i, :])[int(temp_data.shape[1]/2):])
                for j in range(0, temp_data.shape[1]):
                    if temp_data[i, j] > Q3 + 1.5*(Q3 - Q1) or temp_data[i, j] < Q1 - 1.5*(Q3 - Q1):
                        if j not in self.m_artifacts:
                            self.m_artifacts[i].append(j)
            if mode == "variance":
                var = []
                for j in range(0, self.oxyChannels.shape[1] - int(time_window * self.fs)):
                    var.append(np.var(self.deoxyChannels[i, j:j+int(time_window * self.fs)]))
                self.m_artifacts[i].extend((np.where(var > (np.mean(var) + discr_factor * np.std(var))))[0].tolist())
            log.info("{n_art} motion artifacts found on channel {ch}".format(n_art=len(self.m_artifacts[i]), ch=i))
        return

    def remove_motion_artifacts(self, seg_len=5, var_limit=0.7, continuity_adj_margin=0.5):
        """
        PCA-based motion artifact removal. Removes the leading components, then reconstructs the segment from
        the remaining ones.

        :param seg_len: length of the window in which PCA is performed
        :param var_limit: % of explained variance to be removed
        :return: nothing, the modified data is saved to self.oxyChannels and self.deoxyChannels
        """
        if self.m_artifacts == []:
            log.warning("No artifacts to remove, use find_motion_artifacts() first.")
            return

        arts_all_ch = set()

        # Join artifacts from all channels and remove duplicate entries
        for i in range(0, self.oxyChannels.shape[0]):
            arts_all_ch.update(set(self.m_artifacts[i]))

        arts_all_ch = list(arts_all_ch)
        arts_all_ch = np.sort(arts_all_ch)

        art = 0
        while art < len(arts_all_ch):
            seg = [max(0, arts_all_ch[art] - int(seg_len*self.fs)), min(self.oxyChannels.shape[1], arts_all_ch[art] + int(seg_len*self.fs))]
            if art < len(arts_all_ch) - 1:
                # check for multiple motion artifacts within the same segment
                while arts_all_ch[art + 1] <= seg[1]:
                    seg[1] = min(self.oxyChannels.shape[1], arts_all_ch[art + 1] + int(seg_len*self.fs))
                    art += 1
                    if art == len(arts_all_ch) - 1:
                        break
            art += 1
            temp_data = np.vstack((self.oxyChannels[:, seg[0]:seg[1]], self.deoxyChannels[:, seg[0]:seg[1]]))
            n_components = min(temp_data.shape)
            pca = PCA(n_components=n_components)
            pca_channels = pca.fit_transform(temp_data.T)  # estimate PCA sources

            explained_var = 0
            i = 0
            selection = pca.components_
            while explained_var < var_limit and i < n_components:
                selection[i, :] = 0
                explained_var += pca.explained_variance_ratio_[i]
                i += 1

            temp_data2 = (pca_channels.dot(selection) + pca.mean_).T

            self.oxyChannels[:, seg[0]:seg[1]] = temp_data2[0:int(temp_data2.shape[0]/2), :]
            self.deoxyChannels[:, seg[0]:seg[1]] = temp_data2[int(temp_data2.shape[0]/2):, :]

            if seg[0] > 0:
                mean_pre_oxy = np.mean(self.oxyChannels[:, seg[0] - int(continuity_adj_margin * self.fs):seg[0]],
                                       axis=1)
                mean_pre_deoxy = np.mean(self.deoxyChannels[:, seg[0] - int(continuity_adj_margin * self.fs):seg[0]],
                                         axis=1)
                mean_in_oxy = np.mean(self.oxyChannels[:, seg[0]:seg[1]], axis=1)
                mean_in_deoxy = np.mean(self.deoxyChannels[:, seg[0]:seg[1]], axis=1)
                self.oxyChannels[:, seg[0]:seg[1]] = self.oxyChannels[:, seg[0]:seg[1]] - \
                                                 mean_in_oxy[:, np.newaxis] + mean_pre_oxy[:, np.newaxis]
                self.deoxyChannels[:, seg[0]:seg[1]] = self.deoxyChannels[:, seg[0]:seg[1]] - \
                                                       mean_in_deoxy[:, np.newaxis] + mean_pre_deoxy[:, np.newaxis]
                if seg[1] < self.oxyChannels.shape[1]:
                    mean_post_oxy = np.mean(self.oxyChannels[:, seg[1]: seg[1] + int(continuity_adj_margin * self.fs)], axis=1)
                    mean_post_deoxy = np.mean(self.deoxyChannels[:, seg[1]: seg[1] + int(continuity_adj_margin * self.fs)], axis=1)
                    self.oxyChannels[:, seg[1]:] = self.oxyChannels[:, seg[1]:] - \
                                           mean_post_oxy[:, np.newaxis] + mean_pre_oxy[:, np.newaxis]
                    self.deoxyChannels[:, seg[1]:] = self.deoxyChannels[:, seg[1]:] - \
                                           mean_post_deoxy[:, np.newaxis] + mean_pre_deoxy[:, np.newaxis]
            self.m_artifacts = []
        return

    def average_data(self, condition=0, trig_label=1):
        segmented_data, _ = self.split_data_in_segments(condition=condition, trig_label=trig_label)
        avg_data = np.average(np.array(segmented_data), axis=0)
        self.averaged_oxy = avg_data[:, ::2].T
        self.averaged_deoxy = avg_data[:, 1::2].T
        return self.averaged_oxy, self.averaged_deoxy

    def split_data_in_segments(self, condition=0, trig_label=1, eq_length=True):
        if self.trigger.ndim < 2:
            trigger = np.expand_dims(self.trigger, axis=1)
        else:
            trigger = self.trigger
        data_segments = []
        trigger_blocks = []
        block_indices = np.where(trigger[:, condition] == trig_label)[0]
        block_indices = [i for i in block_indices if i+1 not in block_indices]
        seg_len = np.min(np.array(block_indices[1:]) - np.array(block_indices[:-1]))
        seg_len = min(seg_len, trigger.shape[0] - block_indices[-1] - 1)
        for n, ind in enumerate(block_indices):
            if eq_length is True:
                end_ind = seg_len + ind
            else:
                if n < len(block_indices)-1:
                    end_ind = block_indices[n+1]
                else:
                    end_ind = -1
            data_chunk = np.zeros((end_ind - ind, self.n_ch*2))
            for ch_ind in range(self.n_ch):
                data_chunk[:, ch_ind*2] = self.oxyChannels[ch_ind, ind:end_ind]
                data_chunk[:, ch_ind*2+1] = self.deoxyChannels[ch_ind, ind:end_ind]
            data_segments.append(data_chunk)
            trigger_blocks.append(trigger[ind:end_ind])
        return data_segments, trigger_blocks

    def update_spatial_coordinates(self, src_xyz, det_xyz):
        """
        Updates positions of optodes and channels
        Args:
            src_xyz: sources xyz coordinates
            det_xyz: detectors xyz coordinates

        Returns:

        """
        self.srcPos = src_xyz
        self.detPos = det_xyz
        self.xyz = self.srcPos[self.sources] - 0.5 * (self.srcPos[self.sources] - self.detPos[self.detectors])
        channel_dist = self.srcPos[self.sources] - self.detPos[self.detectors]
        self.channel_distance = np.linalg.norm(channel_dist, axis=1)
        self.short_channels_ind = np.where(self.channel_distance < np.mean(self.channel_distance) -
                                           2 * np.std(self.channel_distance))[0]
        return


def convert_nirx_trigger(trig, label=1, block=1):
    temp = np.zeros(trig.shape[0])
    result = np.zeros(trig.shape[0])
    for i in range(trig.shape[1]):
        temp += 2**i * trig[:, i]
    peaks = np.where(temp == label)[0]
    for ind in peaks:
        result[ind:ind+block] = 1
    return result
