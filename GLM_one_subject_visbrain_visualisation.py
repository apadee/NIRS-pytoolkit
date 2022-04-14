# -*- coding: utf-8 -*-

"""An example file with 3d visualisation using visbrain"""

# Authors: Anna Padée <anna.padee@unifr.ch>
#
# License: BSD-3-Clause

import numpy as np
import re
import matplotlib.pyplot as plt
from nirs_data_class import NIRSData
import filter as ft
from GLM_NIRS import GLM_NIRS
from visbrain.gui import Brain
from visbrain.objects import BrainObj, SourceObj, ConnectObj, \
    ColorbarObj, SceneObj

# path to the data
filepath = "/mnt/data/NIRS/data/sub-005/sub-005.nirs"
# path to a textfile with optode coordinates, recorded from a digitizer
loc_file = '/mnt/data/data/1219denoising/2020-01-20/positions.txt'

def read_coordinates(filename):
    """Reading 3D optodes coordinates from a text file"""
    locations = np.loadtxt(filename, dtype=str)
    src_ind = []
    rec_ind = []
    ref_ind = []
    for i in range(locations.shape[0]):
        if re.search("rec", locations[i, 0], re.IGNORECASE):
            rec_ind.append(i)
        elif re.search("src", locations[i, 0], re.IGNORECASE):
            src_ind.append(i)
        else:
            ref_ind.append(i)

    labels = locations[:, 0]
    locations = locations[:, (2, 1, 3)].astype(float)
    locations[:, 0] = locations[:, 0] * (-1)

    loc_src = locations[src_ind, :]
    loc_rec = locations[rec_ind, :]
    loc_ref = locations[ref_ind, :]

    return locations, labels, loc_src, loc_rec, loc_ref


# Initialize and read data
data = NIRSData()
data.default_labels['src'] = ''
data.default_labels['det'] = ''
data.read_homer2(filepath=filepath)
data.short_channels_ind = [20, 40]

# Convert triggering signal (which in many systems only records changes, not
# block) into a block design, with each stimulus lasting 20s
data.paradigm_make_block_design(block_length=20, conditions=[0])

# Filter data with a lowpass butterworth filter
time_series = ft.butter_lowpass_filter(data.deoxyChannels, cutoff=0.7,
                                       fs=data.fs, order=5)
# Read optode coordinates and update the data object
locations, loc_labels, src_coord, rec_coord, ref_coord = \
    read_coordinates(loc_file)
data.update_spatial_coordinates(src_coord, rec_coord)

# Create design matrix. Short channels are included as nuisance
# regressors in the model. First and second derivatives of the paradigm are
# not used in the analysis
glm = GLM_NIRS()
glm.fs = data.fs
glm.create_design_matrix(data.trigger_block[:, 0],
                         np.atleast_2d(time_series[data.short_channels_ind, :]),
                         first_derivative=False, sec_derivative=False)
# Check the design matrix. The top regressor is the regressor of interest
glm.show_design()

# Compute the model. Short channels are excluded from the analysis.
result_glm = glm.fit(np.delete(time_series, data.short_channels_ind, axis=0))
result_pval = glm.result_pval
result_beta = np.array(result_glm[0][:, 0]).astype(float)
result_ttest = np.array(result_glm[2]).astype(float)

# Define order and color coded regions for each channel. Used for bar display
# of the results
order = np.array([0, 1,  4,  5,  6,  7,  8,  9, 13, 14, 15, 16, 17, 31, 32,
                  33, 34, 36, 37, 38, 22, 23, 24, 25, 26, 27, 28, 29, 30, 35,
                  2,  3, 10, 11, 12, 18, 19, 20, 21])
colors = ['xkcd:violet', 'xkcd:violet', 'xkcd:violet', 'xkcd:violet',
          'xkcd:periwinkle', 'xkcd:periwinkle', 'xkcd:periwinkle',
          'xkcd:periwinkle', 'xkcd:periwinkle', 'xkcd:lavender',
          'xkcd:lavender', 'xkcd:lavender', 'xkcd:lavender',
          'xkcd:deep green', 'xkcd:deep green', 'xkcd:deep green',
          'xkcd:deep green', 'xkcd:deep green', 'xkcd:deep green',
          'xkcd:deep green', 'xkcd:leafy green', 'xkcd:leafy green',
          'xkcd:leafy green', 'xkcd:leafy green', 'xkcd:leafy green',
          'xkcd:leafy green', 'xkcd:leafy green', 'xkcd:leafy green',
          'xkcd:leafy green', 'xkcd:leafy green', 'xkcd:deep blue',
          'xkcd:deep blue', 'xkcd:cornflower', 'xkcd:cornflower',
          'xkcd:cornflower', 'xkcd:light blue', 'xkcd:light blue',
          'xkcd:marigold', 'xkcd:marigold']
legend = {'Frontal left': 'xkcd:violet', 'Frontal center': 'xkcd:periwinkle',
          'Frontal right': 'xkcd:lavender', 'Motor left': 'xkcd:deep green',
          'Motor center': 'xkcd:leafy green',
          'Prefrontal left': 'xkcd:deep blue',
          'Prefrontal center': 'xkcd:cornflower',
          'Prefrontal right': 'xkcd:light blue',
          'Temporal left': 'xkcd:marigold'}

glm.show_results_bar(mode="t-test", order=order, colors=colors,
                     labels=np.array(data.chLabels), color_labels=legend)

# Visbrain 3d visualisation
# remove short channels, as they are not part of the GLM result
data.remove_channels(data.short_channels_ind)
sc = SceneObj(size=(1000, 1000), bgcolor='white')
#Make symmetrical limits, so that 0 is in the middle of the scale
clim = (min(np.min(result_ttest), -1*np.max(result_ttest)),
                  max(np.max(result_ttest), -1*np.min(result_ttest)))
kwargs = {}
kwargs['color'] = "magenta"
kwargs['alpha'] = 0.7
kwargs['data'] = result_ttest
kwargs['radius_min'] = 20
kwargs['radius_max'] = 21
kwargs['symbol'] = 'o'
kwargs['text_size'] = 10.0
kwargs['text'] = data.chLabels
kwargs['clim'] = clim
cb_kw = dict(cblabel="Task related activation", cbtxtsz=3., border=False,
             cmap='seismic', clim=clim)

xyz = data.xyz

s_obj = SourceObj('Sobj', xyz, **kwargs)
s_obj.color_sources(data=result_ttest, cmap='seismic')
b_obj = BrainObj('B2', **cb_kw, translucent=False)

# Use vertices of the barim model to move sources inside of the brain.
# Visbrain cannot project on the surface sources that are above the brain
b_obj_vert = b_obj.vertices
s_obj.fit_to_vertices(b_obj_vert)
fitted_coord = s_obj._sources._data['a_position']

s1_obj = SourceObj('Sobj', fitted_coord * 0.95, **kwargs)
s1_obj.color_sources(data=result_ttest, cmap='seismic')
s1_obj.project_sources(b_obj, cmap='seismic', clim=clim)

sc.add_to_subplot(s1_obj, row=0, col=0, title='Task-related activation',
                  width_max=900)
sc.add_to_subplot(b_obj, row=0, col=0, rotate='axial_0', use_this_cam=True,
                  width_max=900)

CBAR_STATE = dict(cbtxtsz=18, txtsz=15., width=.5, cbtxtsh=1.8,
                  txtcolor='black', rect=(1., -2., 1., 4.))
cb = ColorbarObj(s1_obj, cblabel='T-stat values', **CBAR_STATE)
sc.add_to_subplot(cb, row=0, col=1, width_max=80)
sc.preview()
