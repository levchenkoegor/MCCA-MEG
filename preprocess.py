import os
import mne
import numpy as np
import matplotlib
import random
from pathlib import Path

# Basic preprocessing
def downsample(raw, target_fs):
    raw.resample(target_fs, npad='auto')
    return raw

def linear_filtering(raw, notch=None, l_freq=None, h_freq=None):
    raw.notch_filter(notch, filter_length='auto', phase='zero')
    raw.filter(l_freq, h_freq, fir_design='firwin')
    return raw

def draw_psd(raw, show=False, savefig=True):
    p = raw.plot_psd(show=show, fmax=150)
    p.axes[0].set_ylim(-30, 60)
    #p.savefig(output[0])

# ICA
def fit_ica(raw, h_freq=1):
    raw.filter(h_freq, None)
    ica = mne.preprocessing.ICA(random_state=2, n_components=25, verbose=False)
    ica.fit(raw)
    return ica

def find_ics(raw, ica, verbose=False):
    heart_ics, _ = ica.find_bads_ecg(raw, verbose=verbose)
    horizontal_eye_ics, _ = ica.find_bads_eog(raw, ch_name='MLF14-1609', verbose=verbose)
    vertical_eye_ics, _ = ica.find_bads_eog(raw, ch_name='MLF21-1609', verbose=verbose)

    all_ics = heart_ics + horizontal_eye_ics + vertical_eye_ics
    # find_bads_E*G returns list of np.int64, not int
    all_ics = map(int, all_ics)
    # Remove duplicates.
    all_ics = list(set(all_ics))

    return all_ics

def find_ics_iteratively(raw, ica, verbose=False):
    ics = []

    new_ics = True  # so that the while loop initiates at all
    while new_ics:
        raw_copy = raw.copy()

        # Remove all components we've found so far
        ica.exclude = ics
        ica.apply(raw_copy)
        # Identify bad components in cleaned data
        new_ics = find_ics(raw_copy, ica, verbose=verbose)

        print(new_ics)
        ics += new_ics

    return ics

def apply_ica(raw, ica, ics): #RECHECK
    ica.apply(raw)
    return raw

# SSP
def find_eog_chs(raw):
    eog_chs = [ch_name for ch_name in raw.info['ch_names'] if 'EOG' in ch_name]
    return eog_chs

def find_ecg_chs(raw):
    ecg_chs = [ch_name for ch_name in raw.info['ch_names'] if 'ECG' in ch_name]
    return ecg_chs

def fit_ssp_eog(raw, eog_chs):
    eog_projs_source1, eog_events_source1 = mne.preprocessing.compute_proj_eog(raw, n_grad=1, n_mag=1, n_eeg=0,
                                                                               ch_name=eog_chs[0], event_id=990)
    eog_projs_source2, eog_events_source2 = mne.preprocessing.compute_proj_eog(raw, n_grad=1, n_mag=1, n_eeg=0,
                                                                               ch_name=eog_chs[1], event_id=991)
    return [eog_projs_source1, eog_projs_source2], [eog_events_source1, eog_events_source2]

def fit_ssp_ecg(raw, ecg_chs):
    ecg_projs, ecg_events = mne.preprocessing.compute_proj_ecg(raw, n_grad=1, n_mag=1, n_eeg=0,
                                                               ch_name=ecg_chs, event_id=988)
    return ecg_projs, ecg_events

#def apply_ssp(raw):
    # something


#Preprocessing:
#downsample
#draw_psd & save
#filtering & save
#draw_psd & save
#check for EOG&ECG channels
#fit_ica & save ICA object
#fit_notICA & save notICA object
#exclude_EOG&ECG_components & save Raw

#*Save in derivatives
#*Empty_room noise
#*Correct events function
#*Coordinates channels conversion for future MRI coreg (maxfiltering)

#ISC calculations:
#...

proj_root = Path() / '..'
data_raw_dir = proj_root / 'data_raw'
data_bids_dir = proj_root / 'data_bids'

raw = mne.io.read_raw_fif(r"E:\Egor_Levchenko\ISC\Experiment3_videos_MEG\data_bids\sub-1001\ses-20190416\meg\sub-1001_ses-20190416_task-vid1_meg.fif", preload=True)
raw = downsample(raw, 250)
raw = linear_filtering(raw, notch=[50, 100], l_freq=0.3)
#continue the pipeline ->

