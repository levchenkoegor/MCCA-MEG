import os
import mne
import numpy as np


### PATH READING PART
### BEGINS
path = "Y:\\LevchenkoE\\ISC\\"
os.chdir(path)

paths = [path+'\\'+f for (path, folders, files) in os.walk('Experiment3_videos_MEG\\raw_data\\')
         if 'Group' in path
         for f in files if f.endswith('tsss_mc_trans.fif')]

vid_dur = {'vid1': 380, 'vid2': 290, 'vid3': 381, 'vid4': 372}

data = mne.io.read_raw_fif(r"Y:\LevchenkoE\ISC\Experiment3_videos_MEG\raw_data\_pilot_balikoeva_darina\egor_l\200212\Video_2.fif", preload=True)
#data.plot_psd()
### ENDS


# A PACK OF FUNCTIONS FOR PREPROCESSING
def correct_events(mne_data):
    events = mne.find_events(mne_data, stim_channel='STI101')
    print('Raw events:\n', events)

    print('First samp:\n', mne_data.first_samp)

    events_corrected = np.array([[event[0] - mne_data.first_samp, event[1], event[2]] for event in events])
    print('Corrected events:\n', events_corrected)

    onset = events_corrected[-1][0] / mne_data.info['sfreq']
    descr = 'vid' + str(events_corrected[-1][-1])
    event_annot = mne.Annotations(onset=[onset], duration=[0.01], description=descr)

    mne_data.set_annotations(event_annot)

    return mne_data

def preprocess(mne_data):
    # DESCRIPTION

    mne_data.pick_types(meg=True, stim=True, eog=True, ecg=True, misc=False)
    mne_data.resample(500)
    mne_data.notch_filter(freqs=[50, 100, 150, 200])
    mne_data.filter(l_freq=0.1, h_freq=100)

    return mne_data


### PART OF PREPROCESS FUNCTION
### BEGINS
# FIND ECG ARTS
ecg_event_id = 999
n_projs = len(data.info['projs'])
ecg_events, _, _ = mne.preprocessing.find_ecg_events(data, ecg_event_id, ch_name='ECG063')
projs, events = mne.preprocessing.compute_proj_ecg(data, n_grad=1, n_mag=1, n_eeg=0, reject=None)
ecg_projs = projs[n_projs:]

mne.viz.plot_projs_topomap(ecg_projs, info=data.info)
data.add_proj(ecg_projs, remove_existing=False)

# FIND EOG ARTS
eog_event_id = 888
eog_events = mne.preprocessing.find_eog_events(data, eog_event_id, ch_name='EOG062')#, 'EOG061'])
eog_projs, _ = mne.preprocessing.compute_proj_eog(data, n_grad=1, n_mag=1, n_eeg=0, reject=None, no_proj=True)

mne.viz.plot_projs_topomap(eog_projs, info=data.info)
data.add_proj(eog_projs, remove_existing=False)

data.apply_proj()
### ENDS




### MAIN PART
data = correct_events(data)
data = preprocess(data)


# THINGS TO DO:
# 1. IS IT POSSIBLE 2 CHANNELS TO INCLUDE?
# 2. AUTOSSP PROJECTIONS - WHAT ARE THEY?
# 3. HOW TO DIVIDE INTO EPOCHS - CUT EXACT VIDEO TIMING BEFORE SAVING OR NOT?
# 4. CHECK THE WHOLE PREPROCESSING.

