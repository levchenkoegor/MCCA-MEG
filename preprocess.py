import os
import mne
import numpy as np
import matplotlib
matplotlib.use('agg')


path = "Y:\\LevchenkoE\\ISC\\Experiment3_videos_MEG\\"
os.chdir(path)

paths = [path+'\\'+f for (path, folders, files) in os.walk('data\\')
         if 'Group' in path
         for f in files if f.endswith('tsss_mc_trans.fif')]

vid_dur = {'vid1': 380, 'vid2': 290, 'vid3': 381, 'vid4': 372}# in seconds

data = mne.io.read_raw_fif(paths[4], preload=True)


def correct_events(mne_obj_data):
    # DESCRIPTION

    events = mne.find_events(mne_obj_data, stim_channel='STI101')
    print('Raw events:\n', events)

    print('First samp:\n', mne_obj_data.first_samp)

    events_corrected = np.array([[event[0] - mne_obj_data.first_samp, event[1], event[2]] for event in events])
    print('Corrected events:\n', events_corrected)

    onset = events_corrected[-1][0] / mne_obj_data.info['sfreq']
    descr = 'vid' + str(events_corrected[-1][-1])
    event_annot = mne.Annotations(onset=[onset], duration=[0.01], description=descr)

    mne_obj_data.set_annotations(event_annot)

    return mne_obj_data


def preprocess(mne_obj_data):
    # DESCRIPTION

    mne_obj_data.plot_psd(show=False).savefig('PSD_before_ResampleNotchFilter')

    mne_obj_data.pick_types(meg=True, stim=True, eog=True, ecg=True, misc=False)
    mne_obj_data.drop_channels(['STI201', 'STI301'])
    mne_obj_data.resample(250)
    mne_obj_data.notch_filter(freqs=[50, 100])
    mne_obj_data.filter(l_freq=0.5, h_freq=None)

    mne_obj_data.plot_psd(show=False).savefig('PSD_after_ResampleNotchFilter')

    return mne_obj_data


def find_ecg_projs(mne_obj_data):
    # DESCRIPTION

    ecg_ch = [ch_name for ch_name in mne_obj_data.info['ch_names'] if 'ECG' in ch_name]

    if len(ecg_ch) > 0:
        print(f'ECG channel was found: {ecg_ch[0]}. Running SSP')
        ecg_projs, ecg_events = mne.preprocessing.compute_proj_ecg(mne_obj_data, n_grad=1, n_mag=1, n_eeg=0,
                                                                   ch_name=ecg_ch[0], event_id=999)
        ecg_epochs = mne.Epochs(mne_obj_data, ecg_events, tmin=-0.5, tmax=0.5, event_id=999, preload=True)

        ecg_epochs.plot_image(combine='mean', picks=['grad'], show=False)[0].savefig('ECG_Epochs_Grad')
        ecg_epochs.plot_image(combine='mean', picks=['mag'], show=False)[0].savefig('ECG_Epochs_Mag')
        ecg_epochs.average().plot_joint(show=False)[0].savefig('ECG_EpochsAveraged_Grad')
        ecg_epochs.average().plot_joint(show=False)[1].savefig('ECG_EpochsAveraged_Mag')
        mne.viz.plot_projs_topomap(ecg_projs, info=mne_obj_data.info, show=False).savefig('ECG_Topoplots')
        mne_obj_data.add_proj(ecg_projs)

        return ecg_projs, ecg_events

    else:
        print('ECG channel was NOT found. Running ICA')
        ica = mne.preprocessing.ICA(n_components=15)
        data_hpass = mne_obj_data.filter(l_freq=1, h_freq=40)
        ica.fit(data_hpass)
        eog_indices, eog_scores = ica.find_bads_eog(mne_obj_data)
        ica.plot_scores(eog_scores)
        ica.plot_properties(mne_obj_data, picks=eog_indices)

        ica.plot_sources(mne_obj_data)
        ica.plot_components()

        return ica, eog_indices, eog_scores


def find_eog_projs(mne_obj_data):
    # DESCRIPTION

    eog_chs = [ch_name for ch_name in mne_obj_data.info['ch_names'] if 'EOG' in ch_name]

    if len(eog_chs) > 0:
        print(f'EOG channels were found: {eog_chs}. Running SSP')
        eog_projs, eog_events = mne.preprocessing.compute_proj_eog(mne_obj_data, n_grad=1, n_mag=1, n_eeg=0, ch_name=eog_chs[0],
                                                                   event_id=998)
        eog_epochs = mne.Epochs(mne_obj_data, eog_events, tmin=-0.5, tmax=0.5, event_id=998, preload=True)
        eog_epochs.plot_image(combine='mean', picks=['grad'])
        eog_epochs.plot_image(combine='mean', picks=['mag'])
        eog_epochs.average().plot_joint()
        mne.viz.plot_projs_topomap(eog_projs, info=mne_obj_data.info)

        return eog_projs, eog_events

    else:
        print('EOG channels were NOT found. Running ICA')
        ica = mne.preprocessing.ICA(n_components=15)
        data_hpass = mne_obj_data.filter(l_freq=1, h_freq=40)
        ica.fit(data_hpass)
        eog_indices, eog_scores = ica.find_bads_eog(data)
        ica.plot_scores(eog_scores)
        ica.plot_properties(mne_obj_data, picks=eog_indices)

        ica.plot_sources(mne_obj_data)
        ica.plot_components()

        return ica, eog_indices, eog_scores


# MAIN PART
data = correct_events(data)
data = preprocess(data)
eog_projs, eog_events = find_ecg_projs(data)



# THINGS TO DO:
# 2. AUTOSSP PROJECTIONS - WHAT ARE THEY?
# 3. HOW TO DIVIDE INTO EPOCHS - CUT EXACT VIDEO TIMING BEFORE SAVING OR NOT?
# 4. CHECK THE WHOLE PREPROCESSING.

