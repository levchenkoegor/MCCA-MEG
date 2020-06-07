import os
import mne
import numpy as np
import matplotlib
import random

from pathlib import Path


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


def preprocess(mne_obj_data, subj_path):
    # DESCRIPTION

    mne_obj_data.plot_psd(show=False).savefig(subj_path/'PSD_before_ResampleFilters')

    # Downsample
    mne_obj_data.pick_types(meg=True, stim=True, eog=True, ecg=True, misc=False)
    mne_obj_data.drop_channels(['STI201', 'STI301'])
    mne_obj_data.resample(250)

    # Filter and save properties plots
    mne_obj_data.notch_filter(freqs=[50, 100])
    #notch_filt_prms = mne.filter.notch_filter(mne_obj_data.get_data(), mne_obj_data.info['sfreq'], freqs=np.array([50, 100]))
    #mne.viz.plot_filter(notch_filt_prms, mne_obj_data.info['sfreq'], flim=(0.01, 5)).savefig(subj_path.parents[1] / 'Preprocessing/NotchFilter_Properties')

    mne_obj_data.filter(l_freq=0.5, h_freq=None)
    hp_filt_prms = mne.filter.create_filter(mne_obj_data.get_data(), mne_obj_data.info['sfreq'], l_freq=0.5, h_freq=None)
    mne.viz.plot_filter(hp_filt_prms, mne_obj_data.info['sfreq'], flim=(0.01, 5)).savefig(subj_path/'HPFilter_Properties')

    mne_obj_data.plot_psd(show=False).savefig(subj_path/'PSD_after_ResampleNotchFilter')

    return mne_obj_data


def explore_ecg_artifacts(mne_obj_data, subj_path):
    # DESCRIPTION

    ecg_ch = [ch_name for ch_name in mne_obj_data.info['ch_names'] if 'ECG' in ch_name]

    if len(ecg_ch) > 0:
        print(f'ECG channel was found: {ecg_ch[0]}. Running SSP')
        ecg_projs, ecg_events = mne.preprocessing.compute_proj_ecg(mne_obj_data, n_grad=1, n_mag=1, n_eeg=0,
                                                                   ch_name=ecg_ch[0], event_id=999)
        ecg_epochs = mne.Epochs(mne_obj_data, ecg_events, tmin=-0.5, tmax=0.5, event_id=999, preload=True)

        # Save pictures
        ecg_epochs.plot_image(combine='mean', picks=['grad'], show=False)[0].savefig(subj_path/'ECG_SSP_Epochs_Grad')
        ecg_epochs.plot_image(combine='mean', picks=['mag'], show=False)[0].savefig(subj_path/'ECG_SSP_Epochs_Mag')
        ecg_epochs.average().plot_joint(show=False)[0].savefig(subj_path/'ECG_SSP_EpochsAveraged_Grad')
        ecg_epochs.average().plot_joint(show=False)[1].savefig(subj_path/'ECG_SSP_EpochsAveraged_Mag')
        mne.viz.plot_projs_topomap(ecg_projs, info=mne_obj_data.info, show=False).savefig(subj_path/'ECG_SSP_Topoplots')

        np.save(subj_path/'ECG_SSP_Projs', ecg_projs)

        return ecg_projs, ecg_events

    else:
        print('ECG channel was NOT found. Running ICA')
        ica = mne.preprocessing.ICA(n_components=20, random_state=42)
        data_hpass = mne_obj_data.filter(l_freq=1, h_freq=40)
        ica.fit(data_hpass)

        ica.plot_sources(mne_obj_data).savefig(subj_path/'ECG_ICA_Components')
        ica.plot_components()[0].savefig(subj_path/'ECG_ICA_Topographies')
        ecg_comps_inds, ecg_comps_scores = ica.find_bads_ecg(mne_obj_data)
        ica.plot_scores(ecg_comps_scores).savefig(subj_path/'ECG_ICA_Scores')

        if len(ecg_comps_inds) != 0:
             top_corr_comps_properties = ica.plot_properties(mne_obj_data, picks=ecg_comps_inds)
             [comp_prop.savefig(subj_path/'ECG_ICA_TopCompsProperties_'+str(comp_ind))
              for comp_ind, comp_prop in zip(ecg_comps_inds, top_corr_comps_properties)]

        np.save(subj_path/'ECG_ICAfit', ica)

        return ica, np.array([ecg_comps_inds, ecg_comps_scores])


def explore_eog_artifacts(mne_obj_data, subj_path):
    # DESCRIPTION

    eog_chs = [ch_name for ch_name in mne_obj_data.info['ch_names'] if 'EOG' in ch_name]

    if len(eog_chs) > 0:
        print(f'EOG channels were found: {eog_chs}. Running SSP')
        eog_projs, eog_events = mne.preprocessing.compute_proj_eog(mne_obj_data, n_grad=1, n_mag=1, n_eeg=0,
                                                                   ch_name=eog_chs[0], event_id=998)

        # Save pictures
        eog_epochs = mne.Epochs(mne_obj_data, eog_events, tmin=-0.5, tmax=0.5, event_id=998, preload=True)
        eog_epochs.plot_image(combine='mean', picks=['grad'])[0].savefig(subj_path/'EOG_SSP_ERPGrad')
        eog_epochs.plot_image(combine='mean', picks=['mag'])[0].savefig(subj_path/'EOG_SSP_ERPMag')
        [fig.savefig(subj_path/('EOG_SSP_ERPflyTopo_'+str(i))) for i, fig in enumerate(eog_epochs.average().plot_joint())]
        mne.viz.plot_projs_topomap(eog_projs, info=mne_obj_data.info).savefig(subj_path/'EOG_SSP_Topoplots')

        return eog_projs, eog_events

    else:
        print('EOG channels were NOT found. Running ICA')
        ica = mne.preprocessing.ICA(n_components=20, random_state=42)
        data_hpass = mne_obj_data.filter(l_freq=1, h_freq=40)
        ica.fit(data_hpass)

        eog_comps_indices, eog_comps_scores = ica.find_bads_eog(data, ch_name='MEG0523') #Channels near eyes
        ica.plot_scores(eog_comps_scores).savefig(subj_path/'EOG_ICA_Scores')

        if len(eog_comps_indices) != 0:
            [fig.savefig(subj_path/('EOG_ICA_Properties'+str(i))) for i, fig in
             enumerate(ica.plot_properties(mne_obj_data, picks=eog_comps_indices))]

        ica.plot_sources(mne_obj_data).savefig(subj_path/'EOG_ICA_Sources')
        [fig.savefig(subj_path/('EOG_ICA_Topoplots'+str(i))) for i, fig in enumerate(ica.plot_components())]

        return ica, np.array([eog_comps_indices, eog_comps_scores])


### MAIN PART

# Paths
path = Path(os.getcwd())

paths = [Path(path)/f for (path, folders, files) in os.walk(path.parents[0]/"data/raw_data/")
         if 'Group' in path
         for f in files if f.endswith(('vid1.fif', 'vid2.fif', 'vid3.fif', 'vid4.fif'))]

vid_dur = {'vid1': 380, 'vid2': 290, 'vid3': 381, 'vid4': 372}#seconds


# Main
def concat_anon_resample_save(paths_to_raws, bad_files):

    subj_codes = ['subject-' + str(random.randint(100, 200)) for i in range(0, int(len(paths_to_raws)))]

    subjs_names_codes = dict()
    for subj_i in range(0, len(paths_to_raws), 4):

        if any(subj_i in bad_files for subj_i in list(range(subj_i, subj_i+4))):
            print(f'Subject {subj_i} is skiiped')
            continue

        # parse subject name
        subj_sec_fir_name = str(paths_to_raws[subj_i]).split('\\')[-1].split('_')[:2]
        subj_name = '_'.join(subj_sec_fir_name)

        # load subj files and concat
        print(f'Subject i - {subj_i}, subj name - {subj_name}')
        print(f'Concatenating files: {paths_to_raws[subj_i:subj_i+4]}')
        data_raws_conds = [mne.io.read_raw_fif(paths_to_raws[cond_i], preload=True) for cond_i in range(subj_i, subj_i+4)]
        data_raw = mne.concatenate_raws(data_raws_conds)

        data_raw_anon = data_raw.anonymize()
        data_raw_anon.resample(256)

        # save dict with codes-names and raw data
        subj_code = subj_codes[subj_i]
        subjs_names_codes[subj_name] = subj_code

        Path(paths_to_raws[subj_i].parents[4], 'raw_data_concatenated', subj_code).mkdir()
        data_raw_anon.save(Path(paths_to_raws[subj_i].parents[4], 'raw_data_concatenated', subj_code, subj_code + '_ts_meg.fif'), fmt='int')

    return subjs_names_codes

bad_files = [] #[25, 27, 44, 46, 72, 73, 88, 90]
subj_codes = concat_anon_resample_save(paths[100:], bad_files)


subj_i_paths = []
data_raws_conds = [mne.io.read_raw_fif(paths[cond_i], preload=True) for cond_i in range(0, 4)]
data_raw = mne.concatenate_raws(data_raws_conds)

for subj_i in range(26, len(paths)):
    data = mne.io.read_raw_fif(paths[subj_i], preload=True)

    data = correct_events(data_raw)
    data = preprocess(data, paths[subj_i].parents[1]/'Preprocessing/')
    ecg_projs, ecg_info = explore_ecg_artifacts(data, paths[subj_i].parents[1]/'Preprocessing/')
    matplotlib.pyplot.close('all')
    eog_projs, eog_info = explore_eog_artifacts(data, paths[subj_i].parents[1]/'Preprocessing/')
    matplotlib.pyplot.close('all')

    print(subj_i)



# THINGS TO DO:
# 2. AUTOSSP PROJECTIONS - WHAT ARE THEY?
# 3. HOW TO DIVIDE INTO EPOCHS - CUT EXACT VIDEO TIMING BEFORE SAVING OR NOT?
# 4. CHECK THE WHOLE PREPROCESSING.

###WHAT TO DO?
#1. Save plot of SSP components for ECG removal - ?
#2. Adjust ICA for ECG artifacts.
#3. Make SSP for EOG artifacts.
#4. Adjust ICA for ECG artifacts.
#5. Add function to remove components (semi-manually)
#6. Establish paths without "hardcoding"
#7. Run the preprocessing for all subjects

#8 SUBJ_I = 3, 24, 25, 30 DOESNT WORK EOG ARTIFACTS



