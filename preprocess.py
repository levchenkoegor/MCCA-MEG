import os
import mne

from pathlib import Path
from bids import BIDSLayout


# Basic preprocessing
def downsample(raw, target_fs, savefile=None, overwrite=True):
    raw.resample(target_fs, npad='auto')
    if savefile:
        raw.save(savefile, overwrite=overwrite)
    return raw


def linear_filtering(raw, notch=None, l_freq=None, h_freq=None, savefile=None, overwrite=True):
    raw.notch_filter(notch, filter_length='auto', phase='zero')
    raw.filter(l_freq, h_freq, fir_design='firwin')
    if savefile:
        raw.save(savefile, overwrite=overwrite)
    return raw


def draw_psd(raw, show=False, savefile=None):
    p = raw.plot_psd(show=show, fmax=125)
    p.axes[0].set_ylim(-30, 60)
    if savefile:
        p.savefig(savefile)


# ICA
def fit_ica(raw, h_freq=1, savefile=None, verbose=False):
    raw.filter(h_freq, None)
    ica = mne.preprocessing.ICA(random_state=2, n_components=25, verbose=verbose)
    ica.fit(raw)
    if savefile:
        ica.save(savefile)
    return ica


def find_ics(raw, ica, verbose=False):
    heart_ics, _ = ica.find_bads_ecg(raw, threshold='auto', verbose=verbose)
    horizontal_eye_ics, _ = ica.find_bads_eog(raw, ch_name='MEG0521', verbose=verbose)
    vertical_eye_ics, _ = ica.find_bads_eog(raw, ch_name='MEG0921', verbose=verbose)

    all_ics = heart_ics + horizontal_eye_ics + vertical_eye_ics
    # find_bads_E*G returns list of np.int64, not int
    all_ics = map(int, all_ics)
    # Remove duplicates.
    all_ics = list(set(all_ics))

    return all_ics


def find_ics_iteratively(raw, ica, savefile=None, verbose=False):
    ics = []

    new_ics = True  # so that the while loop initiates at all
    while new_ics:
        raw_copy = raw.copy()

        # Remove all components we've found so far
        ica.exclude = ics
        ica.apply(raw_copy)
        # Identify bad components in cleaned data
        new_ics = find_ics(raw_copy, ica, verbose=verbose)

        #print(new_ics)
        ics += new_ics

    if savefile:
        f = open(savefile, 'w')
        f.write(str(ics))
        f.close()

    return ics


def apply_ica(raw, ica, ics, savefile=None, overwrite=True):
    ica.apply(raw, exclude=ics)
    if savefile:
        raw.save(savefile, overwrite=overwrite)
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


# def apply_ssp(raw):
    # something


# Maxwell filtering
def find_bad_ch_maxwell(raw):
    noisy_chs, flat_chs, scores = mne.preprocessing.find_bad_channels_maxwell(raw,
                                                                              return_scores=True,
                                                                              verbose=False)
    #print(noisy_chs)
    #print(flat_chs)

    raw.info['bads'] = raw.info['bads'] + noisy_chs + flat_chs
    return raw


def maxwell_filtering(raw, savefile=None, overwrite=True, verbose=False):
    raw_sss = mne.preprocessing.maxwell_filter(raw, verbose=verbose)
    if savefile:
        raw_sss.save(savefile, overwrite=overwrite)
    return raw_sss




# MAIN
proj_root = Path() / '..'
data_raw_dir = proj_root / 'data_raw'
data_bids_dir = proj_root / 'data_bids'
data_deriv_dir = data_bids_dir / 'derivatives'

layout = BIDSLayout(data_bids_dir, validate=True)
json_files = layout.get(suffix='meg', extension='json')

subjects = [json_file.get_entities()['subject'] for json_file in json_files]
sessions = [json_file.get_entities()['session'] for json_file in json_files]
tasks = [json_file.get_entities()['task'] for json_file in json_files]

template = os.path.join('sub-{subject}', 'ses-{session}', 'meg', 'sub-{subject}_ses-{session}_task-{task}')

raw_files_paths = [raw_file_path for raw_file_path in data_bids_dir.glob('*/*/*/*.fif')]
start_i = 15
raw_files_paths = raw_files_paths[start_i:]

for i, raw_file_path in enumerate(raw_files_paths):
    i = i+start_i
    raw = mne.io.read_raw_fif(raw_file_path, preload=True, verbose=False)
    path_savefile = data_deriv_dir / template.format(subject=subjects[i],
                                                     session=sessions[i],
                                                     task=tasks[i])
    path_savefile.parent.mkdir(parents=True, exist_ok=True)
    overwrite = True

    # Cut-off until 20 seconds before video starts
    # something

    # Basic preprocessing
    raw = downsample(raw, target_fs=250)
    draw_psd(raw, savefile=str(path_savefile) + '_PSD_before.png')
    raw = linear_filtering(raw, notch=[50, 100], l_freq=0.3,
                           savefile=str(path_savefile) + '_linear_filtering_meg.fif')
    draw_psd(raw, savefile=str(path_savefile) + '_PSD_after.png')

    # Maxwell filtering:
    # https://mne.tools/stable/auto_tutorials/preprocessing/plot_60_maxwell_filtering_sss.html#sphx-glr-auto-tutorials-preprocessing-plot-60-maxwell-filtering-sss-py
    raw = find_bad_ch_maxwell(raw)
    raw = maxwell_filtering(raw, savefile=str(path_savefile) + '_maxwell_filtering_meg.fif')

    # ECG/EOG artifacts removal
    # eog_chs = find_eog_chs(raw)
    # if eog_chs:
    #     [eog_projs_source1, eog_projs_source2], [eog_events_source1, eog_events_source2] = fit_ssp_eog(raw, eog_chs)
    # else:
    #     ica = fit_ica(raw, h_freq=1)
    #     ics = find_ics_iteratively(raw, ica)
    #     ica.apply(raw, exclude=ics)
    #     # ICA
    ica = fit_ica(raw, h_freq=1, savefile=str(path_savefile) + '_ica.fif')
    ics = find_ics_iteratively(raw, ica, savefile=str(path_savefile) + '_ics.txt')
    raw = apply_ica(raw, ica, ics, savefile=str(path_savefile) + '_applied_ICA_meg.fif')

    # continue the pipeline ->

# TODO-Preprocessing:
## check for EOG&ECG channels
## fit_ica & save ICA object
## fit_notICA & save notICA object
## exclude_EOG&ECG_components & save Raw
## check the whole pipeline - make refactoring
## *Save in derivatives
## *Empty_room noise
## *Correct events function
## *Coordinates channels conversion for future MRI coreg (maxfiltering)
## Overwrite=False looks useles know

# TODO-ISC:
## Adapt ISC function for this pipeline


# from scipy import io
# atlas = io.loadmat(r'E:\Egor_Levchenko\Epilepsy\code\code\atlas_brainnetome_labels.mat')
#
# atlas_labels = [element[0] for element in [el.tolist() for el in np.squeeze(atlas['ans']).flatten()]
#                 if 'Right' in element[0]]
#
# heading1 = [label.split(',')[0] for label in atlas_labels]
# heading2 = [label.split(',')[-1] for label in atlas_labels]
