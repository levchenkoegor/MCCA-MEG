import os
import mne
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot
from pathlib import Path
from bids import BIDSLayout


# Helpers
def set_annot_from_events(raw, verbose=False):
    events = mne.find_events(raw, stim_channel='STI101', verbose=verbose)  # find all events
    onsets = events[:, 0] / raw.info['sfreq']  # calculate onsets in seconds
    durations = np.zeros_like(onsets)
    descriptions = [str(event_id) for event_id in events[:, -1]]  # '1', '2', '3', '4'
    annot_from_events = mne.Annotations(onset=onsets, duration=durations,
                                        description=descriptions,
                                        orig_time=raw.info['meas_date'])
    raw.set_annotations(annot_from_events)
    return raw


def increase_raw_length(raw, t=30):
    raw_crop = raw.copy().crop(tmax=t)
    raw.append(raw_crop)
    return raw


def decrease_raw_length(raw, events, t_before_event=30):

    return raw


# ICA
def find_ics(raw, ica, eog_chs=('MEG0521', 'MEG0921'), verbose=False):
    eog_chs_raw = [ch_name for ch_name in raw.info['ch_names'] if 'EOG' in ch_name]
    eog_chs = eog_chs_raw if eog_chs_raw else eog_chs

    heart_ics, _ = ica.find_bads_ecg(raw, threshold='auto', verbose=verbose)
    horizontal_eye_ics, _ = ica.find_bads_eog(raw, ch_name=eog_chs[0], verbose=verbose)
    vertical_eye_ics, _ = ica.find_bads_eog(raw, ch_name=eog_chs[1], verbose=verbose)

    all_ics = heart_ics + horizontal_eye_ics + vertical_eye_ics
    # find_bads_E*G returns list of np.int64, not int
    all_ics = map(int, all_ics)
    # Remove duplicates.
    all_ics = list(set(all_ics))

    return all_ics


def find_ics_iteratively(raw, ica, savefile=None, visualization=False, verbose=False):
    ics = []

    new_ics = True  # so that the while loop initiates at all
    i = 0
    while new_ics and i < 3:  # why it goes infinity?
        raw_copy = raw.copy()

        # Remove all components we've found so far
        ica.exclude = ics
        ica.apply(raw_copy)
        # Identify bad components in cleaned data
        new_ics = find_ics(raw_copy, ica, verbose=verbose)

        # print(new_ics)
        ics += new_ics
        i += 1

    if visualization:
        # plot diagnostics
        ica.plot_properties(raw, picks=ics)

        # plot ICs applied to raw data, with EOG matches highlighted
        ica.plot_sources(raw, show_scrollbars=False)

    if savefile:
        f = open(savefile, 'w')
        f.write(str(ics))
        f.close()

    return ics


# SSP
def fit_ssp_eog(raw, eog_chs, savefile=None, verbose=False):
    eog_projs_source1, eog_events_source1 = mne.preprocessing.compute_proj_eog(raw, n_grad=1, n_mag=1, n_eeg=0, no_proj=True,
                                                              ch_name=eog_chs[0], event_id=990, verbose=verbose)
    eog_projs_source2, eog_events_source2 = mne.preprocessing.compute_proj_eog(raw, n_grad=1, n_mag=1, n_eeg=0, no_proj=True,
                                                              ch_name=eog_chs[1], event_id=991, verbose=verbose)
    eog_projs = eog_projs_source1 + eog_projs_source2
    eog_events = np.vstack((eog_events_source1, eog_events_source2))
    if savefile:
        mne.write_proj(savefile, eog_projs)
    return eog_projs, eog_events


def fit_ssp_ecg(raw, ecg_chs, savefile=None, verbose=False):
    ecg_projs, ecg_events = mne.preprocessing.compute_proj_ecg(raw, n_grad=1, n_mag=1, n_eeg=0, no_proj=True,
                                                               ch_name=ecg_chs, event_id=988, verbose=verbose)
    if savefile:
        mne.write_proj(savefile, ecg_projs)
    return ecg_projs, ecg_events


def apply_ssp_proj(raw, projs, verbose=False):
    raw.add_proj(projs)
    raw.apply_proj(verbose=verbose)
    return raw


def ssp_routine(raw, eog_chs, ecg_chs):
    eog_projs, _ = fit_ssp_eog(raw, eog_chs, savefile=str(path_savefile) + 'eog_projs.fif')
    ecg_projs, _ = fit_ssp_ecg(raw, ecg_chs, savefile=str(path_savefile) + 'ecg_projs.fif')
    # plot projs and etc
    raw = apply_ssp_proj(raw, projs=eog_projs+ecg_projs)
    return raw


# Maxwell filtering
def plot_maxwell_bad_ch(auto_scores, savefile=None):
    ch_type = 'grad'
    ch_subset = auto_scores['ch_types'] == ch_type
    ch_names = auto_scores['ch_names'][ch_subset]
    scores = auto_scores['scores_noisy'][ch_subset]
    limits = auto_scores['limits_noisy'][ch_subset]
    bins = auto_scores['bins']  # The the windows that were evaluated.
    # We will label each segment by its start and stop time, with up to 3
    # digits before and 3 digits after the decimal place (1 ms precision).
    bin_labels = [f'{start:3.3f} – {stop:3.3f}' for start, stop in bins]
    # We store the data in a Pandas DataFrame. The seaborn heatmap function
    # we will call below will then be able to automatically assign the correct
    # labels to all axes.
    data_to_plot = pd.DataFrame(data=scores,
                                columns=pd.Index(bin_labels, name='Time (s)'),
                                index=pd.Index(ch_names, name='Channel'))
    # First, plot the "raw" scores.
    fig, ax = pyplot.subplots(1, 2, figsize=(12, 8))
    fig.suptitle(f'Automated noisy channel detection: {ch_type}',
                 fontsize=16, fontweight='bold')
    sns.heatmap(data=data_to_plot, cmap='Reds', cbar_kws=dict(label='Score'),
                ax=ax[0])
    [ax[0].axvline(x, ls='dashed', lw=0.25, dashes=(25, 15), color='gray')
     for x in range(1, len(bins))]
    ax[0].set_title('All Scores', fontweight='bold')

    # Now, adjust the color range to highlight segments that exceeded the limit.
    sns.heatmap(data=data_to_plot,
                vmin=np.nanmin(limits),  # bads in input data have NaN limits
                cmap='Reds', cbar_kws=dict(label='Score'), ax=ax[1])
    [ax[1].axvline(x, ls='dashed', lw=0.25, dashes=(25, 15), color='gray')
     for x in range(1, len(bins))]
    ax[1].set_title('Scores > Limit', fontweight='bold')

    # The figure title should not overlap with the subplots.
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the figure in a subfolder 'Bad_Channels'
    pyplot.savefig(savefile)


# MAIN
proj_root = Path() / '..'
data_raw_dir = proj_root / 'data_raw'
data_bids_dir = proj_root / 'data_bids_test2'
data_deriv_dir = data_bids_dir / 'derivatives'

layout = BIDSLayout(data_bids_dir, validate=True)
subjects = layout.get_subjects()

template = os.path.join('sub-{subject}', 'ses-{session}', 'meg', 'sub-{subject}_ses-{session}_task-{task}')

for subject in subjects[:1]:  # test on 1 subj at first
    meg_files_subj = layout.get(subject=subject, task='vid*',
                                extension='fif', regex_search=True)  # 4 files per subj
    for meg_file_subj in meg_files_subj:
        session = meg_file_subj.get_entities()['session']
        task = meg_file_subj.get_entities()['task']
        print(f'sub-{subject}_session-{session}_task-{task} is calculating...')
        overwrite = True

        # read raw file and create derivatives folder
        raw = mne.io.read_raw_fif(meg_file_subj, preload=True)
        path_savefile = data_deriv_dir / template.format(subject=subject, session=session, task=task)
        path_savefile.parent.mkdir(parents=True, exist_ok=True)

        # prepare for preprocessing
        raw = set_annot_from_events(raw)  # make annotations
        raw = increase_raw_length(raw, t=30)  # add a chunk to avoid filtering problem
        # raw = decrease_raw_length(raw, events, t_before_event=30)  # drop useless part before the event starts

        # linear filtering
        raw.notch_filter(freqs=[50, 100])
        raw.filter(l_freq=0.3, h_freq=None)
        raw.save(str(path_savefile) + '_linear_filtering_meg.fif', overwrite=overwrite)

        # maxwell filtering
        # find bad channels
        crosstalk_file = layout.get(subject=subject, acquisition='crosstalk', extension='fif')[0].path
        fine_cal_file = layout.get(subject=subject, acquisition='calibration', extension='dat')[0].path
        noisy_chs, flat_chs, auto_scores = mne.preprocessing.find_bad_channels_maxwell(raw,
                                                                                       cross_talk=crosstalk_file,
                                                                                       calibration=fine_cal_file,
                                                                                       min_count=3,
                                                                                       return_scores=True)
        raw.info['bads'] = raw.info['bads'] + noisy_chs + flat_chs
        plot_maxwell_bad_ch(auto_scores, savefile=str(path_savefile) + '_bad_channels.png')
        # head position
        chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw)
        chpi_locs = mne.chpi.compute_chpi_locs(raw.info, chpi_amplitudes)
        head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs)
        mne.chpi.write_head_pos(str(path_savefile) + '_head_pos.pos', head_pos)
        # filtering
        raw = mne.preprocessing.maxwell_filter(raw, st_duration=10,
                                               cross_talk=crosstalk_file, calibration=fine_cal_file,
                                               head_pos=head_pos)
        raw.save(str(path_savefile) + '_maxwell_meg_tsss.fif', overwrite=overwrite)

        # downsample
        raw.resample(sfreq=250)  # do we need to include events matrix?
        raw.save(str(path_savefile) + '_downsampled.fif', overwrite=overwrite)

        # ECG/EOG artifacts removal
        # fit ica
        raw_hp_ica = raw.copy().filter(l_freq=None, h_freq=1)
        ica = mne.preprocessing.ICA(random_state=2, n_components=25)
        ica.fit(raw_hp_ica)
        ica.save(str(path_savefile) + '_ica.fif')
        # find all bad components iteratively
        ics = find_ics_iteratively(raw_hp_ica, ica, savefile=str(path_savefile) + '_ics.txt')
        # apply ica
        ica.apply(raw, exclude=ics)
        raw.save(str(path_savefile) + '_applied_ICA_meg.fif', overwrite=overwrite)

        # continue the pipeline ->

# TODO-MUST-HAVE-PREPROCESSING:
#   decrease_raw_length()
#   function to pick only video interval (?)
#   choose the best channels for ICA when there are no EOG chs (line 37)
#   solve the infinite problem with iterative ICA (why it goes to infinity?)
#   check the signal manually - topoplots looks very temporal (probably because of the noise)
#   construct kind of a report to spot noisy patients (html format?)
#   degrees of freedom for number of PCA components based on Kaisu Lankien

# TODO-NICE-TO-HAVE:
#   maxfilter: manuall bad_channels detection after automatic
#   fine calibration file and crosstalk compensation file?
#   Overwrite=False looks useless now
#   suppress warning about leading dot (.)
#   plotting functions for saving ICA and SSP pictures
#   make refactoring
#   *coordinates channels conversion for future MRI coreg (maxfiltering)
#   *Empty_room noise
#   *Correct events function
#   Filenames inside the function to make loop cleaner


# References:
# https://mne.tools/stable/auto_tutorials/preprocessing/plot_60_maxwell_filtering_sss.html#sphx-glr-auto-tutorials-preprocessing-plot-60-maxwell-filtering-sss-py
