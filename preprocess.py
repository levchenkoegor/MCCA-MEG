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


# Basic preprocessing
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


def find_ics(raw, ica, eog_chs=('MEG0521', 'MEG0921'), verbose=False):
    eog_chs_raw = find_chs(raw, ch_type='EOG')
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
    while new_ics and i < 3:
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


def apply_ica_proj(raw, ica, ics, savefile=None, overwrite=True):
    ica.apply(raw, exclude=ics)
    if savefile:
        raw.save(savefile, overwrite=overwrite)
    return raw


def ica_routine(raw):
    ica = fit_ica(raw, h_freq=1, savefile=str(path_savefile) + '_ica.fif')
    ics = find_ics_iteratively(raw, ica, savefile=str(path_savefile) + '_ics.txt')
    raw = apply_ica_proj(raw, ica, ics, savefile=str(path_savefile) + '_applied_ICA_meg.fif')
    return raw


# def plot_ica():
    # something


# SSP
def find_chs(raw, ch_type=None):
    chs_names = [ch_name for ch_name in raw.info['ch_names'] if ch_type in ch_name]
    return chs_names


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


# def plot_ssp(raw):
    # something


# Maxwell filtering
def find_bad_ch_maxwell(raw, visualization=False, savefile=None, crosstalk_file=None, fine_cal_file=None):
    noisy_chs, flat_chs, auto_scores = mne.preprocessing.find_bad_channels_maxwell(raw,
                                                                                   cross_talk=crosstalk_file,
                                                                                   calibration=fine_cal_file,
                                                                                   min_count=3,
                                                                                   return_scores=True,
                                                                                   verbose=False)
    raw.info['bads'] = raw.info['bads'] + noisy_chs + flat_chs

    if visualization:
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

    return raw


def maxwell_filtering(raw, st_duration=10, head_pos=None, savefile=None, overwrite=True, crosstalk_file=None, fine_cal_file=None, verbose=False):
    raw_tsss = mne.preprocessing.maxwell_filter(raw, st_duration=st_duration,
                                                cross_talk=crosstalk_file, calibration=fine_cal_file,
                                                head_pos=head_pos, verbose=verbose)
    if savefile:
        raw_tsss.save(savefile, overwrite=overwrite)
    return raw_tsss


def chpi_find_head_pos(raw, savefile=None, verbose=False):
    chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw, verbose=verbose)
    chpi_locs = mne.chpi.compute_chpi_locs(raw.info, chpi_amplitudes, verbose=verbose)
    head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs, verbose=verbose)
    #mne.viz.plot_head_positions(head_pos, mode='traces')
    if savefile:
        mne.chpi.write_head_pos(savefile, head_pos)
    return head_pos


# MAIN
proj_root = Path() / '..'
data_raw_dir = proj_root / 'data_raw'
data_bids_dir = proj_root / 'data_bids_test2'
data_deriv_dir = data_bids_dir / 'derivatives'

layout = BIDSLayout(data_bids_dir, validate=True)
subjects = layout.get_subjects()

template = os.path.join('sub-{subject}', 'ses-{session}', 'meg', 'sub-{subject}_ses-{session}_task-{task}')

for subject in subjects[:1]:  # test on 1 subj at first
    # 4 files per subj
    meg_files_subj = layout.get(subject=subject, task='vid*', extension='fif', regex_search=True)
    for meg_file_subj in meg_files_subj:
        session = meg_file_subj.get_entities()['session']
        task = meg_file_subj.get_entities()['task']
        print(f'sub-{subject}_session-{session}_task-{task} is calculating...')

        raw = mne.io.read_raw_fif(meg_file_subj, preload=True, verbose=False)
        path_savefile = data_deriv_dir / template.format(subject=subject, session=session, task=task)
        path_savefile.parent.mkdir(parents=True, exist_ok=True)

        # Prepare for preprocessing: make annotations, add a chunk to avoid filtering problem and drop
        # too long part before event starts
        raw = set_annot_from_events(raw)
        raw = increase_raw_length(raw, t=30)
        # raw = decrease_raw_length(raw, events, t_before_event=30)

        # Basic preprocessing
        raw_filtered = linear_filtering(raw, notch=[50, 100], l_freq=0.3,
                                        savefile=str(path_savefile) + '_linear_filtering_meg.fif')

        # Maxwell filtering
        crosstalk_file = layout.get(subject=subject, acquisition='crosstalk', extension='fif')
        fine_cal_file = layout.get(subject=subject, acquisition='calibration', extension='fif')
        raw = find_bad_ch_maxwell(raw_filtered, visualization=False, crosstalk_file=None, fine_cal_file=None, savefile=str(path_savefile) + '_bad_channels.png')
        head_pos = chpi_find_head_pos(raw, savefile=str(path_savefile) + '_head_pos.pos')
        raw = maxwell_filtering(raw, st_duration=30, head_pos=head_pos, crosstalk_file=None, fine_cal_file=None,
                                savefile=str(path_savefile) + '_maxwell_meg_tsss.fif')

        # Downsample
        raw = downsample(raw, target_fs=250)

        # ECG/EOG artifacts removal
        raw = ica_routine(raw)

        # continue the pipeline ->


# TODO-MUST-HAVE-PREPROCESSING:
#   decrease_raw_length()
#   function to pick only video interval (?)
#   choose the best channels for ICA when there are no EOG chs (line 66)
#   solve the infinite problem with iterative ICA (why it goes to infinity?)
#   check the signal manually - topoplots looks very temporal (probably because of the noise)
#   construct kind of a report to spot noisy patients (html format?)
#   degrees of freedom for number of PCA components based on Kaisu Lankien

# TODO-NICE-TO-HAVE:
#   maxfilter: manuall bad_channels detection after automatic
#   fine calibration file and crosstalk compensation file?
#   Overwrite=False looks useles now
#   suppress warning about leading dot (.)
#   plotting functions for saving ICA and SSP pictures
#   make refactoring
#   *coordinates channels conversion for future MRI coreg (maxfiltering)
#   *Empty_room noise
#   *Correct events function
#   Filenames inside the function to make loop cleaner


# References:
# https://mne.tools/stable/auto_tutorials/preprocessing/plot_60_maxwell_filtering_sss.html#sphx-glr-auto-tutorials-preprocessing-plot-60-maxwell-filtering-sss-py
