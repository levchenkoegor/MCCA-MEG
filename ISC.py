from pathlib import Path
from timeit import default_timer

import mne
import numpy as np
from scipy.linalg import eigh


def train_cca(data):
    """Run Correlated Component Analysis on your training data.

        Parameters:
        ----------
        data : dict
            Dictionary with keys are names of conditions and values are numpy
            arrays structured like (subjects, channels, samples).
            The number of channels must be the same between all conditions!

        Returns:
        -------
        W : np.array
            Columns are spatial filters. They are sorted in descending order, it means that first column-vector maximize
            correlation the most.
        ISC : np.array
            Inter-subject correlation sorted in descending order

    """

    start = default_timer()

    C = len(data.keys())
    print(f'train_cca - calculations started. There are {C} conditions: {data.keys()[0]}')

    gamma = 0.1
    Rw, Rb = 0, 0
    for cond in data.values():
        N, D, T, = cond.shape
        print(f'Condition has {N} subjects, {D} sensors and {T} samples')
        cond = cond.reshape(D * N, T)

        # Rij
        Rij = np.swapaxes(np.reshape(np.cov(cond), (N, D, N, D)), 1, 2)

        # Rw
        Rw = Rw + np.mean([Rij[i, i, :, :]
                           for i in range(0, N)], axis=0)

        # Rb
        Rb = Rb + np.mean([Rij[i, j, :, :]
                           for i in range(0, N)
                           for j in range(0, N) if i != j], axis=0)

    # Divide by number of condition
    Rw, Rb = Rw/C, Rb/C

    # Regularization
    Rw_reg = (1 - gamma) * Rw + gamma * np.mean(eigh(Rw)[0]) * np.identity(Rw.shape[0])

    # ISCs and Ws
    [ISC, W] = eigh(Rb, Rw_reg)

    # Make descending order
    ISC, W = ISC[::-1], W[:, ::-1]

    stop = default_timer()

    print(f'Elapsed time: {round(stop - start)} seconds.')
    return W, ISC


def apply_cca(X, W, fs):
    """Applying precomputed spatial filters to your data.

        Parameters:
        ----------
        X : ndarray
            3-D numpy array structured like (subject, channel, sample)
        W : ndarray
            Spatial filters.
        fs : int
            Frequency sampling.
        Returns:
        -------
        ISC : ndarray
            Inter-subject correlations values are sorted in descending order.
        ISC_persecond : ndarray
            Inter-subject correlations values per second where first row is the most correlated.
        ISC_bysubject : ndarray
            Description goes here.
        A : ndarray
            Scalp projections of ISC.
    """

    start = default_timer()
    print('apply_cca - calculations started')

    N, D, T = X.shape
    # gamma = 0.1
    window_sec = 5
    X = X.reshape(D * N, T)

    # Rij
    Rij = np.swapaxes(np.reshape(np.cov(X), (N, D, N, D)), 1, 2)

    # Rw
    Rw = np.mean([Rij[i, i, :, :]
                  for i in range(0, N)], axis=0)
    # Rw_reg = (1 - gamma) * Rw + gamma * np.mean(eigh(Rw)[0]) * np.identity(Rw.shape[0])

    # Rb
    Rb = np.mean([Rij[i, j, :, :]
                  for i in range(0, N)
                  for j in range(0, N) if i != j], axis=0)

    # ISCs
    ISC = np.sort(np.diag(np.transpose(W) @ Rb @ W) / np.diag(np.transpose(W) @ Rw @ W))[::-1]

    # Scalp projections
    A = np.linalg.solve(Rw @ W, np.transpose(W) @ Rw @ W)

    # ISC by subject
    print('by subject is calculating')
    ISC_bysubject = np.empty((D, N))

    for subj_k in range(0, N):
        Rw, Rb = 0, 0
        Rw = np.mean([Rw + 1 / (N - 1) * (Rij[subj_k, subj_k, :, :] + Rij[subj_l, subj_l, :, :])
                      for subj_l in range(0, N) if subj_k != subj_l], axis=0)
        Rb = np.mean([Rb + 1 / (N - 1) * (Rij[subj_k, subj_l, :, :] + Rij[subj_l, subj_k, :, :])
                      for subj_l in range(0, N) if subj_k != subj_l], axis=0)

        ISC_bysubject[:, subj_k] = np.diag(np.transpose(W) @ Rb @ W) / np.diag(np.transpose(W) @ Rw @ W)

    # ISC per second
    print('by persecond is calculating')
    ISC_persecond = np.empty((D, int(T / fs) + 1))
    window_i = 0

    for t in range(0, T, fs):

        Xt = X[:, t:t+window_sec*fs]
        Rij = np.cov(Xt)
        Rw = np.mean([Rij[i:i + D, i:i + D]
                      for i in range(0, D * N, D)], axis=0)
        Rb = np.mean([Rij[i:i + D, j:j + D]
                      for i in range(0, D * N, D)
                      for j in range(0, D * N, D) if i != j], axis=0)

        ISC_persecond[:, window_i] = np.diag(np.transpose(W) @ Rb @ W) / np.diag(np.transpose(W) @ Rw @ W)
        window_i += 1

    stop = default_timer()
    print(f'Elapsed time: {round(stop - start)} seconds.')

    return ISC, ISC_persecond, ISC_bysubject, A


# MAIN
vid_dur = {'vid1': 380, 'vid2': 290, 'vid3': 381, 'vid4': 372}

proj_root = Path() / '..'
data_raw_dir = proj_root / 'data_raw'
data_bids_dir = proj_root / 'data_bids'
data_deriv_dir = data_bids_dir / 'derivatives'

vid_i = 2
raw_files_paths = [raw_file_path for raw_file_path in data_deriv_dir.glob(f'*/*/*/*vid{vid_i}_applied_ICA_meg.fif')]

# Load
gr1_raws = [mne.io.read_raw_fif(raw_file_path, preload=True) for raw_file_path in raw_files_paths[:21]]
gr2_raws = [mne.io.read_raw_fif(raw_file_path, preload=True) for raw_file_path in raw_files_paths[21:] if '2007' not in str(raw_file_path)]

# Pick only vid2 epoch and only MEG channels
gr1_vid2_epochs = []
for gr1_raw in gr1_raws:
    events = mne.find_events(gr1_raw, stim_channel='STI101', verbose=False)
    epoch = mne.Epochs(gr1_raw, events, event_id=vid_i, tmin=0, tmax=vid_dur['vid2'], baseline=None, picks=['grad'])

    gr1_vid2_epochs.append(np.squeeze(epoch.get_data()))

gr2_vid2_epochs = []
for gr2_raw in gr2_raws:
    events = mne.find_events(gr2_raw, stim_channel='STI101', verbose=False)
    epoch = mne.Epochs(gr2_raw, events, event_id=vid_i, tmin=0, tmax=vid_dur['vid2'], baseline=None, picks=['grad'])

    gr2_vid2_epochs.append(np.squeeze(epoch.get_data()))

gr1_data_3d = np.stack(gr1_vid2_epochs)
gr2_data_3d = np.stack(gr2_vid2_epochs)

X_both_cat = dict(Gr1_2=np.concatenate((gr1_data_3d[:, :, :43750], gr2_data_3d[:, :, :43750]), axis=0))
X_both_sep = dict(Gr1=gr1_data_3d[:, :, :43750], Gr2=gr2_data_3d[:, :, :43750])

np.save(str(proj_root / 'test' / 'gr1_data_3d'), gr1_data_3d)
np.save(str(proj_root / 'test' / 'gr2_data_3d'), gr2_data_3d)
np.save(str(proj_root / 'test' / 'X_both_cat'), X_both_cat)
np.save(str(proj_root / 'test' / 'X_both_sep'), X_both_sep)

isc_results = dict()
[W, ISC_overall] = train_cca(X_both_cat)

for name, group in X_both_sep.items():
    print(name)
    print(group.shape)
    print('Seconds: ', group.shape[2]/250)
    isc_results[str(name)] = dict(zip(['ISC', 'ISC_persecond', 'ISC_bysubject', 'A'], apply_cca(group, W, 250)))
    print('Apply over')

np.save(str(proj_root / 'test' / 'isc_results'), isc_results)


# TODO-MUST-HAVE:
## Check if get_data() return channels in the same order between patients
## Refactor loops and lists - too many repeats of myself
## PCA before CCA?
## Save in BIDS-validated format output files
## What type of channels should I pick?
## Check video duration
