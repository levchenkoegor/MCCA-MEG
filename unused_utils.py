# SSP
def fit_ssp_eog(raw, eog_chs, savefile=None, verbose=False):
    eog_projs_source1, eog_events_source1 = mne.preprocessing.compute_proj_eog(raw, n_grad=1, n_mag=1, n_eeg=0,
                                                                               no_proj=True, ch_name=eog_chs[0],
                                                                               event_id=990, verbose=verbose)
    eog_projs_source2, eog_events_source2 = mne.preprocessing.compute_proj_eog(raw, n_grad=1, n_mag=1, n_eeg=0,
                                                                               no_proj=True, ch_name=eog_chs[1],
                                                                               event_id=991, verbose=verbose)
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
