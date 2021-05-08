import os
from pathlib import Path

import mne
from bids import BIDSLayout
from matplotlib import pyplot


# MAIN
proj_root = Path() / '..'
data_raw_dir = proj_root / 'data_raw'
data_bids_dir = proj_root / 'data_bids_test2'
data_deriv_dir = data_bids_dir / 'derivatives'

layout = BIDSLayout(data_bids_dir, validate=True)
subjects = layout.get_subjects()


# Summarize preprocessing procedure in a report
freq_before = pyplot.figure(1)
bad_ch = pyplot.figure(2)
mov_comp = pyplot.figure(3)

# Check how many plots have been generated (it varies depending on the No of ICs
fig_numbers = [x.num for x in pyplot._pylab_helpers.Gcf.get_all_fig_managers()]
ics = len(fig_numbers) - 5

for fig in range(1, 1 + ics):
    globals()['ic' + str(fig)] = pyplot.figure(fig + 3)  # because we have 3 other plots before ICA plots

freq_after = pyplot.figure(len(fig_numbers))
all_comps = pyplot.figure(len(fig_numbers) - 1)

path_report = data_deriv_dir / os.path.join('sub-' + str(subjects[i]) + '/', 'ses-' + str(sessions[i]) + '/',
                                            'meg/')

report = mne.Report(verbose=True)
report.parse_folder(path_report, pattern='*.fif', render_bem=False)
report.save(str(os.path.join(path_report, 'report.h5')), overwrite=True, open_browser=False)

with mne.open_report(str(os.path.join(path_report, 'report.h5'))) as report:
    report.add_figs_to_section(freq_before,
                               section='Power Spectrum Density',
                               captions='Before filtering',
                               replace=True)
    report.save(str(os.path.join(path_report, 'report.h5')), overwrite=True)

with mne.open_report(str(os.path.join(path_report, 'report.h5'))) as report:
    report.add_figs_to_section(freq_after,
                               section='Power Spectrum Density',
                               captions='After filtering',
                               replace=True)
    report.save(str(os.path.join(path_report, 'report.h5')), overwrite=True)

with mne.open_report(str(os.path.join(path_report, 'report.h5'))) as report:
    report.add_figs_to_section(bad_ch,
                               section='Bad Channels',
                               captions='Automated bad channel detection',
                               replace=True)
    report.save(str(os.path.join(path_report, 'report.h5')), overwrite=True)

with mne.open_report(str(os.path.join(path_report, 'report.h5'))) as report:
    report.add_figs_to_section(all_comps,
                               section='ICA',
                               captions='All components',
                               replace=True)
    report.save(str(os.path.join(path_report, 'report.h5')), overwrite=True)

for fig in range(1, ics + 1):
    with mne.open_report(str(os.path.join(path_report, 'report.h5'))) as report:
        report.add_figs_to_section(globals()['ic' + str(fig)],
                                   section='ICA',
                                   captions='Artifactual Component number' + str(fig),
                                   replace=True)
        report.save(str(os.path.join(path_report, 'report.h5')), overwrite=True)

with mne.open_report(str(os.path.join(path_report, 'report.h5'))) as report:
    report.add_figs_to_section(mov_comp,
                               section='Movement compensation',
                               captions='Movement compensation',
                               replace=True)
    report.save(str(os.path.join(path_report, 'report.html')), overwrite=True)
