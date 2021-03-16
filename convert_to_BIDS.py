import os
import time
#import sys
from pathlib import Path

import mne
import mne_bids
import pandas as pd
#import numpy as np


t = time.process_time()  # measure time of script execution

proj_root = Path() / '..'
data_raw_dir = proj_root / 'data_raw'
data_bids_dir = proj_root / 'data_bids_test2'

#sys.stdout = open(data_raw_dir / 'log_convert-to-BIDS', 'w')

subj_fullnames = [os.listdir(directory) for directory in [data_raw_dir / 'Group1', data_raw_dir / 'Group2']]

sub_id_name = {name: bids_id + 1001 for bids_id, name in enumerate(subj_fullnames[0])}  # group1
sub_id_name.update({name: bids_id + 2001 for bids_id, name in enumerate(subj_fullnames[1])})  # group2

raw_files_paths = list(data_raw_dir.glob('**/**/**/*.fif'))

# mri_paths = np.concatenate([list(data_raw_dir.glob('../MRI_scans/**/NIFTI/'+reg_exp)) for reg_exp in
#                             ['*_sT1W_3D_*.nii', '*t1_*_sag_*iso.nii',  '*T1_Cube.nii',
#                              '*FSPGR.nii', '*t1_tse_sag_3mm.nii']])
# mri_subjnames = [mri_path.parts[-3] for mri_path in mri_paths]

for raw_file_path in raw_files_paths[20:]:
    raw_meg = mne.io.read_raw_fif(raw_file_path)

    subj_fullname = raw_file_path.parts[-3]  # subject fullname or 'empty_room'

    if subj_fullname in sub_id_name.keys():
        subj_id = str(sub_id_name[subj_fullname])
        task_name = raw_file_path.parts[-1].split('.')[0][-4:]  # vid%
    else:
        subj_id = 'emptyroom'
        task_name = 'noise'

    date_record = str(raw_meg.info['meas_date']).split(' ')[0].replace('-', '')
    meg_bids_path = mne_bids.BIDSPath(subject=subj_id, session=date_record, task=task_name,
                                      root=data_bids_dir)

    # mri_bids_path = mne_bids.BIDSPath(subject=subj_id, session=date_record,
    #                                   root=data_bids_dir)
    #

    mne_bids.write_raw_bids(raw=raw_meg, bids_path=meg_bids_path,
                            anonymize={'daysback': 40000}, overwrite=True, verbose=True)

    # mri_presence = mri_subjnames.count(subj_fullname)
    # if mri_presence:
    #     mri_path_i = mri_subjnames.index(subj_fullname)
    #     mne_bids.write_anat(t1w=mri_paths[mri_path_i], bids_path=mri_bids_path, raw=raw_meg, overwrite=True)
    #     print(f'{subj_fullname} has MRI file. Path to MRI file: {mri_paths[mri_path_i]}')

df_subj_bids_codes = pd.DataFrame.from_dict(sub_id_name, orient='index')
df_subj_bids_codes.to_csv(data_raw_dir / 'BIDS_subjects_codes.csv')

elapsed_time = time.process_time() - t
print(f'Elapsed time - {elapsed_time}')
#sys.stdout.close()

# TODO:
#   MRI data: from DICOM to NIFTI (using dcm2niix) to BIDS
#   Make code more readable (example, line 35, 39)
#   Validate the whole dataset (problem with `split` suffix - issues/731
