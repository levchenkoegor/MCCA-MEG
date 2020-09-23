import os
from pathlib import Path

import mne
import mne_bids
import pandas as pd

proj_root = Path() / '..'
data_raw_dir = proj_root / 'data_raw'
data_bids_dir = proj_root / 'data_bids'

subj_fullnames = [os.listdir(directory) for directory in [data_raw_dir / 'Group1', data_raw_dir / 'Group2']]

subj_bids_codes = {name: bids_id+1001 for bids_id, name in enumerate(subj_fullnames[0])}
subj_bids_codes.update({name: bids_id+2001 for bids_id, name in enumerate(subj_fullnames[1])})

raw_files_paths = list(data_raw_dir.glob('**/**/**/*.fif'))
raw_files_paths = [raw_file_path for raw_file_path in raw_files_paths if 'process' not in str(raw_file_path)]

for raw_file_path in raw_files_paths:
    raw_file = mne.io.read_raw_fif(str(raw_file_path))

    subj_filename = str(raw_file_path).split('\\')[-1].split('_')
    subj_fullname = '_'.join(subj_filename[:2])

    if subj_fullname in subj_bids_codes.keys():
        subj_id = subj_bids_codes[subj_fullname]
        task_name = subj_filename[-1].split('.')[0] # vid%
    else:
        subj_id = 'emptyroom'
        task_name = 'noise'

    date_record = str(raw_file.info['meas_date']).split(' ')[0].replace('-', '')
    bids_filename = mne_bids.make_bids_basename(subject=subj_id,
                                                session=date_record,
                                                task=task_name)
    try:
        mne_bids.write_raw_bids(raw=raw_file, bids_basename=bids_filename, bids_root=data_bids_dir, verbose=False,
                                anonymize={'daysback': 40000, 'keep_his': False}, overwrite=True)
    except AttributeError:
        continue

df_subj_bids_codes = pd.DataFrame.from_dict(subj_bids_codes, orient='index')
df_subj_bids_codes.to_csv(data_raw_dir / 'BIDS_subjects_codes.csv')

# Add logging



