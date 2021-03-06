import json
import csv
import os
import time
from pathlib import Path

import mne
import mne_bids
import pandas as pd

t = time.process_time()  # measure time of script execution

proj_root = Path() / '..'
data_raw_dir = proj_root / 'data_raw'
data_bids_dir = proj_root / 'data_bids_test2'
data_bids_dir.mkdir(exist_ok=True)

#sys.stdout = open(data_raw_dir / 'log_convert-to-BIDS', 'w')

subj_fullnames = [os.listdir(directory) for directory in [data_raw_dir / 'Group1', data_raw_dir / 'Group2']]

sub_id_name = {name: bids_id + 1001 for bids_id, name in enumerate(subj_fullnames[0])}  # group1
sub_id_name.update({name: bids_id + 2001 for bids_id, name in enumerate(subj_fullnames[1])})  # group2

raw_files_paths = sorted(list(data_raw_dir.glob('**/**/**/*.fif')))[1:]  # without cross-talk file

ct_file = data_raw_dir / 'ct_fc_files' / 'ct_sparse.fif'
fc_files = {'Before_sept2019': data_raw_dir / 'ct_fc_files' / 'sss_cal_before_sept2019.dat',
            'Current': data_raw_dir / 'ct_fc_files' / 'sss_cal.dat'}

# mri_paths = np.concatenate([list(data_raw_dir.glob('../MRI_scans/**/NIFTI/'+reg_exp)) for reg_exp in
#                             ['*_sT1W_3D_*.nii', '*t1_*_sag_*iso.nii',  '*T1_Cube.nii',
#                              '*FSPGR.nii', '*t1_tse_sag_3mm.nii']])
# mri_subjnames = [mri_path.parts[-3] for mri_path in mri_paths]

for raw_file_path in raw_files_paths:
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

    if subj_id != 'emptyroom':
        mne_bids.write_meg_crosstalk(ct_file, meg_bids_path)
        if int(date_record[:4]) < 2020:
            mne_bids.write_meg_calibration(fc_files['Before_sept2019'], meg_bids_path)
        else:
            mne_bids.write_meg_calibration(fc_files['Current'], meg_bids_path)

    # mri_presence = mri_subjnames.count(subj_fullname)
    # if mri_presence:
    #     mri_path_i = mri_subjnames.index(subj_fullname)
    #     mne_bids.write_anat(t1w=mri_paths[mri_path_i], bids_path=mri_bids_path, raw=raw_meg, overwrite=True)
    #     print(f'{subj_fullname} has MRI file. Path to MRI file: {mri_paths[mri_path_i]}')

# save fullnames and bids_id
df_subj_bids_codes = pd.DataFrame.from_dict(sub_id_name, orient='index').reset_index()
df_subj_bids_codes.columns = ['Fullname', 'BIDS_id']
df_subj_bids_codes.to_csv(data_raw_dir / 'Fullnames_BIDS-ids_subjects.csv', index=False)

# convert questionnaire responses
data_bids_phenotype = data_bids_dir / 'phenotype'
data_bids_phenotype.mkdir(exist_ok=True)

questionnaire = pd.read_excel(data_raw_dir / 'questionnaire_responses_raw.xls',
                              sheet_name='Answers_cleaned', usecols='A:X')
descriptions = pd.read_excel(data_raw_dir / 'questionnaire_responses_raw.xls',
                             sheet_name='Columns')

questionnaire.replace({"Subject name record": sub_id_name}, inplace=True)  # change fullnames to bids_id
questionnaire.rename(columns={'Subject name record': 'participant_id'}, inplace=True)
questionnaire.rename(columns=descriptions.set_index('Description').to_dict()['Question'], inplace=True)

meta_json = {}
for _, row in descriptions.iterrows():
    meta_json[row['Question']] = {'Description': row['Description'],
                                  'Levels': row['Levels']}

with open(data_bids_phenotype / 'questionnaire.json', 'w', encoding='utf-8') as que_json:
    json.dump(meta_json, que_json, indent=4, ensure_ascii=False)

questionnaire.to_csv(data_bids_phenotype / 'questionnaire.tsv', sep='\t', na_rep='n/a',
                     index=False, encoding='utf-8', quoting=csv.QUOTE_NONE)

# save levels properly
# check encoding of json files
# filter out answers in group 2 for BYD (no text before the movie)
# save tsv file with data with proper encoding

elapsed_time = time.process_time() - t
print(f'Elapsed time - {elapsed_time}')
#sys.stdout.close()

# TODO:
#   Add behavioural data
#   Add stimuli data
#   Validate dataset
#   MRI data: from DICOM to NIFTI (using dcm2niix) to BIDS
#   Validate the whole dataset with MRI scans
#   Make code more readable (example, line 35, 39)
#   Save log file (probably to derivatives?)
#   Should I add cross-talk and fine calibrations files to 'emptyroom'?
