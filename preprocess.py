import os

path = "Y:\\LevchenkoE\\ISC\\"
os.chdir(path)

paths = [path+'\\'+f for (path, folders, files) in os.walk('Experiment3_videos_MEG\\raw_data\\')
         if 'Group' in path
         for f in files if f.endswith('tsss_mc_trans.fif')]

paths

