import numpy as np
import os
from ANT_dataset_loader import DatasetLoader, glob_sorted, load_npy
import collections


for subject_dir in glob_sorted('./gaming_data/*'):
    subject_id = os.path.basename(subject_dir)
    for record_dir in glob_sorted(subject_dir + "/*"):
        raw_data = load_npy(record_dir)
        eeg_data = raw_data['eeg']
        stageRecord = raw_data['stageRecord']

        print(collections.Counter(stageRecord))

        a=0
