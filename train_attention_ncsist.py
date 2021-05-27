from ANT_dataset_loader import DatasetLoader

loader = DatasetLoader()
data_trials = loader.load_data(data_type="session", feature_type="time")

for nth_trial, data_trial in enumerate(data_trials):
    a = 0

from statistics import median
