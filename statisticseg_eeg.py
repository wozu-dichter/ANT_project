import numpy as np
from ANT_dataset_loader import DatasetLoader
import xlwt

loader = DatasetLoader()
loader.rest_signal_len = 300
loader.apply_signal_normalization = False
loader.apply_bandpass_filter = False
subjects_trials_data, _ = loader.load_data(data_type="rest", feature_type="time",
                                           fatigue_basis="by_feedback")
eeg_channel = ["Fp1", "Fp2", "F3", "Fz", "F4", "T7", "C3", "Cz",
               "C4", "T8", "P3", "Pz", "P4", "P7", "P8", "Oz",
               "AF3", "AF4", "F7", "F8", "FT7", "FC3", "FCz", "FC4",
               "FT8", "TP7", "CP3", "CPz", "CP4", "TP8", "O1", "O2"]

subject_ids = loader.get_subject_ids()

print(loader.apply_signal_normalization)

workbook = xlwt.Workbook(encoding='ascii')
alignment = xlwt.Alignment()  # Create Alignment
alignment.horz = xlwt.Alignment.HORZ_CENTER  # May be: HORZ_GENERAL, HORZ_LEFT, HORZ_CENTER, HORZ_RIGHT, HORZ_FILLED, HORZ_JUSTIFIED, HORZ_CENTER_ACROSS_SEL, HORZ_DISTRIBUTED
alignment.vert = xlwt.Alignment.VERT_CENTER  # May be: VERT_TOP, VERT_CENTER, VERT_BOTTOM, VERT_JUSTIFIED, VERT_DISTRIBUTED
style = xlwt.XFStyle()  # Create Style
style.alignment = alignment  # Add Alignment to Style

for key in subject_ids:
    worksheet = workbook.add_sheet(key)
    worksheet.write(0, 1, 'mean', style)  #
    worksheet.write(0, 2, 'amplitude', style)  #
    worksheet.write(0, 3, 'standard', style)  #
    worksheet.write(0, 4, 'Variance', style)

    for eeg_array in subjects_trials_data[key]:
        array = eeg_array['eeg']
        if eeg_array['fatigue_level'] == 'high':
            worksheet.write(0, 0, 'fatigue=high', style)
            worksheet.col(0).width = 3333
            print("key:{:s}, fatigue={:s}".format(key, str(eeg_array['fatigue_level'])))
            for i_channel_array in range(array.shape[-1]):
                worksheet.write(i_channel_array + 1, 0, eeg_channel[i_channel_array], style)
                channel_array = array[:, i_channel_array]
                amp = np.ptp(channel_array)
                mean = np.mean(channel_array)
                std = np.std(channel_array)
                var = np.var(channel_array)
                # print('channel:{:s}, mean={:.3f}, amplitude={:.3f}, standard={:.3f}'.format(
                #     eeg_channel[i_channel_array], mean, amp, std))
                worksheet.write(i_channel_array + 1, 1, str(mean), style)
                worksheet.write(i_channel_array + 1, 2, str(amp), style)
                worksheet.write(i_channel_array + 1, 3, str(std), style)
                worksheet.write(i_channel_array + 1, 4, str(var), style)
workbook.save('eeg_statist.xls')  # 儲存檔案
a = 0
