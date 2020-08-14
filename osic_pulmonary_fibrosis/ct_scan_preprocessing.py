import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pydicom
import seaborn as sns
from skimage import exposure


###################
# CT scans analysis
###################


def load_ct_scans_paths(base_dir):
    scan_paths = {}
    ct_scan_subdirs = os.listdir(base_dir)
    print(f'training set size: {len(ct_scan_subdirs)}')
    num_of_scans = []
    for ct_scan_subdir in ct_scan_subdirs:
        scans_per_patient = os.listdir(f'{base_dir}/{ct_scan_subdir}')
        scans_per_patient = sorted(scans_per_patient, key=lambda x: int(x.split('.')[0]))
        scan_paths[ct_scan_subdir] = scans_per_patient
        num_of_scans.append(len(scans_per_patient))
    plt.figure()
    plt.xlabel('Number of scans')
    plt.ylabel('Patient count')
    sns.distplot(num_of_scans, kde=False, bins=50)
    return scan_paths


def even_select(array, m):
    """
    select m elements from array, trying to pick them as evenly as possible
    :param array: inputs array
    :param m: number of elements to select, must by < array.shape[0]
    :return: sub-array
    """
    n = array.shape[0]
    assert m <= n
    if n == m:
        return array
    elif m > n / 2:
        cut = np.ones(n, dtype=int)
        q, r = divmod(n, n - m)
        indices = [q * i + min(i, r) for i in range(n - m)]
        cut[indices] = 0
    else:
        cut = np.zeros(n, dtype=int)
        q, r = divmod(n, m)
        indices = [q * i + min(i, r) for i in range(m)]
        cut[indices] = 1
    return array[cut == 1]


def load_and_normalize_ct_scans(training_data_dir, scan_paths):
    number_of_scans_per_patient = 12
    start = time()
    scan_sizes_list = []
    scans_per_patient = {}
    i = 0
    ct_volumes = []

    # visualize scans of first 10 patients
    fig, axs = plt.subplots(10, number_of_scans_per_patient, figsize=(2 * number_of_scans_per_patient, 2 * 10))

    for patient_id in scan_paths:
        scans = []
        # output_patient_dir = f'/kaggle/working/{patient_dir}'
        # os.makedirs(output_patient_dir, exist_ok=True)
        scans_array = np.array(scan_paths[patient_id])
        print(f'{i}) {patient_id}')
        j = 0
        for scan_file in even_select(scans_array, number_of_scans_per_patient):
            dicom = pydicom.dcmread(f'{training_data_dir}/{patient_id}/{scan_file}')
            pixels = dicom.pixel_array
            scans.append(pixels)
            # np.save(f'{output_patient_dir}/{scan_file}.npy', pixels)
            scan_sizes_list.append(pixels.shape)
            # axs[j].title(dicom['PatientID']._value +': ' + dicom['BodyPartExamined']._value)
            j += 1
        scans_per_patient[patient_id] = scans

        scans_3d = np.stack(scans, axis=0)
        scans_3d = exposure.equalize_hist(scans_3d)
        # scans_3d = (scans_3d - np.mean(scans_3d)) / np.std(scans_3d)
        # print(f'mean: {np.mean(scans_3d)}, min: {np.min(scans_3d)}, max: {np.max(scans_3d)}')
        # convert to 0-255 int scale
        scans_3d += np.abs(np.min(scans_3d))
        scans_3d = (((scans_3d) / np.max(scans_3d)) * 255).astype('uint8')
        # print(f'mean: {np.mean(scans_3d)}, min: {np.min(scans_3d)}, max: {np.max(scans_3d)}')
        if i < 10:
            for j in range(number_of_scans_per_patient):
                axs[i, j].imshow(scans_3d[j], cmap="plasma")
        elif i == 10:
            plt.show()
        ct_volumes.append(scans_3d)
        i += 1
    print(f'time = {time() - start}')
    return ct_volumes


def main():
    training_data_dir = '../input/osic-pulmonary-fibrosis-progression/train'
    scan_paths = load_ct_scans_paths(training_data_dir)
    ct_volumes = load_and_normalize_ct_scans(training_data_dir, scan_paths)


if __name__ == '__main__':
    main()
else:
    main()
