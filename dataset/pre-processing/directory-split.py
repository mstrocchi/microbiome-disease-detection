import os
import shutil
import pandas as pd
from glob import glob
from tqdm import tqdm
from typing import Optional


DATA_PATH = '/Volumes/2BIG/RP/dataset/'
RAW_DATA_PATH = DATA_PATH + 'raw_data/'
SAMPLES_PATH = RAW_DATA_PATH + 'samples/'
METADATA_PATH = RAW_DATA_PATH + 'metadata/metadata.csv'
SAVE_PATH = DATA_PATH + 'processed/'


def lookup_class(metadata_df: pd.DataFrame, run_prefix: str) -> Optional[str]:
    """ Returns the label of a given sample. """
    matches = metadata_df[metadata_df['run_prefix'] == run_prefix]
    if len(matches['sample_name']) != 1:
        return None
    return list(matches['diagnosis'])[0].lower()


def print_stats(metadata_df):
    """ Prints some stats about the dataset. """
    # control, no disease, inflammatory colitis, ulcerative colitis, chron disease
    print("Original stats")
    diagnosis = ['control', 'no', 'IC', 'UC', 'CD']
    for disease in diagnosis:
        string = f"\'{disease}\'".ljust(10)
        print(f"Class {string} from metadata: {len(metadata_df[metadata_df['diagnosis'] == disease])}")


def get_run_prefix(file_path):
    """ Returns the run prefix of a sample. """
    file_name = os.path.basename(file_path)
    return file_name[4:-9]


if __name__ == '__main__':

    metadata = pd.read_csv(METADATA_PATH)

    actual_stats = {
        'no': 0,
        'cd': 0
    }

    for file_path in tqdm(glob(SAMPLES_PATH + '*.gz')):
        label = lookup_class(metadata, get_run_prefix(file_path))
        if label and label in actual_stats.keys():
            save_path = SAVE_PATH + f'{label}/{os.path.basename(file_path)}'
            shutil.copyfile(file_path, save_path)
            actual_stats[label] += 1

    print_stats(metadata)
    print(f"Actual stats: \n{actual_stats}")
