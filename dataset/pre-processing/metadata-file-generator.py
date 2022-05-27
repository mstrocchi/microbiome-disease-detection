import pandas as pd


DATA_PATH = '/Volumes/2BIG/RP/dataset/'
RAW_DATA_PATH = DATA_PATH + 'raw_data/'
METADATA_PATH = RAW_DATA_PATH + 'metadata/'


if __name__ == '__main__':

    mapping = pd.read_csv(METADATA_PATH + 'mapping_file.txt', sep="\t")
    metadata = pd.read_csv(METADATA_PATH + 'sample_info.txt', sep="\t")
    merged = metadata.merge(mapping, on='sample_name')
    merged.to_csv(METADATA_PATH + 'metadata.csv')
