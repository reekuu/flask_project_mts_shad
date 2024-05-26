import warnings

import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)

# Define columns used in the model
MODEL_FEATURES = [
    'частота_пополнения', 'доход', 'объем_данных',
    'on_net', 'продукт_1', 'секретный_скор',
]


def import_data(path_to_file):
    """Import input DataFrame from a CSV file."""
    input_df = pd.read_csv(path_to_file, usecols=MODEL_FEATURES)
    print('Data imported...')
    return input_df
