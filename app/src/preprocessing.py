# Import libraries
import pandas as pd

# Define columns
features = [
    'частота_пополнения', 'доход', 'объем_данных',
    'on_net', 'продукт_1', 'секретный_скор',
]


def import_data(path_to_file):

    # Get input dataframe
    input_df = pd.read_csv(path_to_file, usecols=features)
    print('Data imported...')

    return input_df
