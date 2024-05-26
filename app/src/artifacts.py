import json
import os

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from src.scorer import import_model  # noqa


def output_density_plot(predictions, filename):
    """Create a density plot and save it as a PNG file in output folder."""
    matplotlib.use('agg')
    sns.kdeplot(predictions, fill=True)
    plt.title('Prediction density plot')
    plt.xlabel('Predictions')
    plt.ylabel('Density')

    new_filename = f'{filename.split(".")[0]}_density_plot.png'
    save_location = os.path.join('output', new_filename)
    plt.savefig(save_location)
    plt.close()
    print('Density plot created.')


def output_feature_importance(top_n=5):
    """Extract feature importance from model and save it as a JSON file in output folder."""
    model = import_model()
    f_importance = model.get_feature_importance()
    f_names = model.feature_names_

    values = list(zip(f_names, f_importance))
    values_sorted = sorted(values, key=lambda item: item[1], reverse=True)
    values_top_n = dict(values_sorted[:top_n])

    save_location = os.path.join('output', 'feature_importance.json')
    with open(save_location, 'w') as destination:
        json.dump(values_top_n, destination, ensure_ascii=False, indent=2)
    print('Feature importance exported.')
