import pandas as pd
from catboost import CatBoostClassifier


def import_model():
    """Import pretrained model."""
    model = CatBoostClassifier()
    model.load_model('./models/catboost_model.cbm')
    return model


# Make prediction
def make_predict(dt, path_to_file):
    """Return DataFrame with prediction."""
    print('Importing pretrained model...')
    model = import_model()

    # Make submission DataFrame
    submission = pd.DataFrame({
        'client_id':  pd.read_csv(path_to_file)['client_id'],
        'preds': model.predict(dt)  # noqa
    })
    print('Prediction complete!')
    return submission
