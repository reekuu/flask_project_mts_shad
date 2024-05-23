import pandas as pd

# Import libs to solve classification task
from catboost import CatBoostClassifier


# Make prediction
def make_predict(dt, path_to_file):

    print('Importing pretrained model...')
    # Import model
    model = CatBoostClassifier()
    model.load_model('./models/catboost_model.cbm')

    # Make submission dataframe
    submission = pd.DataFrame({
        'client_id':  pd.read_csv(path_to_file)['client_id'],
        'prediction': model.predict(dt)
    })
    print('Prediction complete!')

    # Return submission DataFrame
    return submission
