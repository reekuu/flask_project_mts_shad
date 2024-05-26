import os
from datetime import datetime

import src.artifacts as artifacts
import src.preprocessing as preprocessing
import src.scorer as scorer
from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for)
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'csv'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_app():
    app = Flask(__name__)

    @app.route('/')
    def redirect_to_upload():
        return redirect('/upload', 302)

    @app.route('/upload', methods=['GET', 'POST'])
    def upload():
        if request.method == 'POST':

            # Import file
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)

                # Store imported file locally
                new_filename = f'{filename.split(".")[0]}_{str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))}.csv'
                save_location = os.path.join('input', new_filename)
                file.save(save_location)

                # Get input dataframe
                input_df = preprocessing.import_data(save_location)

                # Run scorer to get submission file for competition
                submission = scorer.make_predict(input_df, save_location)
                submission.to_csv(save_location.replace('input', 'output'), index=False)

                # Save Density plot
                plot_checkbox = request.form.get('plot')
                if plot_checkbox:
                    artifacts.output_density_plot(submission['preds'], new_filename)  # noqa

                # Save Feature Importance
                json_checkbox = request.form.get('json')
                if json_checkbox:
                    artifacts.output_feature_importance()

                return redirect(url_for('download'))

        return render_template('upload.html')

    @app.route('/download')
    def download():
        return render_template('download.html', files=os.listdir('output'))

    @app.route('/download/<filename>')
    def download_file(filename):
        return send_from_directory('output', filename)
    return app
