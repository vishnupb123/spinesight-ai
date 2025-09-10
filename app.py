# app.py
import csv
import io
import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_file , jsonify , session
import joblib
import numpy as np
import pandas as pd
import traceback
from io import BytesIO

from utils.visualization import generate_3d_spine_plot
from utils.log_prediction import log_prediction_to_csv
from utils.pdf_generator import generate_individual_report, generate_bulk_report
from utils.ai_integration import get_ai_verdict
from utils.visualization_utils import generate_comparison_bar_chart, generate_radar_chart
from utils.utils import get_records 


# from config import Config
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'default-secret')
    PDFKIT_CONFIG = {
        'wkhtmltopdf': os.environ.get('WKHTMLTOPDF_PATH', '/usr/local/bin/wkhtmltopdf')
    }

app = Flask(__name__)
app.config.from_object(Config)
session_secret_key = os.urandom(24)  # Generate a random session secret key
app.secret_key = session_secret_key

# List of expected input features
input_features = [
    'pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle',
    'sacral_slope', 'pelvic_radius', 'degree_spondylolisthesis',
    'age', 'gender', 'abdominal_pain'
]

# Load models and transformers
lr_model = joblib.load('models/spine_lr.pkl')
svc_model = joblib.load('models/spine_svc.pkl')
rf_model = joblib.load('models/spine_rf.pkl')
scaler = joblib.load('models/scaler.pkl')
power_transformer = joblib.load('models/power_transformer.pkl')

# Labels for prediction display: (Result, Bootstrap color class, Message)
label_map = {
    0: ("Hernia", "warning", "Signs of herniated disc detected. Consider a clinical consultation for further evaluation."),
    1: ("Normal", "success", "Your spine alignment appears normal. Keep maintaining a healthy lifestyle!"),
    2: ("Spondylolisthesis", "danger", "Warning: Potential spondylolisthesis detected. Please consult a spine specialist promptly.")
}

def categorize_slip(degree):
    """Categorize slip severity based on degree value."""
    if degree < 6:
        return 'Normal'
    elif degree < 17:
        return 'Mild'
    else:
        return 'Severe'

def preprocess_input(input_data: dict, model_type: str = 'lr_svc') -> np.ndarray:
    df = pd.DataFrame([input_data])[input_features].copy()
    
    # Cast types
    df['gender'] = 1 if str(df['gender'].iloc[0]).lower() in ['male', 'm', '1'] else 0
    df['abdominal_pain'] = 1 if str(df['abdominal_pain'].iloc[0]).lower() in ['yes', '1', 'true'] else 0
    df['age'] = df['age'].astype(float)

    # Clean up
    df[['pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle',
        'sacral_slope', 'pelvic_radius', 'degree_spondylolisthesis']] = \
        df[['pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle',
            'sacral_slope', 'pelvic_radius', 'degree_spondylolisthesis']].astype(float)

    # Clip & transform
    df['pelvic_tilt'] = df['pelvic_tilt'].clip(lower=0)
    df['sacral_slope'] = df['sacral_slope'].clip(lower=0)
    df['degree_spondylolisthesis'] = df['degree_spondylolisthesis'].clip(lower=0)
    df[['pelvic_tilt', 'pelvic_radius']] = power_transformer.transform(df[['pelvic_tilt', 'pelvic_radius']])

    # Categorize slip
    slip_cat = categorize_slip(df['degree_spondylolisthesis'].values[0])
    df['slip_Mild'] = int(slip_cat == 'Mild')
    df['slip_Normal'] = int(slip_cat == 'Normal')
    df['slip_Severe'] = int(slip_cat == 'Severe')

    if model_type == 'lr_svc':
        df['degree_spondylolisthesis_log'] = np.log1p(df['degree_spondylolisthesis'])
        df.drop(columns=['degree_spondylolisthesis'], inplace=True)
        final_features = [
            'pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle',
            'sacral_slope', 'pelvic_radius', 'age', 'gender', 'abdominal_pain',
            'slip_Mild', 'slip_Normal', 'slip_Severe', 'degree_spondylolisthesis_log'
        ]
        return scaler.transform(df[final_features])
    else:
        final_features = [
            'pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle',
            'sacral_slope', 'pelvic_radius', 'degree_spondylolisthesis',
            'age', 'gender', 'abdominal_pain',
            'slip_Mild', 'slip_Normal', 'slip_Severe'
        ]
        return df[final_features].values


@app.route('/')
def index():
     # Check if the session has any previous prediction data
    input_data = session.get('input_data', {})
    result = session.get('result', "")
    confidence = session.get('confidence', "")
    message = session.get('message', "")
    fig_div = session.get('fig_div', "")
    radar_chart_div = session.get('radar_chart_div', "")
    ai_verdict = session.get('ai_verdict', "")

    return render_template(
        'index.html',
        input_features=input_data,
        prediction_text=result,
        prediction_color="success" if result == "Normal" else "danger",  # Example for "Normal" or "Spondylolisthesis"
        prediction_message=message,
        confidence_score=confidence,
        fig_div=fig_div,
        radar_chart_div=radar_chart_div,
        ai_verdict=ai_verdict
    )
    # Render prediction page; initial 3D plot is empty.
    # return render_template('index.html', input_features={}, plotly_3d_div="", prediction_text="")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Gather user input
        input_data = {
        feature: request.form[feature] if feature in ['gender', 'abdominal_pain']
        else float(request.form[feature])
        for feature in input_features
        }
        model_choice = request.form['model']

        if model_choice == 'lr':
            X_processed = preprocess_input(input_data, 'lr_svc')
            prediction = lr_model.predict(X_processed)[0]
            confidence = lr_model.predict_proba(X_processed).max() * 100

        elif model_choice == 'svc':
            X_processed = preprocess_input(input_data, 'lr_svc')
            prediction = svc_model.predict(X_processed)[0]
            scores = svc_model.decision_function(X_processed)
            confidence = scores[0][prediction] if scores.ndim > 1 else scores[0]
            # Scale the confidence value appropriately
            confidence = min(abs(confidence), 10.0) / 10.0 * 100

        elif model_choice == 'rf':
            X_processed = preprocess_input(input_data, 'rf')
            prediction = rf_model.predict(X_processed)[0]
            confidence = rf_model.predict_proba(X_processed).max() * 100

        else:
            raise ValueError("Invalid model selected.")

        result, color, message = label_map[prediction]
        
        # # Generate a 3D spine visualization using the entered data
        # plotly_3d_div = generate_3d_spine_plot(input_data)
         # Generate heatmap and radar chart visualizations
        fig_div = generate_comparison_bar_chart(input_data)  # Call the heatmap generation function
        radar_chart_div = generate_radar_chart(input_data)  # Call the radar chart generation function
        
        # Generate an AI verdict (recommendation) based on the results
        ai_verdict = get_ai_verdict(result, input_data, message)
        log_prediction_to_csv(input_data, result, confidence, message)
        
        # #preserve the sesson data
        # session['input_data'] = input_data
        # session['result'] = result
        # session['confidence'] = confidence
        # session['message'] = message
        # session['ai_verdict'] = ai_verdict
        # session['fig_div'] = fig_div  # Assuming `fig_div` is generated in your function
        # session['radar_chart_div'] = radar_chart_div
        # # Render the result page with the prediction and visualizations
        
        

        return render_template(
            'index.html',
            prediction_text=result,
            prediction_color=color,
            prediction_message=message,
            confidence_score=f"{confidence:.2f}",
            input_features=input_data,
            # plotly_3d_div=plotly_3d_div,
            fig_div=fig_div,
            radar_chart_div=radar_chart_div,
            ai_verdict=ai_verdict
        )

    except Exception as e:
        traceback.print_exc()
        return render_template(
            'index.html',
            prediction_text=f"Error: {str(e)}",
            prediction_color="secondary",
            prediction_message="Something went wrong. Please double-check your input values.",
            input_features=request.form,
            plotly_3d_div="",
            ai_verdict=""
        )

@app.route('/bulk_upload', methods=['GET', 'POST'])
def bulk_upload():
    if request.method == 'POST':
        file = request.files.get('csv_file')
        if file:
            try:
                df = pd.read_csv(file)
                results = []  # Accumulate prediction results per row
                for _, row in df.iterrows():
                    input_data = {feature: float(row[feature]) for feature in input_features}
                    # For bulk, we choose one default model (Logistic Regression) for consistency.
                    X_processed = preprocess_input(input_data, 'rf')
                    prediction = lr_model.predict(X_processed)[0]
                    confidence = lr_model.predict_proba(X_processed).max() * 100
                    result, color, message = label_map[prediction]
                    
                    # Also generate a 3D visualization image (if needed for the report)
                    results.append({
                        'input': input_data,
                        'prediction': result,
                        'color': color,
                        'confidence': f"{confidence:.2f}",
                        'message': message
                    })
                # Generate the bulk PDF report
                pdf_data = generate_bulk_report(results, app.config["PDFKIT_CONFIG"])
                return send_file(BytesIO(pdf_data), download_name="bulk_report.pdf", as_attachment=True)
            except Exception as e:
                traceback.print_exc()
                flash(f"Error processing CSV: {str(e)}", "danger")
                return redirect(url_for('bulk_upload'))
    return render_template('bulk_upload.html')

@app.route('/download_report', methods=['POST'])
def download_report():
    try:
        # After a single prediction, generate a detailed PDF report
        # Get data from hidden fields in form if needed; here we assume it's sent as JSON string or individual fields.
        input_data = {
            feature: request.form[feature] if feature in ['gender', 'abdominal_pain']
            else float(request.form[feature])
            for feature in input_features
        }
        result = request.form.get("result")
        confidence = request.form.get("confidence")
        message = request.form.get("message")
        ai_verdict = request.form.get("ai_verdict")
        # Generate 3D visualization image as HTML snippet; you might also choose to generate a static image.
        # plotly_3d_div = generate_3d_spine_plot(input_data)
        radar_chart_div = generate_radar_chart(input_data)  # Call the radar chart generation function
        fig_div = generate_comparison_bar_chart(input_data)  # Call the heatmap generation function
        
        # Prepare report data dictionary for the template
        report_data = {
            'input_data': input_data,
            'result': result,
            'confidence': confidence,
            'message': message,
            # 'plotly_3d_div': plotly_3d_div,
            'fig_div': fig_div,
            'radar_chart_div': radar_chart_div,
            # 'prediction_color': label_map[int(result)][1],
            'ai_verdict': ai_verdict
        }
        pdf_data = generate_individual_report(report_data, app.config["PDFKIT_CONFIG"])
        return send_file(BytesIO(pdf_data), download_name="report.pdf", as_attachment=True)
    except Exception as e:
        traceback.print_exc()
        flash(f"Error generating report: {str(e)}", "danger")
        return redirect(url_for('index'))

@app.route('/history', methods=['GET'])
def history():
    # Fetch all records
    records = get_records()

    # Get filter condition (Normal, Hernia, Spondylolisthesis) from query string
    condition_filter = request.args.get('condition', '').capitalize()

    # If there's a condition filter, filter the records
    if condition_filter:
        records = [record for record in records if record['prediction'] == condition_filter]

    # Prepare data for charts
    timestamps = [record['timestamp'] for record in records]
    confidences = [record['confidence'] for record in records]
    predictions = [record['prediction'] for record in records]
    
    # For feature tracking (e.g., pelvic_tilt, sacral_slope)
    feature_names = ['pelvic_tilt', 'sacral_slope','degree_spondylolisthesis','pelvic_radius','lumbar_lordosis_angle']  # List of features to track
    feature_data = {feature: [record[feature] for record in records] for feature in feature_names}

    return render_template(
        'history.html', 
        records=records, 
        timestamps=timestamps, 
        confidences=confidences, 
        predictions=predictions,
        feature_data=feature_data,
        condition_filter=condition_filter  # Pass current filter back to the template
    )

@app.route('/download_history', methods=['GET'])
def download_history():
    # Fetch all records
    records = get_records()

    # Get filter condition (Normal, Hernia, Spondylolisthesis) from query string
    condition_filter = request.args.get('condition', '').capitalize()

    # If there's a condition filter, filter the records
    if condition_filter:
        records = [record for record in records if record['prediction'] == condition_filter]

    # Create a CSV in-memory file
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)
    output.seek(0)

    # Return the CSV as a downloadable file
    return send_file(
        output,
        mimetype='text/csv',
        as_attachment=True,
        download_name='history.csv'
    )

if __name__ == '__main__':
    app.run(port=8000 , debug=True)
