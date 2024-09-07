from flask import Flask, request, redirect, url_for, render_template, flash
from flask_pymongo import PyMongo
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib
from io import BytesIO
import base64
import os
from werkzeug.utils import secure_filename
from config import UPLOAD_FOLDER, MONGO_URI, SECRET_KEY
import joblib
import logging
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression

# Set the Matplotlib backend to 'Agg' to avoid GUI issues on macOS
matplotlib.use('Agg')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MONGO_URI'] = MONGO_URI
app.config['SECRET_KEY'] = SECRET_KEY
mongo = PyMongo(app)

# Set up logging for better error tracking
logging.basicConfig(level=logging.DEBUG)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv'}

def clean_data(df):
    # Convert categorical columns to numeric
    categorical_cols = df.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders

def generate_cluster_plot(df, labels):
    # Use PCA for dimensionality reduction to plot the first two principal components
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df)
    
    fig, ax = plt.subplots()
    scatter = ax.scatter(df_pca[:, 0], df_pca[:, 1], c=labels, cmap='viridis')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('Clustering')
    plt.colorbar(scatter)
    
    return fig  # Return the figure object

def generate_feature_importance_plot(df):
    # Dummy feature importance: using variance as a proxy for importance
    importance = df.var()
    
    fig, ax = plt.subplots()
    ax.bar(range(len(df.columns)), importance)
    ax.set_xlabel('Features')
    ax.set_ylabel('Variance (used as importance)')
    ax.set_title('Feature Importance')
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=90)
    
    return fig  # Return the figure object

def convert_plot_to_base64(figure):
    buf = BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(figure)  # Close the figure to free up resources
    return 'data:image/png;base64,' + image_base64

def get_cluster_info(df, labels):
    cluster_info = df.groupby(labels).mean().to_dict()
    return cluster_info

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        choice = request.form.get('model_type')  # Get the user's choice

        if not choice:
            flash('No analysis type selected')
            return redirect(request.url)

        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Save file metadata to MongoDB
            mongo.db.uploads.insert_one({'filename': filename, 'filepath': filepath, 'choice': choice})

            # Redirect to the appropriate function based on user's choice
            if choice == 'customer_segmentation':
                return redirect(url_for('process_segmentation', filename=filename))
            elif choice == 'sales_prediction':
                return redirect(url_for('process_sales_prediction', filename=filename))
            elif choice == 'churn_prediction':  # Handle churn prediction
                return redirect(url_for('process_churn_prediction', filename=filename))

        flash('Invalid file format')
        return redirect(request.url)

    return render_template('index.html')

@app.route('/process_segmentation/<filename>')
def process_segmentation(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)

    try:
        # Process and clean the data
        df_cleaned, label_encoders = clean_data(df)
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df_cleaned), columns=df_cleaned.columns)
    except ValueError as e:
        flash(f"Data processing error: {e}")
        return redirect(url_for('index'))

    if df_scaled.empty:
        flash('Data processing error: DataFrame is empty after cleaning')
        return redirect(url_for('index'))

    # Perform clustering
    n_clusters = 3  # You can adjust the number of clusters or make it dynamic
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans_model.fit_predict(df_scaled)

    # Save clustering results in MongoDB with model info
    mongo.db.results.insert_one({
        'filename': filename,
        'labels': labels.tolist(),
        'centroids': kmeans_model.cluster_centers_.tolist(),
        'n_clusters': n_clusters
    })

    # Generate visualizations
    try:
        cluster_plot = generate_cluster_plot(df_scaled, labels)
        feature_importance_plot = generate_feature_importance_plot(df_scaled)

        cluster_plot_url = convert_plot_to_base64(cluster_plot)
        feature_importance_plot_url = convert_plot_to_base64(feature_importance_plot)

        cluster_info = get_cluster_info(df_scaled, labels)
    except Exception as e:
        flash(f"Error generating plots: {e}")
        return redirect(url_for('index'))

    return render_template('cluster_results.html', cluster_plot_url=cluster_plot_url,
                           feature_importance_plot_url=feature_importance_plot_url,
                           cluster_info=cluster_info)


@app.route('/process_sales_prediction/<filename>')
def process_sales_prediction(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)

    try:
        # Ensure the 'Date' column is correctly formatted as datetime
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')

        # Prepare data for training
        df = df.reset_index()  # Reset index to use 'Date' as a column for training
        df['Day'] = (df['Date'] - df['Date'].min()).dt.days  # Create a numerical feature from dates
        X = df[['Day']]
        y = df['Sales']  # Assuming 'Sales' is the column with sales data

        # Train a new model
        model = LinearRegression()
        model.fit(X, y)

        # Save the trained model to a file
        model_path = os.path.join(app.config['UPLOAD_FOLDER'], f'sales_model_{filename}.pkl')
        joblib.dump(model, model_path)

        # Generate future dates and predict sales
        future_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=10), periods=19, freq='10D')
        future_days = (future_dates - df['Date'].min()).days.to_numpy()  # Convert to NumPy array
        future_days = future_days.reshape(-1, 1)  # Reshape for the model input
        future_predictions = model.predict(future_days)

        future_sales_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Sales': future_predictions
        })

        # Plotting future sales predictions
        sales_plot = plt.figure()
        plt.plot(future_sales_df['Date'], future_sales_df['Predicted Sales'])
        plt.xlabel('Date')
        plt.ylabel('Predicted Sales')
        plt.title('Future Sales Predictions')
        sales_plot_url = convert_plot_to_base64(sales_plot)

        # Calculate summary statistics
        sales_summary = {
            'highest_sales_date': future_sales_df.loc[future_sales_df['Predicted Sales'].idxmax()]['Date'],
            'lowest_sales_date': future_sales_df.loc[future_sales_df['Predicted Sales'].idxmin()]['Date']
        }
    except Exception as e:
        flash(f"Error processing sales prediction: {e}")
        return redirect(url_for('index'))

    return render_template('sales_results.html', sales_plot_url=sales_plot_url, sales_summary=sales_summary)

@app.route('/process_churn_prediction', methods=['GET', 'POST'])
def process_churn_prediction():
    if request.method == 'POST':
        file = request.files.get('file')

        if not file or file.filename == '':
            flash('No file provided or file not selected.')
            return redirect(url_for('index'))

        if not allowed_file(file.filename):
            flash('Invalid file format. Please upload a CSV file.')
            return redirect(url_for('index'))

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess the data and predict churn
        try:
            churn_results, df_processed = process_churn_data(filepath)
            print("Churn results processed:", churn_results.head())  # Debugging log

            # Save results for download
            result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'churn_results_{filename}')
            churn_results.to_csv(result_filepath, index=False)
            print(f"Results saved to {result_filepath}")  # Debugging log

            # Generate plots for insights
            churn_plot = generate_churn_plot(churn_results)
            churn_plot_url = convert_plot_to_base64(churn_plot)
            print("Churn plot generated successfully.")  # Debugging log

            return render_template('churn_results.html',
                                   churn_plot_url=churn_plot_url,
                                   download_link=result_filepath)
        except Exception as e:
            flash(f"Error processing churn prediction: {e}")
            print(f"Error processing churn prediction: {e}")  # Debugging log
            return redirect(url_for('index'))

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
