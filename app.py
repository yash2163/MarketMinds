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
import joblib
import logging
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression
from flask import send_from_directory

# Set the Matplotlib backend to 'Agg' to avoid GUI issues on macOS
matplotlib.use('Agg')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Update if needed
app.config['MONGO_URI'] = 'mongodb://localhost:27017/yourdatabase'  # Update if needed
app.config['SECRET_KEY'] = 'your_secret_key'  # Update if needed
app.config['RESULTS_FOLDER'] = os.path.join(app.root_path, 'static')

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
    # Example implementation; replace with actual logic for feature importance
    feature_importance = np.random.rand(df.shape[1])  # Dummy feature importances
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df.columns, feature_importance)
    ax.set_title('Feature Importances')
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    plt.xticks(rotation=90)
    plt.tight_layout()
    return fig


def convert_plot_to_base64(figure):
    try:
        buf = BytesIO()
        figure.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(figure)  # Close the figure to free up resources
        app.logger.info('Plot converted to base64.')
        return 'data:image/png;base64,' + image_base64
    except Exception as e:
        app.logger.error('Error converting plot to base64: %s', str(e))
        raise

def get_cluster_info(df, labels):
    cluster_info = df.groupby(labels).mean().to_dict()
    return cluster_info

def preprocess_data(df):
    # Example preprocessing steps
    df = df.dropna()  # Remove missing values
    df = pd.get_dummies(df)  # Convert categorical columns to numeric
    return df

def generate_churn_plot(churn_results):
    try:
        app.logger.info('Generating churn plot...')
        fig, ax = plt.subplots(figsize=(10, 6))
        churn_results['Churn Label'].value_counts().plot(kind='bar', ax=ax)
        ax.set_title('Churn Distribution')
        ax.set_xlabel('Churn Label')
        ax.set_ylabel('Count')
        plt.tight_layout()
        app.logger.info('Churn plot generated.')
        return fig
    except Exception as e:
        app.logger.error('Error generating churn plot: %s', str(e))
        raise

def process_churn_data(df):
    try:
        # Placeholder for churn prediction logic
        # Replace with actual model predictions
        churn_results = df.copy()  # Example placeholder
        churn_results['Churn Label'] = np.random.randint(0, 2, size=len(df))  # Dummy prediction
        return churn_results, df
    except Exception as e:
        app.logger.error(f"Error during churn data processing: {str(e)}")
        raise

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
            elif choice == 'churn_prediction':
                return redirect(url_for('process_churn_prediction', filename=filename))

        flash('Invalid file format')
        return redirect(request.url)

    return render_template('index.html')

# Route for about page
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contactMe')
def contact_me():
    return render_template('contactMe.html')

@app.route('/howToUse')
def howTouse():
    return render_template('howToUse.html')


@app.route('/submit_contact_form', methods=['POST'])
def submit_contact_form():
    name = request.form.get('name')
    email = request.form.get('email')
    subject = request.form.get('subject')
    message = request.form.get('message')
    urgency = request.form.get('urgency')

    # Process form data (e.g., save to database, send email, etc.)
    # For now, we'll just print it to the console
    app.logger.info(f"Received contact form submission: {name}, {email}, {subject}, {message}, {urgency}")

    # Redirect back to the contact page with a success message
    flash('Your message has been sent successfully!', 'success')
    return redirect(url_for('contact_me'))

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
        # Assuming you have a function for feature importance plot
        feature_importance_plot = generate_feature_importance_plot(df_scaled)  # Define this function if needed

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
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        df = df.reset_index()
        df['Day'] = (df['Date'] - df['Date'].min()).dt.days
        X = df[['Day']]
        y = df['Sales']

        model = LinearRegression()
        model.fit(X, y)

        model_path = os.path.join(app.config['UPLOAD_FOLDER'], f'sales_model_{filename}.pkl')
        joblib.dump(model, model_path)

        future_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=10), periods=19, freq='10D')
        future_days = (future_dates - df['Date'].min()).days.to_numpy()
        future_days = future_days.reshape(-1, 1)
        future_predictions = model.predict(future_days)

        future_sales_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Sales': future_predictions
        })

        highest_sales_index = future_sales_df['Predicted Sales'].idxmax()
        lowest_sales_index = future_sales_df['Predicted Sales'].idxmin()
        sales_summary = {
            'highest_sales_date': future_sales_df.loc[highest_sales_index, 'Date'].strftime('%Y-%m-%d'),
            'lowest_sales_date': future_sales_df.loc[lowest_sales_index, 'Date'].strftime('%Y-%m-%d'),
        }

        sales_plot = plt.figure(figsize=(12, 6))
        plt.plot(df['Date'], df['Sales'], label='Historical Sales')
        plt.plot(future_dates, future_predictions, label='Predicted Sales', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.title('Sales Prediction')
        plt.legend()
        plt.tight_layout()

        sales_plot_path = os.path.join(app.config['RESULTS_FOLDER'], 'sales_plot.png')
        sales_plot.savefig(sales_plot_path)
        plt.close(sales_plot)

        if not os.path.isfile(sales_plot_path):
            app.logger.error(f"Plot file not found: {sales_plot_path}")

        predictions = future_sales_df.to_dict(orient='records')

        return render_template('sales_results.html',
                               sales_plot_url=url_for('static', filename='sales_plot.png'),
                               sales_summary=sales_summary,
                               predictions=predictions)
    except Exception as e:
        app.logger.error(f"Error during sales prediction: {str(e)}")
        flash(f"Error during sales prediction: {e}")
        return redirect(url_for('index'))

@app.route('/process_churn_prediction/<filename>')
def process_churn_prediction(filename):
    app.logger.debug(f'Filename received for churn prediction: {filename}')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    app.logger.debug(f'Filepath constructed: {filepath}')
    
    if not os.path.isfile(filepath):
        app.logger.error(f'File does not exist: {filepath}')
        flash('File not found')
        return redirect(url_for('index'))
    
    try:
        df = pd.read_csv(filepath)
        churn_results, _ = process_churn_data(df)

        # Generate and save churn plot
        churn_plot = generate_churn_plot(churn_results)
        churn_plot_path = os.path.join('static', 'churn_plot.png')  # Save in 'static'
        churn_plot.savefig(churn_plot_path)
        plt.close(churn_plot)

        # Save churn results CSV to static
        churn_results_filepath = os.path.join('static', 'churn_results.csv')  # Save in 'static'
        churn_results.to_csv(churn_results_filepath, index=False)
        
        return render_template('churn_results.html', plot_url=url_for('static', filename='churn_plot.png'),
                               churn_results=churn_results.to_html(classes='table table-striped'),
                               download_link=url_for('download_churn_results'))
    except Exception as e:
        app.logger.error(f"Error during churn prediction: {str(e)}")
        flash(f"Error during churn prediction: {e}")
        return redirect(url_for('index'))

    
@app.route('/download_churn_results')
def download_churn_results():
    try:
        filename = 'churn_results.csv'  # Ensure this matches the file saved earlier
        return send_from_directory('static', filename, as_attachment=True)
    except Exception as e:
        app.logger.error(f"Error serving file: {str(e)}")
        flash(f"Error serving file: {e}")
        return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
