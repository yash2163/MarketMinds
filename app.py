import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg to avoid GUI issues

from flask import Flask, request, redirect, url_for, render_template, flash
from flask_pymongo import PyMongo
from werkzeug.utils import secure_filename
import os
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)
app.config.from_pyfile('config.py')

mongo = PyMongo(app)

# Allowed file extensions for CSV
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
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
            mongo.db.uploads.insert_one({'filename': filename, 'filepath': filepath})

            # Process and clean the data
            df = pd.read_csv(filepath)
            try:
                df_cleaned = clean_data(df)
            except ValueError as e:
                flash(f"Data processing error: {e}")
                return redirect(request.url)

            # Perform clustering
            n_clusters = 3  # You can adjust the number of clusters or make it dynamic
            kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans_model.fit_predict(df_cleaned)

            # Save clustering results in MongoDB
            mongo.db.results.insert_one({'filename': filename, 'labels': labels.tolist()})

            # Generate visualizations
            cluster_plot = generate_cluster_plot(df_cleaned, labels)
            feature_importance_plot = generate_feature_importance_plot(df_cleaned, labels)
            
            # Encode plots to display in HTML
            cluster_plot_url = convert_plot_to_base64(cluster_plot)
            feature_importance_plot_url = convert_plot_to_base64(feature_importance_plot)

            # Get cluster information
            cluster_info = get_cluster_info(df_cleaned, labels)

            return render_template('results.html', cluster_plot_url=cluster_plot_url,
                                   feature_importance_plot_url=feature_importance_plot_url,
                                   cluster_info=cluster_info)

    return render_template('index.html')

def clean_data(df):
    # Drop rows with missing values
    df = df.dropna()
    
    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # One-hot encode categorical columns
    if len(categorical_cols) > 0:
        df = pd.get_dummies(df, columns=categorical_cols)
    
    # Normalize numerical features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df

def generate_cluster_plot(data, labels):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
    plt.title("Customer Segmentation Clusters")
    plt.xlabel("PCA Feature 1")
    plt.ylabel("PCA Feature 2")
    plt.colorbar(label="Cluster")
    
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return buf

def generate_feature_importance_plot(data, labels):
    cluster_data = pd.DataFrame(data)
    cluster_data['Cluster'] = labels
    cluster_summary = cluster_data.groupby('Cluster').mean()
    
    cluster_summary.T.plot(kind='bar', figsize=(12, 8))
    plt.title("Feature Importance Across Clusters")
    plt.xlabel("Features")
    plt.ylabel("Average Value per Cluster")
    plt.xticks(rotation=45)
    
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return buf

def get_cluster_info(data, labels):
    cluster_info = {}
    cluster_data = pd.DataFrame(data)
    cluster_data['Cluster'] = labels
    
    for cluster in cluster_data['Cluster'].unique():
        cluster_subset = cluster_data[cluster_data['Cluster'] == cluster]
        info = {
            "Size": len(cluster_subset),
            "Avg_Spending": cluster_subset.iloc[:, 0].mean(),  # Example feature
            "Common_Age_Group": cluster_subset.iloc[:, 1].mode()[0],  # Example feature
            "Top_Location": cluster_subset.iloc[:, 2].mode()[0],  # Example feature
            "Other_Metrics": cluster_subset.describe().T.to_dict()  # Example feature
        }
        cluster_info[cluster] = info
    
    return cluster_info

def convert_plot_to_base64(buf):
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{image_base64}'

if __name__ == "__main__":
    app.run(debug=True)
