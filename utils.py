import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib
from io import BytesIO
import base64

# Set the Matplotlib backend to 'Agg' to avoid GUI issues on macOS
matplotlib.use('Agg')

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
    # Use PCA to reduce dimensionality for plotting
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df)
    plt.figure()
    plt.scatter(df_pca[:, 0], df_pca[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.title('Cluster Plot')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    return plt

def generate_feature_importance_plot(df):
    plt.figure()
    df.mean().plot(kind='bar')
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Mean Value')
    return plt

def convert_plot_to_base64(plot):
    buffer = BytesIO()
    plot.savefig(buffer, format='png')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def get_cluster_info(df, labels):
    df['Cluster'] = labels
    cluster_info = df.groupby('Cluster').mean()
    return cluster_info.to_html()

def process_churn_data(filepath):
    df = pd.read_csv(filepath)
    # Process the churn data
    # This function should return a DataFrame of results and potentially other useful information
    return df, df  # Modify this as needed for your actual processing

def generate_churn_plot(churn_results):
    plt.figure()
    churn_results['Churned'].value_counts().plot(kind='bar')
    plt.title('Churn Distribution')
    plt.xlabel('Churned')
    plt.ylabel('Count')
    return plt
