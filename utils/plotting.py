import matplotlib.pyplot as plt
import seaborn as sns

def generate_churn_plot(df):
    # Count the number of at-risk and not-at-risk customers
    churn_counts = df['Churn Prediction'].value_counts()

    plt.figure(figsize=(8, 6))
    sns.barplot(x=churn_counts.index, y=churn_counts.values, palette='coolwarm')
    plt.title('Customer Churn Prediction Distribution')
    plt.xlabel('Churn Status')
    plt.ylabel('Number of Customers')

    return plt
