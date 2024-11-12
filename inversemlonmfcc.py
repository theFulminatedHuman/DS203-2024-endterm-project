import numpy as np
import polars as pl
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from IPython.display import Audio
import librosa
import glob
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
def load_and_process_mfcc(file_path):
    try:
        mfcc_data = pd.read_csv(file_path, header=None)
        
        if mfcc_data.empty:
            print(f"Warning: Empty data in file {file_path}")
            return None

        # Retain only the first 3 coefficients
        mel_spec = mfcc_data.iloc[:3, :]

        # Convert MFCCs to Mel spectrogram
        #mel_spec = librosa.feature.inverse.mfcc_to_mel(mel_spec.values)

        # Calculate statistical metrics on Mel spectrogram
        stats_features = [
            float(np.mean(mel_spec)),
            float(np.std(mel_spec)),
            float(np.percentile(mel_spec, 25)),
            float(np.median(mel_spec)),
            float(np.percentile(mel_spec, 75)),
            float(np.max(mel_spec)),
            float(np.min(mel_spec)),
            float(stats.kurtosis(mel_spec.flatten())),
            float(stats.skew(mel_spec.flatten()))
        ]

        return np.array(stats_features)

    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def perform_pca_and_cluster(features_list):
    X = np.array(features_list)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=6, random_state=42)
    cluster_labels = kmeans.fit_predict(X_pca)

    return X_pca, cluster_labels, kmeans

def plot_clusters_2d(X_pca, cluster_labels, file_names):
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='Set1', alpha=0.6, s=100)
    for i, txt in enumerate(file_names):
        plt.annotate(txt.split('-')[0], (X_pca[i, 0], X_pca[i, 1]), fontsize=8, alpha=0.7)

    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('Clustering Results - 2D PCA Visualization')
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    plt.show()

def cluster_audio_files(directory_path):
    mfcc_files = glob.glob(os.path.join(directory_path, "*-MFCC.csv"))
    if not mfcc_files:
        raise ValueError("No MFCC CSV files found")

    features_list = []
    file_names = []

    for file_path in tqdm(mfcc_files):
        features = load_and_process_mfcc(file_path)
        if features is not None:
            features_list.append(features)
            file_names.append(os.path.basename(file_path))

    if not features_list:
        raise ValueError("No files were successfully processed")

    X_pca, cluster_labels, kmeans = perform_pca_and_cluster(features_list)
    silhouette = silhouette_score(X_pca, cluster_labels)
    calinski_harabasz = calinski_harabasz_score(X_pca, cluster_labels)
    davies_bouldin = davies_bouldin_score(X_pca, cluster_labels)
        
    print(f"Silhouette Score: {silhouette:.2f}")
    print(f"Calinski-Harabasz Index: {calinski_harabasz:.2f}")
    print(f"Davies-Bouldin Index: {davies_bouldin:.2f}")
    plot_clusters_2d(X_pca, cluster_labels, file_names)

    results = pl.DataFrame({
        'File': file_names,
        'Cluster': cluster_labels.tolist()
    })

    return results

if __name__ == "__main__":
    directory_path = r"C:\Users\vedan\OneDrive - Indian Institute of Technology Bombay\Resnet-34\MFCC-files-v2\MFCC-files-v2"
    try:
        if not os.path.exists(directory_path):
            raise ValueError("Directory does not exist")
        
        results = cluster_audio_files(directory_path)
        print(results)
        
    except Exception as e:
        print(f"Error: {str(e)}")
