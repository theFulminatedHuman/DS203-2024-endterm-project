import numpy as np
import polars as pl
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_process_mfcc(file_path):
    try:
        mfcc_data = pl.read_csv(file_path, has_header=False)
        
        if mfcc_data.is_empty():
            print(f"Warning: Empty data in file {file_path}")
            return None
            
        if mfcc_data.shape[0] != 20:  
            print(f"Warning: Unexpected number of MFCC coefficients in {file_path}: {mfcc_data.shape[0]}")
            return None

        features = []
        mfcc_numpy = mfcc_data.to_numpy()
        
        for mfcc_row in mfcc_numpy:
            coefficient_data = mfcc_row.flatten()
            
            try:
                stats_features = [
                    float(np.mean(coefficient_data)),
                    float(np.std(coefficient_data)),
                    float(np.percentile(coefficient_data, 25)),
                    float(np.median(coefficient_data)),
                    float(np.percentile(coefficient_data, 75)),
                    float(np.max(coefficient_data)),
                    float(np.min(coefficient_data)),
                    float(stats.kurtosis(coefficient_data)),
                    float(stats.skew(coefficient_data))
                ]
                features.extend(stats_features)
            except Exception as stat_error:
                print(f"Error calculating statistics for {file_path}: {str(stat_error)}")
                return None

        return np.array(features)
        
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def plot_clusters_2d(X_pca, cluster_labels, file_names, algorithm_name):
    
    plt.figure(figsize=(12, 8))
    
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                         cmap='Set1', alpha=0.6, s=100)
    
    for i, txt in enumerate(file_names):
        file_num = txt.split('-')[0]
        plt.annotate(file_num, (X_pca[i, 0], X_pca[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.7)
    
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title(f'Clustering Results - {algorithm_name}')
    
    plt.tight_layout()
    plt.show()

def cluster_and_evaluate(X_pca, file_names):
    clustering_algorithms = {
        'KMeans': KMeans(n_clusters=6, random_state=42, n_init=10),
        'Agglomerative': AgglomerativeClustering(n_clusters=6),
        'DBSCAN': DBSCAN(eps=0.5, min_samples=5)
    }
    
    evaluation_results = []

    for algo_name, model in clustering_algorithms.items():
        print(f"\nRunning {algo_name} clustering...")
        
        if algo_name == 'DBSCAN':
            cluster_labels = model.fit_predict(X_pca)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            print(f"{algo_name} estimated clusters: {n_clusters}")
            if n_clusters < 2:
                print("Skipping evaluation as DBSCAN found less than 2 clusters")
                continue
        else:
            cluster_labels = model.fit_predict(X_pca)

        # Compute evaluation metrics
        silhouette = silhouette_score(X_pca, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(X_pca, cluster_labels)
        davies_bouldin = davies_bouldin_score(X_pca, cluster_labels)
        
        print(f"Silhouette Score: {silhouette:.2f}")
        print(f"Calinski-Harabasz Index: {calinski_harabasz:.2f}")
        print(f"Davies-Bouldin Index: {davies_bouldin:.2f}")

        # Append metrics to results
        evaluation_results.append({
            'Algorithm': algo_name,
            'Silhouette Score': silhouette,
            'Calinski-Harabasz Index': calinski_harabasz,
            'Davies-Bouldin Index': davies_bouldin
        })

        # Plot clustering result
        plot_clusters_2d(X_pca, cluster_labels, file_names, algo_name)
    
    return evaluation_results

def summarize_results(evaluation_results):
    print("\nClustering Evaluation Summary:")
    print("{:<15} {:<20} {:<25} {:<20}".format(
        'Algorithm', 'Silhouette Score', 'Calinski-Harabasz Index', 'Davies-Bouldin Index'))
    
    for result in evaluation_results:
        print("{:<15} {:<20.2f} {:<25.2f} {:<20.2f}".format(
            result['Algorithm'], result['Silhouette Score'], 
            result['Calinski-Harabasz Index'], result['Davies-Bouldin Index']))

if __name__ == "__main__":
    directory_path = r"C:\Users\vedan\OneDrive - Indian Institute of Technology Bombay\Resnet-34\MFCC-files-v2\MFCC-files-v2"
    
    try:
        if not os.path.exists(directory_path):
            raise ValueError(f"Directory does not exist: {directory_path}")
     
        mfcc_files = glob.glob(os.path.join(directory_path, "*-MFCC.csv"))
        
        features_list = []
        file_names = []

        for file_path in tqdm(mfcc_files):
            features = load_and_process_mfcc(file_path)
            if features is not None:
                features_list.append(features)
                file_names.append(os.path.basename(file_path))

        X = np.array(features_list)

        print("Standardizing features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        print("Applying PCA for visualization...")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        print("Running clustering and evaluation...")
        evaluation_results = cluster_and_evaluate(X_pca, file_names)
        
        # Print summary of results
        summarize_results(evaluation_results)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
