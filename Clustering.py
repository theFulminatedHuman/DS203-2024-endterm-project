import numpy as np
import polars as pl
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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

def plot_clusters_2d(X_pca, cluster_labels, file_names):
    
    plt.figure(figsize=(12, 8))
    
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                         cmap='Set1', alpha=0.6, s=100)
    
    for i, txt in enumerate(file_names):
        file_num = txt.split('-')[0]
        plt.annotate(file_num, (X_pca[i, 0], X_pca[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.7)
    
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=scatter.cmap(scatter.norm(i)), 
                                 label=f'Cluster {i+1}', markersize=10)
                      for i in range(6)]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('Clustering Results - 2D PCA Visualization')
    
    plt.tight_layout()
    
    return plt.gcf()

def cluster_audio_files(directory_path):
    
    mfcc_files = glob.glob(os.path.join(directory_path, "*-MFCC.csv"))
    
    if not mfcc_files:
        raise ValueError(f"No MFCC CSV files found in directory: {directory_path}")
    
    print(f"Found {len(mfcc_files)} MFCC files")
    
    print("Processing MFCC files...")
    features_list = []
    file_names = []
    
    for file_path in tqdm(mfcc_files):
        features = load_and_process_mfcc(file_path)
        if features is not None:
            features_list.append(features)
            file_names.append(os.path.basename(file_path))
    
    if not features_list:
        raise ValueError("No files were successfully processed")
    
    print(f"Successfully processed {len(features_list)} files")
    
    X = np.array(features_list)
    
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    
    print("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Applying PCA for visualization...")
    pca = PCA(n_components=2)  
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    
    print("Performing clustering with 6 clusters...")
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_pca)
    
    plot = plot_clusters_2d(X_pca, cluster_labels, file_names)
    
    results = pl.DataFrame({
        'File': file_names,
        'Cluster': cluster_labels.tolist()
    })
    
    return results, X_pca, kmeans, plot

def analyze_clusters(results, X_pca, kmeans):
    
    cluster_stats = []
    
    results_numpy = results.to_numpy()
    files = results_numpy[:, 0]
    clusters = results_numpy[:, 1].astype(int)
    
    for cluster in range(kmeans.n_clusters):
        cluster_mask = clusters == cluster
        cluster_files = files[cluster_mask]
        cluster_points = X_pca[cluster_mask]
        
        if len(cluster_points) > 0:
            cohesion = float(np.mean(np.linalg.norm(cluster_points - kmeans.cluster_centers_[cluster], axis=1)))
        else:
            cohesion = 0.0
            
        cluster_info = {
            'Cluster': cluster + 1,  
            'Size': len(cluster_files),
            'Files': cluster_files.tolist(),
            'Cohesion': cohesion
        }
        cluster_stats.append(cluster_info)
    
    return cluster_stats

if __name__ == "__main__":
    directory_path = 'home/downloads/MFCC-files-v2'
    
    try:
        if not os.path.exists(directory_path):
            raise ValueError(f"Directory does not exist: {directory_path}")
     
        results, X_pca, kmeans, plot = cluster_audio_files(directory_path)
        
        cluster_stats = analyze_clusters(results, X_pca, kmeans)
        
        print("\nClustering Results:")
        for stat in cluster_stats:
            print(f"\nCluster {stat['Cluster']}:")
            print(f"Number of files: {stat['Size']}")
            print(f"Cluster cohesion: {stat['Cohesion']:.2f}")
            print("Files:", ', '.join(stat['Files'][:5]), '...' if stat['Size'] > 5 else '')
        
        plot.savefig('cluster_visualization.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'cluster_visualization.png'")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()