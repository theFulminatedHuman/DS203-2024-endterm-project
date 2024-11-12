import numpy as np
import polars as pl
from scipy import stats
from sklearn.preprocessing import StandardScaler
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_process_mfcc(file_path):
    """
    Load MFCC data from CSV and compute statistical features
    """
    try:
        # Read MFCC data using polars
        mfcc_data = pl.read_csv(file_path, has_header=False)
        
        # Verify data is loaded correctly
        if mfcc_data.is_empty():
            print(f"Warning: Empty data in file {file_path}")
            return None
            
        if mfcc_data.shape[0] != 20:  # We expect 20 MFCC coefficients
            print(f"Warning: Unexpected number of MFCC coefficients in {file_path}: {mfcc_data.shape[0]}")
            return None

        # Calculate statistical features for each MFCC coefficient
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

        # Normalize the feature vector to unit length
        feature_vector = np.array(features)
        feature_vector = feature_vector / np.linalg.norm(feature_vector)
        return feature_vector
        
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def create_similarity_heatmap(feature_vectors, file_names):
    """
    Create a heatmap visualization of the dot product similarity between audio files
    """
    # Compute dot product similarity matrix
    similarity_matrix = np.dot(feature_vectors, feature_vectors.T)
    
    # Create the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, annot=True, cmap='YlOrRd')
    plt.title('Audio File Similarity Heatmap')
    plt.xlabel('File')
    plt.ylabel('File')
    plt.tight_layout()
    plt.show()
    
    return plt.gcf()

def cluster_audio_files(directory_path):
    """
    Main function to compute audio file similarity heatmap
    """
    # Get list of all MFCC CSV files
    mfcc_files = glob.glob(os.path.join(directory_path, "*-MFCC.csv"))
    
    if not mfcc_files:
        raise ValueError(f"No MFCC CSV files found in directory: {directory_path}")
    
    print(f"Found {len(mfcc_files)} MFCC files")
    
    # Process each file and create feature matrix
    print("Processing MFCC files...")
    feature_vectors = []
    file_names = []
    
    for file_path in tqdm(mfcc_files):
        features = load_and_process_mfcc(file_path)
        if features is not None:
            feature_vectors.append(features)
            file_names.append(os.path.basename(file_path))
    
    if not feature_vectors:
        raise ValueError("No files were successfully processed")
    
    # Create similarity heatmap
    similarity_plot = create_similarity_heatmap(np.array(feature_vectors), file_names)
    
    return similarity_plot

if __name__ == "__main__":
    directory_path = r"C:\Users\vedan\OneDrive - Indian Institute of Technology Bombay\Resnet-34\MFCC-files-v2\MFCC-files-v2"
    
    try:
        # Perform similarity analysis
        similarity_plot = cluster_audio_files(directory_path)
        
        # Save the heatmap
        similarity_plot.savefig('similarity_heatmap.png', dpi=300, bbox_inches='tight')
       
        print("\nSimilarity heatmap saved as 'similarity_heatmap.png'")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()