from IPython.display import Audio
import pandas as pd
import librosa
import numpy as np
import os
from tqdm import tqdm
import soundfile as sf

def mfcc_to_audio(mfccs, sr=44100, n_mels=128, n_fft=2048, hop_length=512):
    """
    Convert MFCC coefficients back to audio signal.
    """
    try:
        # Convert MFCCs to Mel Spectrogram
        mel_spec = librosa.feature.inverse.mfcc_to_mel(mfccs, n_mels=n_mels)
        # Convert Mel Spectrogram to Linear-Frequency STFT Spectrogram
        stft_spec = librosa.feature.inverse.mel_to_stft(mel_spec, sr=sr, n_fft=n_fft)
        # Reconstruct Waveform from STFT Spectrogram using Griffin-Lim
        y_reconstructed = librosa.griffinlim(stft_spec, hop_length=hop_length, n_iter=100)
        return y_reconstructed
    except Exception as e:
        print(f"Error in audio reconstruction: {str(e)}")
        return None

def batch_convert_mfcc_to_audio(input_dir=r"C:\Users\vedan\OneDrive - Indian Institute of Technology Bombay\Resnet-34\MFCC-files-v2\MFCC-files-v2", output_dir=r"C:\Users\vedan\OneDrive - Indian Institute of Technology Bombay\Audio Files", sr=44100):
    """
    Convert all MFCC CSV files in a directory to audio files.
    
    Args:
        input_dir (str): Directory containing MFCC CSV files
        output_dir (str): Directory to save reconstructed audio files
        sr (int): Sampling rate for the audio files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of all CSV files in the input directory
    mfcc_files = [f for f in os.listdir(input_dir) if f.endswith('-MFCC.csv')]
    
    print(f"Found {len(mfcc_files)} MFCC files to process")
    
    # Process each file
    for mfcc_file in tqdm(mfcc_files, desc="Converting MFCC to Audio"):
        try:
            # Read MFCC coefficients
            file_path = os.path.join(input_dir, mfcc_file)
            df = pd.read_csv(file_path, header=None)
            mfccs = df.values
            
            # Convert to audio
            reconstructed_audio = mfcc_to_audio(mfccs, sr=sr)
            
            if reconstructed_audio is not None:
                # Generate output filename
                output_filename = mfcc_file.replace('-MFCC.csv', '.wav')
                output_path = os.path.join(output_dir, output_filename)
                
                # Save audio file
                sf.write(output_path, reconstructed_audio, sr)
                
            else:
                print(f"Failed to reconstruct audio for {mfcc_file}")
                
        except Exception as e:
            print(f"Error processing {mfcc_file}: {str(e)}")
            continue
    
    print("\nConversion complete!")
    print(f"Audio files saved in: {output_dir}")

# Run the batch conversion
if __name__ == "__main__":
    batch_convert_mfcc_to_audio()

