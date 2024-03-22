import librosa
import argparse
import numpy as np
import librosa.display
from augment import SpecAugment
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='D:/AST_Vit/vocalsound/data_44k', help='path to dataset/dir to look for files')
parser.add_argument('--policy', default='LD', help='augmentation policies - LB, LD, SM, SS')

args = parser.parse_args()


if __name__ == '__main__':

    my_dict = {"laughter": 0,
    "sigh": 1,
    "cough": 2,
    "throatclearing": 3,
    "sneeze": 4,
    "sniff": 5
    }
    
    df2=pd.DataFrame()
    # make a list of all training files in the LibriSpeech Dataset
    # training_files = librosa.util.find_files(args.dir, ext=['flac'], recurse=True)
    training_files=args.dir
    print('Number of Training Files: ', len(os.listdir(training_files)))
    
    count=0
    with h5py.File('aug_vocalsound.hdf5', 'w') as f:
        for file in os.listdir(training_files):
            print(file)
            file_path=os.path.join(args.dir,file)
            print(file_path)
            # break
            
            # Load the audio file
            try:
                audio, sr = librosa.load(file_path)
                
                # Extract Mel Spectrogram Features from the audio file
                mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=256, hop_length=128, fmax=8000)
                apply = SpecAugment(mel_spectrogram, args.policy)
                
                time_warped = apply.time_warp() # Applies Time Warping to the mel spectrogram
                freq_masked = apply.freq_mask() # Applies Frequency Masking to the mel spectrogram
                time_masked = apply.time_mask() # Applies Time Masking to the mel spectrogram
                Label=my_dict[file[8:-4]]
            
                new_rows = [{0: file ,1:Label},
                        {0: file[:-4]+'_tw.wav' ,1:Label},
                        {0: file[:-4]+'_fm.wav' ,1:Label},
                        {0: file[:-4]+'_tm.wav' ,1:Label}]
                df2 = pd.concat([df2, pd.DataFrame(new_rows)], ignore_index=True)
            

                f.create_dataset(file, data=mel_spectrogram)# Store the spectrogram of a signal into h5py database and indexed with a signal name(file variable here)
                f.create_dataset(file[:-4]+'_tw.wav', data=time_warped)
                f.create_dataset(file[:-4]+'_fm.wav', data=freq_masked)
                f.create_dataset(file[:-4]+'_tm.wav', data=time_masked)
                count=count+1
                print(count)
            except:
                continue
        
    
#save as csv
df2.to_csv('aug_vocalsound.csv',index=False)

# read that csv and split into train and test sets
temp=np.array(pd.read_csv("aug_vocalsound.csv"))
array=temp[:,0]
label=temp[:,1]
X_train, X_test, y_train, y_test = train_test_split( array, label, test_size=0.3, random_state=42)

df_train=pd.DataFrame(data=np.column_stack((X_train,y_train)))
df_test=pd.DataFrame(data=np.column_stack((X_test,y_test)))


df_train.to_csv('aug_vocalsoundTrainFile.csv',index=False)
df_test.to_csv('aug_vocalsoundTestFile.csv',index=False)
        
       
