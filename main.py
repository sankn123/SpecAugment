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
parser.add_argument('--dir', default='/home/guest1/DCASE Metadata/SpecAugment/data_44k', help='path to dataset/dir to look for files')
parser.add_argument('--policy', default='LD', help='augmentation policies - LB, LD, SM, SS')

args = parser.parse_args()
def spec_to_image(spec, eps=1e-6):
  mean = spec.mean()
  std = spec.std()
  spec_norm = (spec - mean) / (std + eps)
  spec_min, spec_max = spec_norm.min(), spec_norm.max()
  spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
  spec_scaled = spec_scaled.astype(np.uint8)
  return spec_scaled

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
    count1=0
    count=0
    top_db=80
    with h5py.File('/home/guest1/DCASE Metadata/aug_vocalsound.hdf5', 'w') as f:
        for file in os.listdir(training_files):
            count1=count1+1
            print(count1)
            print(file)
            file_path=os.path.join(args.dir,file)
            # print(file_path)
            # break
            
            # Load the audio file

            # audio, sr = librosa.load(file_path)
            try:
                
                audio, sr = librosa.load(file_path)
                # print("audio extracted")
                
                # Extract Mel Spectrogram Features from the audio file
                if audio.shape[0]<5*sr:
                    audio=np.pad(audio,int(np.ceil((5*sr-audio.shape[0])/2)),mode='reflect')
                else:
                    audio=audio[:5*sr]
                mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr,n_fft=2048, hop_length=512,n_mels=128, fmin=20, fmax=8300)
                apply = SpecAugment(mel_spectrogram, args.policy)
                
                time_warped = apply.time_warp() # Applies Time Warping to the mel spectrogram
                freq_masked = apply.freq_mask() # Applies Frequency Masking to the mel spectrogram
                time_masked = apply.time_mask() # Applies Time Masking to the mel spectrogram
                Label=my_dict[file[8:-4]]
            
                # new_rows = [{0: file ,1:Label},
                #         {0: file[:-4]+'_tw.wav' ,1:Label},
                #         {0: file[:-4]+'_fm.wav' ,1:Label},
                #         {0: file[:-4]+'_tm.wav' ,1:Label}]
                new_rows = [{0: file ,1:Label},
                        {0: file[:-4]+'_fm.wav' ,1:Label},
                        {0: file[:-4]+'_tm.wav' ,1:Label}]
                df2 = pd.concat([df2, pd.DataFrame(new_rows)], ignore_index=True)

                spec_db=librosa.power_to_db(mel_spectrogram,top_db=top_db)
                # spec_db_tw=librosa.power_to_db(time_warped,top_db=top_db)
                spec_db_fm=librosa.power_to_db(freq_masked,top_db=top_db)
                spec_db_tm=librosa.power_to_db(time_masked,top_db=top_db)

                feature=spec_to_image(spec_db)
                # feature_tw=spec_to_image(spec_db_tw)
                feature_fm=spec_to_image(spec_db_fm)
                feature_tm=spec_to_image(spec_db_tm)

                # feature_tw=np.squeeze(feature_tw)
                feature_fm=np.squeeze(feature_fm)
                feature_tm=np.squeeze(feature_tm)

                print(feature.shape,feature_fm.shape,feature_tm.shape)
            
            
                f.create_dataset(file, data=feature)# Store the spectrogram of a signal into h5py database and indexed with a signal name(file variable here)
                # f.create_dataset(file[:-4]+'_tw.wav', data=feature_tw)
                f.create_dataset(file[:-4]+'_fm.wav', data=feature_fm)
                f.create_dataset(file[:-4]+'_tm.wav', data=feature_tm)
                
                count=count+1
                print(count)
                if count==120:
                    break
            
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


df_train.to_csv('/home/guest1/DCASE Metadata/aug_vocalsoundTrainFile.csv',index=False)
df_test.to_csv('/home/guest1/DCASE Metadata/aug_vocalsoundTestFile.csv',index=False)
        
       
