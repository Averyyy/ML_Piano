import os
import glob
import numpy as np
import torch
import torch.utils.data as data
import torchaudio

torchaudio.set_audio_backend('soundfile')

class PianoAudioDataset(data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.filepaths = glob.glob(self.data_dir + '\*.wav')
        self.transforms = torchaudio.transforms.MelSpectrogram(
            sample_rate=44100, n_fft=2048, hop_length=512, n_mels=128)

    def __getitem__(self, index):
        filepath = self.filepaths[index]
        waveform, sr = torchaudio.load(filepath)
        mel_spec = self.transforms(waveform).squeeze(0).transpose(0, 1)

        # Extract MIDI pitch and velocity from the filename
        filename = os.path.basename(filepath)
        pitch_str, velocity_str = filename.split('_')[2:]
        pitch = int(pitch_str.split('=')[1])-21
        velocity = int(velocity_str.split('=')[1].split('.')[0])

        return mel_spec, pitch, velocity


    def __len__(self):
        return len(self.filepaths)


if __name__ == '__main__':
    data_dir = 'data/train'
    dataset = PianoAudioDataset(data_dir)
    dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    for i, batch in enumerate(dataloader):
        print('Batch shape:', batch)
        
    print('Done!')
