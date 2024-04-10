import os

import pandas as pd
import numpy as np
import numpy.matlib as matlib
import scipy
import scipy.signal
import scipy.stats
import scipy.io.wavfile
import scipy.fftpack

from pynwb import NWBHDF5IO
from algobci.decode import NeuralPreprocessWithHilbert

import MelFilterBank as mel


def extractHG(data, sr, windowLength=0.05, frameshift=0.01):
    """
    Window data and extract frequency-band envelope using the hilbert transform

    Parameters
    ----------
    data: array (samples, channels)
        EEG time series
    sr: int
        Sampling rate of the data
    windowLength: float
        Length of window (in seconds) in which spectrogram will be calculated
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    Returns
    ----------
    feat: array (windows, channels)
        Frequency-band feature matrix
    """
    N, C = data.shape

    fpts = windowLength * sr
    foffset = frameshift * sr

    # Number of points of each input
    ninp = int(np.floor(fpts))
    # Number of windows
    nwin = int(np.floor((N - fpts) / foffset))

    preprocessor = NeuralPreprocessWithHilbert(
        ninp, sr, [[70, 170]], 4, [100, 150]
    )

    feat = np.zeros((nwin, C))

    for i in range(nwin):
        nbeg = int(np.floor(i * foffset))
        nend = int(np.floor((i + 1) * foffset))
        y = preprocessor.apply(data[nbeg:nend], car="mean")
        feat[i, :] = y.T
    return feat


def stackFeatures(features, modelOrder=4, stepSize=5):
    """
    Add temporal context to each window by stacking neighboring feature vectors

    Parameters
    ----------
    features: array (windows, channels)
        Feature time series
    modelOrder: int
        Number of temporal context to include prior to and after current window
    stepSize: float
        Number of temporal context to skip for each next context (to compensate
        for frameshift)
    Returns
    ----------
    featStacked: array (windows, feat*(2*modelOrder+1))
        Stacked feature matrix
    """
    N, C = features.shape

    m = modelOrder * stepSize

    featStacked = np.zeros(
        (N - (2 * m), (2 * modelOrder + 1) * C)
    )

    for fNum, i in enumerate(range(m, N - m)):
        ef = features[i - m:i + m + 1:stepSize, :]
        # Add 'F' if stacked the same as matlab
        featStacked[fNum, :] = ef.flatten()
    return featStacked


def downsampleLabels(labels, sr, windowLength=0.05, frameshift=0.01):
    """
    Downsamples non-numerical data by using the mode

    Parameters
    ----------
    labels: array of str
        Label time series
    sr: int
        Sampling rate of the data
    windowLength: float
        Length of window (in seconds) in which mode will be used
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    Returns
    ----------
    newLabels: array of str
        Downsampled labels
    """
    npts = windowLength * sr

    nwins = int(np.floor((labels.shape[0] - npts) / (frameshift * sr)))
    newLabels = np.empty(nwins, dtype="S15")
    for w in range(nwins):
        start = int(np.floor((w * frameshift) * sr))
        stop = int(np.floor(start + npts))
        label = np.unique(labels[start:stop])[0]
        newLabels[w] = label.encode("ascii", errors="ignore").decode()
    return newLabels


def extractMelSpecs(audio, sr, windowLength=0.05, frameshift=0.01):
    """
    Extract logarithmic mel-scaled spectrogram, traditionally used to compress
    audio spectrograms

    Parameters
    ----------
    audio: array
        Audio time series
    sr: int
        Sampling rate of the audio
    windowLength: float
        Length of window (in seconds) in which spectrogram will be calculated
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    numFilter: int
        Number of triangular filters in the mel filterbank
    Returns
    ----------
    spectrogram: array (nwins, numFilter)
        Logarithmic mel scaled spectrogram
    """
    npts = windowLength * sr

    nwins = int(np.floor((audio.shape[0] - npts) / (frameshift * sr)))
    win = scipy.signal.windows.hann(int(np.floor(npts + 1)))[:-1]
    spectrogram = np.zeros((nwins, int(np.floor(npts / 2 + 1))),
                           dtype='complex')
    for w in range(nwins):
        start_audio = int(np.floor((w * frameshift) * sr))
        stop_audio = int(np.floor(start_audio + npts))
        a = audio[start_audio:stop_audio]
        spec = np.fft.rfft(win * a)
        spectrogram[w, :] = spec
    mfb = mel.MelFilterBank(spectrogram.shape[1], 23, sr)
    spectrogram = np.abs(spectrogram)
    spectrogram = (mfb.toLogMels(spectrogram)).astype('float')
    return spectrogram


def nameVector(elecs, modelOrder=4):
    """
    Creates list of electrode names

    Parameters
    ----------
    elecs: array of str
        Original electrode names
    modelOrder: int
        Temporal context stacked prior and after current window
        Will be added as
        T - modelOrder, T - (modelOrder + 1), ..., T0, ..., T + modelOrder
        to the elctrode names
    Returns
    ----------
    names: array of str
        List of electrodes including contexts, will have size
        elecs.shape[0] * (2 * modelOrder + 1)
    """
    names = matlib.repmat(elecs.astype(np.dtype(('U', 10))), 1,
                          2 * modelOrder + 1).T
    for i, off in enumerate(range(-modelOrder, modelOrder + 1)):
        names[i, :] = [e[0] + 'T' + str(off) for e in elecs]
    return names.flatten()  # Add 'F' if stacked the same as matlab


if __name__ == "__main__":
    winL = 0.05
    frameshift = 0.01
    modelOrder = 4
    stepSize = 5
    datapath = r'./SingleWordProductionDutch-iBIDS'
    featpath = r'./features'
    participants = pd.read_csv(os.path.join(datapath, 'participants.tsv'),
                               delimiter='\t')

    for p_id, participant in enumerate(participants['participant_id']):
        # Load data
        fname = f'{participant}_task-wordProduction_ieeg.nwb'
        io = NWBHDF5IO(os.path.join(datapath, participant, 'ieeg', fname), 'r')
        nwbfile = io.read()
        # sEEG
        eeg = nwbfile.acquisition['iEEG'].data[:]
        eeg_sr = 1024
        # audio
        audio = nwbfile.acquisition['Audio'].data[:]
        audio_sr = 48000
        # words (markers)
        words = nwbfile.acquisition['Stimulus'].data[:]
        words = np.array(words, dtype=str)
        io.close()
        # channels
        fname = f'{participant}_task-wordProduction_channels.tsv'
        fpath = os.path.join(datapath, participant, 'ieeg', fname)
        channels = pd.read_csv(fpath, delimiter='\t')
        channels = np.array(channels['name'])

        # Extract HG features
        feat = extractHG(eeg, eeg_sr, windowLength=winL, frameshift=frameshift)

        # Stack features
        feat = stackFeatures(feat, modelOrder=modelOrder, stepSize=stepSize)

        # Process Audio
        target_SR = 16000
        audio = scipy.signal.decimate(audio, int(audio_sr / target_SR))
        audio_sr = target_SR
        scaled = np.int16(audio/np.max(np.abs(audio)) * 32767)
        os.makedirs(os.path.join(featpath), exist_ok=True)
        fname = f'{participant}_orig_audio.wav'
        scipy.io.wavfile.write(os.path.join(featpath, fname), audio_sr, scaled)

        # Extract spectrogram
        melSpec = extractMelSpecs(scaled, audio_sr, windowLength=winL,
                                  frameshift=frameshift)

        # Align to EEG features
        words = downsampleLabels(words, eeg_sr, windowLength=winL,
                                 frameshift=frameshift)
        words = words[modelOrder*stepSize:words.shape[0]-modelOrder*stepSize]
        melSpec = melSpec[modelOrder*stepSize:melSpec.shape[0]-modelOrder*stepSize, :]      # noqa: E501
        # adjust length (differences might occur due to rounding in the number
        # of windows)
        if melSpec.shape[0] != feat.shape[0]:
            tLen = np.min([melSpec.shape[0], feat.shape[0]])
            melSpec = melSpec[:tLen, :]
            feat = feat[:tLen, :]

        # Create feature names by appending the temporal shift
        feat_names = nameVector(channels[:, None], modelOrder=modelOrder)

        # Save everything
        np.save(os.path.join(featpath, f'{participant}_feat.npy'), feat)
        np.save(os.path.join(featpath, f'{participant}_procWords.npy'), words)
        np.save(os.path.join(featpath, f'{participant}_spec.npy'), melSpec)
        np.save(os.path.join(featpath, f'{participant}_feat_names.npy'), feat_names)        # noqa: E501
