import os
import numpy as np
def createDir(directory_path):
    # Check if the directory already exists
    if not os.path.exists(directory_path):
        # Create the directory
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")

def computePowerSpectrum(x, fs=100):
    dt = 1 / fs
    T = len(x) * dt
    xf = np.fft.fft(x - x.mean())  # Compute Fourier transform of x
    Sxx = 2 * dt ** 2 / T * (xf * xf.conj())  # Compute spectrum
    Sxx = Sxx[:int(len(x) / 2)+1]  # Ignore negative frequencies
    p = Sxx.real

    f = np.fft.rfftfreq(len(x), d=1. / fs)

    return f, p

def movingAverage(x, n=5):
    def moveAverage(a):
        ret = np.cumsum(a.filled(0))
        ret[n:] = ret[n:] - ret[:-n]
        counts = np.cumsum(~a.mask)
        counts[n:] = counts[n:] - counts[:-n]
        ret[~a.mask] /= counts[~a.mask]
        ret[a.mask] = np.nan
        return ret

    mx = np.ma.masked_array(x, np.isnan(x))
    x = moveAverage(mx)

    return x