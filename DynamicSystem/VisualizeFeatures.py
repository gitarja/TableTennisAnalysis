import pandas as pd
from Conf import single_results_path, results_path, double_results_path, double_summary_path, single_summary_path
from Conf import x_double_features_all_column
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.signal import hilbert
from sklearn.preprocessing import MinMaxScaler
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot
import nolds
from scipy import signal as s
import random
from scipy.signal import butter, filtfilt
from statsmodels.graphics.tsaplots import plot_acf

sns.set_theme()

def movingAverage(signal):
    # Design a Butterworth low-pass filter
    order = 4  # Filter order
    cutoff_freq = 25  # Cutoff frequency (adjust as needed)
    sampling_freq = 1 / 0.01  # Sampling frequency
    nyquist_freq = 0.5 * sampling_freq
    cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, cutoff, btype='low')

    # Apply the filter to remove noise
    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal
def computePhaseSync(v1, v2):
    v1_analytic = hilbert(v1)
    v2_analytic = hilbert(v2)
    # v1_phase = np.arctan(v1_analytic.imag / v1_analytic.real)
    # v2_phase = np.arctan(v2_analytic.imag / v2_analytic.real)
    v1_phase = np.angle(v1_analytic)
    v2_phase = np.angle(v2_analytic)

    phase_diff = np.exp(1j * (v1_phase - v2_phase))
    # phase_diff = np.unwrap(v1_phase - v2_phase)

    avg_phase_diff = np.abs(np.average(phase_diff))

    # print("phase_diff: "+ str(avg_phase_diff))

    return avg_phase_diff

def generate_periodic_signal(num_samples):
    # Example usage
    period = 30.0  # Period of the signal (in seconds)
    amplitude = 1.0  # Amplitude of the signal
    phase_shift = np.pi / 2  # Phase shift of the signal (in radians)
    """
    Generate a periodic signal using a sine function.

    Parameters:
        period (float): Period of the signal.
        amplitude (float): Amplitude of the signal.
        phase_shift (float): Phase shift of the signal (in radians).
        num_samples (int): Number of samples to generate.

    Returns:
        numpy.ndarray: Array containing the generated signal.
    """
    # Generate a sequence of time points
    t = np.linspace(0, 2 * np.pi * period, num_samples)

    # Generate the periodic signal using sine function
    signal = amplitude * np.sin(t + phase_shift)

    return signal

def generateLogisticFunction(n = 100, x0=0):
    def sine_logistic_map(r, x):
        """
        Compute the sine logistic map for a given parameter 'r' and input 'x'.
        """
        return r * np.sin(np.pi * x)
    def logistic_map(r, x):

        return  (r * x * (1 - x))

    def exponential_logistic_map(r, x):
        """
        Compute the exponential logistic map for a given parameter 'r' and input 'x'.
        """
        return r * x * np.exp(1 - x)

    def piecewise_linear_logistic_map(r, x):
        """
        Compute the piecewise linear logistic map for a given parameter 'r' and input 'x'.
        """
        if 0 <= x < 0.5:
            return r * np.sin(np.pi * x)
            # return r * x
        elif 0.5 <= x <= 0.75:
            return 2 * r * (np.sin(np.pi * x) - 0.5)
        elif 0.75 < x <= 1:
            return r * (1 - np.sin(np.pi * x))

    # Parameters
    r = 0.95  # Control parameter (try different values for different behavior)
    num_iterations = n-1

    # Generate chaotic signal
    chaotic_signal = [x0]
    chaotic_signal_new = [x0]
    for _ in range(num_iterations):
        alpha = random.gauss(0, 0.05)
        x_next = piecewise_linear_logistic_map(r, chaotic_signal[-1])
        chaotic_signal.append(x_next)
        chaotic_signal_new.append(x_next + alpha)

    chaotic_signal = np.asarray(chaotic_signal)
    chaotic_signal_new = np.asarray(chaotic_signal_new)
    return chaotic_signal





df_summary = pd.read_csv(single_summary_path)
df_summary = df_summary[(df_summary["Tobii_percentage"] > 65)]

df = pd.read_pickle(results_path + "single_episode_features.pkl")


df = df.loc[
    (df["skill_subject"] > 0.55) & (df["success"] != -1) & (df["id_subject"].isin(df_summary["Subject1"].values))]

selected_groups = df.groupby(["id_subject", "episode_label"]).filter(lambda x: len(x) > 50)

grouped_episodes = selected_groups.groupby(["id_subject", "episode_label"])

rp = RecurrencePlot(threshold=np.pi/16)
for i, g in grouped_episodes:



    # Generate a random signal
    g = g.sort_values(by=['observation_label'])
    signal = g["pr_p1_al_miDo"].values
    if np.sum(np.isnan(signal)) <= 0:
        plt.figure(figsize=(12, 4))
        # signal = np.random.normal(loc=0, scale=1, size=len(signal))
        signal = signal / np.max(signal)
        random_signal =  generateLogisticFunction(len(signal), signal[0])

        # random_signal = np.random.normal(loc=0, scale=1., size=len(signal))


        random_signal = random_signal /  np.nanmax(random_signal)



        print(computePhaseSync(signal, random_signal))
        print(np.correlate(signal, random_signal) / len(signal))
        random_phase = hilbert(random_signal)
        phase = hilbert(signal)
        X_rp_random = rp.transform(np.expand_dims(random_signal, 0))
        X_rp = rp.transform(np.expand_dims(signal, 0))

        plt.subplot(2, 2, 1)
        plt.plot(random_signal)
        # plt.plot(np.abs(random_phase))
        # plt.plot(random_signal)

        plt.subplot(2, 2, 2)
        # plt.plot(random_signal)
        frequencies = np.fft.fftfreq(len(random_signal), 1 / len(random_signal))
        psd = np.abs(np.fft.fft(random_signal)) ** 2
        # Plot the power spectral density
        plt.plot(frequencies[:len(random_signal) // 2], psd[:len(random_signal) // 2])

        plt.subplot(2, 2, 3)
        plt.plot(signal)
        plt.subplot(2, 2, 4)
        frequencies = np.fft.fftfreq(len(signal), 1 / len(signal))
        psd = np.abs(np.fft.fft(signal)) ** 2
        # Plot the power spectral density
        plt.plot(frequencies[:len(signal) // 2], psd[:len(signal) // 2])


        # plt.plot(np.abs(phase))
        # plt.subplot(2, 2, 3)
        # plt.plot(np.angle(phase))
        # plt.plot(np.real(phase), np.imag(phase))
        # plt.imshow(X_rp[0], cmap='binary', origin='lower',
        #            extent=[0, 4 * np.pi, 0, 4 * np.pi])
        # plt.subplot(2, 2, 4)
        # plt.plot(np.angle(random_phase))
        # plt.plot(np.real(random_phase), np.imag(random_phase))
        # plt.imshow(X_rp_random[0], cmap='binary', origin='lower',
        #          extent=[0, 4 * np.pi, 0, 4 * np.pi])

        # compute Lypanov

        # N_data = int(len(signal)//2)

        # print(nolds.lyap_e(random_signal, emb_dim=21, matrix_dim=5))
        # print(nolds.hurst_rs(random_signal))

        #
        plt.show()
        plt.close()
