from mat4py import loadmat

import math
import numpy as np

import matplotlib.pyplot as plt
import plotly.express as px

import scipy.signal as signal
import scipy.stats as stats

"""
Load file “spiketrain1.mat” into matlab. You’ll see two 
variables “timeaxis” and “Vm”. “Vm” is the membrane 
potential of a pyramidal cell obtained from an intracellular 
sharp electrode recording. “timeaxis”gives the times 
(in sec) at which the values of Vm where sampled. 
"""
class AnalysisOne():
    def __init__(self) -> None:
        # Part I A
        print("Loading .mat files...")
        self.st1 = loadmat("spiketrain1.mat")
        self.st2 = loadmat("spiketrain2.mat")
        self.st1_spikes = self.spike_indices(self.st1, -13)
        self.st1_bins = []
        self.spike_interval = self.st1['timeaxis'][1] - self.st1['timeaxis'][0]
        self.st1_bins.append(self.calculate_binary_bins(self.st1_spikes, self.spike_interval, 0.001))
        self.st1_bins.append(self.calculate_binary_bins(self.st1_spikes, self.spike_interval, 0.0005))
        self.st1_bins.append(self.calculate_binary_bins(self.st1_spikes, self.spike_interval, 0.0001))

        # Part I B
        self.isi = self.calculate_interspike_intervals(self.st1_spikes, self.spike_interval)

    def spike_indices(self, st, threshold):
        index = 0
        spike_indices_new = []
        for i in range(len(st['Vm'])):
            if st['Vm'][i] > threshold and st['Vm'][i-1] < threshold:
                spike_indices_new.append(i)

        return spike_indices_new
    
    def calculate_binary_bins(self, spikes_indices, spike_interval, bin_size):
        total_time = self.st1['timeaxis'][-1]
        bin_count = math.ceil(total_time / bin_size) 
        binary_bins = [0] * bin_count
        count = 0
        for si in spikes_indices:
            time = si * spike_interval
            index = math.floor(time / bin_size)

            binary_bins[index] = 1 / bin_size
        
        return binary_bins
    
    # Compute the interspike interval sequence from the spike times
    def calculate_interspike_intervals(self, spikes_indices, spike_interval):
        interspike_intervals = []
        for i in range(len(spikes_indices) - 1):
            interspike_intervals.append((spikes_indices[i+1] - spikes_indices[i]) * spike_interval)
        
        return interspike_intervals
    
    # Compute the interspike interval histogram with a binwidth 
    # of 1 msec going up to 200 ms
    def isi_histogram(self):
        # include only the first 200 ms
        total = 0
        for i in range(len(self.isi)):
            total += self.isi[i]
            if total > 200:
                self.isi = self.isi[:i] 
                break

        plt.figure()
        plt.hist(self.isi, bins=200)
        plt.xlabel("ISI (ms)")
        plt.ylabel("Count")
        plt.title("ISI Histogram")
        plt.show()

        return

    # Coefficient of Variation
    # cv = std / mean
    def compute_cv(self):
        import numpy as np
        return np.std(self.isi) / np.mean(self.isi)
    
    # Compute the interspike interval correlation coefficients
    # rho = (<ISI_i * ISI_(i+j)> - <ISI_i>^2) / <ISI_i^2> - <ISI_i>^2
    def norm_isi_correlation_coefficients(self, j):
        # rho = []
        # rho.append(1)
        # isi_mean = np.mean(self.isi)
        # isi = [0] * len(self.isi)
        # for i in range(1,len(self.isi)):
        #     isi[i] = self.isi[i] - isi_mean
        # for i in range(1,len(isi) - j):
        #     isi_i = isi[i]
        #     isi_i_j = isi[i+j]
        #     numerator = np.mean(isi_i * isi_i_j) - np.mean(isi_i) ** 2
        #     denominator = np.mean(isi_i ** 2) - np.mean(isi_i) ** 2
        #     if denominator == 0:
        #         rho.append(0)
        #     else:
        #         rho.append(numerator / denominator)

        """
        Compute the autocorrelation of the signal `x` with a maximum lag of `max_lag`.
        The result is normalized to get correlation coefficients.
        """
        isi = np.array(self.isi) - np.mean(self.isi)  # Subtract the mean
        result = np.correlate(isi, isi, mode='full')  # Full autocorrelation
        max_corr = result[result.size // 2]  # Maximum correlation at zero lag
        result = result[result.size // 2 - j : result.size // 2 + j + 1] / max_corr  # Normalize
        lags = np.arange(-j, j + 1)

        # only include the positive values
        result = result[j+1:]
        lags = lags[j+1:]

        # plot the correlation coefficients
        plt.figure()
        plt.plot(lags, result)
        plt.xlabel("Lag (ms)")
        plt.ylabel("Correlation Coefficient")
        plt.title("ISI Correlation Coefficients")
        plt.show()

        return
    
    # A(t, tau) = <X(t) * X(t+tau)> - <X(t)>*<X(t+tau)>
    def auto_correlation(self,dis_time_series, tau):
        sum = 0
        if tau == 0:
            return np.mean(np.array(dis_time_series) * np.array(dis_time_series)) - np.mean(dis_time_series) ** 2
        else:
            first_series = dis_time_series[:-tau]
            second_series = dis_time_series[tau:]
            sum += np.mean(np.array(first_series) * np.array(second_series))
            sum -= np.mean(np.array(dis_time_series[:-tau])) * np.mean(np.array(dis_time_series[tau:]))
            return sum


    # A(t, tau) = <X(t) * X(t+tau)> - <X(t)>^2
    def stationary_auto_correlation(self,dis_time_series, tau):
        sum = 0
        sum += np.mean(dis_time_series * dis_time_series[tau:])
        sum -= np.mean(dis_time_series) ** 2
        return sum
    
    # def autocorr(self,x):
    #     result = np.correlate(x, x, mode='full')
    #     return result[result.size // 2:]
    
    def compute_ac(self, dis_time_series):
        ac = []
        # for tau in range(len(dis_time_series)):
        for tau in range(200): # probably unlikely that the correlation will be significantly after 100
            ac.append(self.auto_correlation(dis_time_series, tau))
        return ac
    
    def compute_ac_bins(self):
        ac_bins = []
        for i in range(len(self.st1_bins)):
            ac_bins.append(self.compute_ac(self.st1_bins[i]))
        return ac_bins
    

    # power spectra at 2kHz
    # python equivalent of matlab's pwelch
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html
    # https://www.mathworks.com/help/signal/ref/pwelch.html#btuf68p_sep_shared-fs
    def compute_ps_bins(self):
        ps_bins = []
        for i in range(len(self.st1_bins)):# compute the power spectrum for each bin
            ps_bins.append(self.compute_power_spectrum(self.st1_bins[i])) 
        return ps_bins
    
    def compute_power_spectrum(self, dis_time_series):
        results = signal.welch(dis_time_series, 
                            window=signal.get_window(window="bartlett", Nx=2048, fftbins=False), 
                            noverlap=1024, 
                            return_onesided=True, # return only the positive frequencies
                            fs=2000) # sampling frequency
        
        for i in range(len(results[1])):
            results[1][i] /= 2 # divide by 2 because we are using a real signal
        return results
    
    def simulate_poisson_spike_train(self, firing_rate, total_time):
        """
        Simulate a Poisson spike train given a neuron's firing rate and duration.
        """
        num_steps = int(total_time / self.spike_interval)
        spike_train = np.random.poisson(firing_rate * self.spike_interval, num_steps)
        return spike_train
    
    def compute_poisson_firing(self, firing_rate):
        """
        Compute the power spectrum of a Poisson spike train given a neuron's firing rate.
        """
        spike_train = self.simulate_poisson_spike_train(firing_rate, self.st1['timeaxis'][-1])
        return self.compute_power_spectrum(spike_train)
    

"""
Part II
Load spiketrain2.mat and you’ll see that the workspace now contains a 
variable called “data”. A given neuron has been stimulated with 
“frozen noise”. I.e. the stimulus has been repeated many times and 
each epoch lasts 50 sec. The stimulus is sampled at 2kHz and is the 
variable “stim” in the workspace.The stimulus is Gaussian white 
noise that has been low-pass filtered (8th order butterworth) at a 
cutoff frequency of 30 Hz, its power spectrum should be flat up to 
about 20Hz. 
  
The first column contains the trial indices, the second 
column contains the spike times (in msec), and the third column 
contains the interspike intervals (in msec). 
"""
class AnalysisTwo():
    def __init__(self) -> None:
        print("Loading .mat files...")
        # self.st1 = loadmat("lecture2/spiketrain1.mat")
        self.st2 = loadmat("spiketrain2.mat")
        # fix the double-listing error
        for i in range(len(self.st2['stim'])):
            self.st2['stim'][i] = self.st2['stim'][i][0]
        print()

    def plot_raster(self):
        plt.figure()
        # get the second column of st2['data'] which is the spike times
        spike_times = [row[1] for row in self.st2['data']]
        # get the first column of st2['data'] which is the trial indices
        trial_indices = [int(row[0]) for row in self.st2['data']]
        # remove the last 5000 elements in the spike times and trial indices
        plt.plot(spike_times, trial_indices, '.')
        plt.xlabel("Time (ms)")
        plt.ylabel("Trial Index")
        plt.title("Raster Plot")
        # shrink the space between the integer ticks of the y-axis
        # plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.gca().set_aspect(10)
        # zoom in so that only time 20000 to 22000 is shown
        plt.xlim(20000, 20400)
        plt.ylim(30,70)
        plt.show()

        return
    
    # find the largest time in the spike train
    def find_largest_time(self):
        largest = 0
        for i in range(len(self.st2['data'])):
            if self.st2['data'][i][1] > largest:
                largest = self.st2['data'][i][1]
        return largest
    
    def find_epoch_count(self):
        epoch_count = 0
        for i in range(len(self.st2['data'])):
            if self.st2['data'][i][0] > epoch_count:
                epoch_count = self.st2['data'][i][0]
        return epoch_count
    
    """
    1. Build a PSTH from the data with binwidth 1 msec. To do this, you 
    have to make a vector of appropriate length (i.e. the length of one 
    epoch divided by the binwidth) and count the number of spikes in 
    each bin. Then divide that number by the number of epochs and by 
    the binwidth to obtain the firing rate. 
    """
    def plot_psth(self):
        bin_widt = 1 # 1 ms
        epoch_count = self.find_epoch_count()
        # bins = length of an epoch / bin size
        # total time was 49999.97 so i just rounded up to 50000
        bins = [0] * int(50000 / bin_widt) 
        for i in range(len(self.st2['data'])):
            # get the spike time
            spike_time = self.st2['data'][i][1]
            # get the bin index
            bin_index = int(spike_time / bin_widt)
            # increment the bin
            bins[bin_index] += 1

        # divide by the number of epochs and bin width
        for i in range(len(bins)):
            bins[i] /= epoch_count * bin_widt

        self.psth_bins = bins

        plt.figure()
        plt.plot(bins)
        plt.xlabel("Time (ms)")
        plt.ylabel("Firing Rate (Hz)")
        plt.title("PSTH")
        plt.show()

        return bins
    
    # create a list of lists where each list contains the spike times for a given trial in ascending order
    def create_spike_bins(self):
        print("Creating spike bins which may take a while...")
        # put the spike times into bins belonging to the trial index
        spike_bins = []
        for index in range(1, 101):
            temp = []
            for item in self.st2['data']:
                if int(item[0]) == index:
                    # temp.append([item[1], item[2]])
                    temp.append(item[1])
            spike_bins.append(temp)
        
        # sort the spike bins so that the spikes in each bin are in ascending order
        for index in range(100):
            # sort the list of tuples by the first element in the tuple
            # spike_bins[index].sort(key=lambda x: x[0])

            # sort list
            spike_bins[index].sort()

        return spike_bins
    
    def calculate_binary_bins(self, spike_times, bin_width, total_time):
        binary_bins = [0] * math.ceil(total_time / bin_width)
        spike_index = 0
        time = 0
        stop = 50000
        while time < stop:
            # if the spike time is within the current time and the next bin
            if spike_times[spike_index] >= time and spike_times[spike_index] < time + bin_width:
                binary_bins[int(time / bin_width)] = 1 / bin_width
                spike_index += 1
                time += bin_width
                # if we have reached the end of the spike times, break
                if spike_index == len(spike_times):
                    break
            # if the spike time is greater than the current bin
            elif spike_times[spike_index] >= time + bin_width:
                time += bin_width                

        return binary_bins

    # bind width is in milliseconds
    def create_binary_bins(self, spike_bins, bin_width):
        binary_bins = []
        # for each of the 100 trials, create a binary representation of the spike train
        for index in range(100):
            binary_bins.append(self.calculate_binary_bins(spike_bins[index], bin_width, 50000))

        return binary_bins

    
    """
    1. Make a binary representation of each trial at 2kHz  
    2. Compute the cross-correlation function between trial 1 and the 
    stimulus. 
    3. Repeat for trial 20.
    4. Compute the cross-correlation function averaged across all trials.
    """
    def cross_correlation_calculations(self):
        # step 1
        self.spike_bins = self.create_spike_bins()
        # there need to be 100,000 bins
        self.bin_width = float(50000) / 100000
        self.binary_bins = self.create_binary_bins(self.spike_bins, self.bin_width)
        # step 2
        tr1_stim_cr_corr = signal.correlate(self.binary_bins[0], self.st2['stim'], mode="same")
        # step 3
        tr20_stim_cr_corr = signal.correlate(self.binary_bins[19], self.st2['stim'], mode="same")
        # step 4
        avg_stim_cr_corr = [0] * len(self.binary_bins[0])
        for i in range(len(self.binary_bins)):
            avg_stim_cr_corr += signal.correlate(self.binary_bins[i], self.st2['stim'], mode="same")
        avg_stim_cr_corr /= len(self.binary_bins)

        # plot the cross-correlation functions as a line 
        # the three cross-correlation functions should be on the same plot
        plt.figure()
        plt.plot(tr1_stim_cr_corr)
        plt.plot(tr20_stim_cr_corr)
        plt.plot(avg_stim_cr_corr)
        plt.xlabel("Bin Index")
        plt.ylabel("Cross-Correlation")
        plt.title("Cross-Correlation Functions")
        plt.legend(["Trial 1", "Trial 20", "Average"])
        plt.show()
        
        return

    """
    1. Compute the cross-spectrum between the stimulus and the spike 
    train from epoch 1. 
    2. Use the formula seen in class to estimate both 
    gain and phase as a function of frequency. 
    3. Normalize the gain by its value at f=1 Hz. 

    Remember: that the stimulus only has power up to about 30Hz.
    """
    def compute_cross_spectrum(self):
        cr_spec_f, cr_spec_Pxy = signal.csd(self.st2['stim'],
                                    self.binary_bins[0],
                                    window=signal.get_window(window="bartlett", Nx=2048, fftbins=False),
                                    noverlap=1024,
                                    return_onesided=True,
                                    fs=60)
        # convert to lists
        cr_spec_f_lst = cr_spec_f.tolist()
        cr_spec_Pxy_lst = cr_spec_Pxy.tolist()

        # cr_spec_Pxy_amp = []
        # for pair in cr_spec_Pxy_lst:
        #     # find absolute value,
        #     # divide by 2,
        #     cr_spec_Pxy_amp.append(np.abs(pair) / 2)

        for i in range(len(cr_spec_Pxy)):
            cr_spec_Pxy[i] /= 2

        auto_spec_f, auto_spec_Pxx = signal.welch(self.binary_bins[0],
                                    window=signal.get_window(window="bartlett", Nx=2048, fftbins=False),
                                    noverlap=1024,
                                    return_onesided=True,
                                    fs=60)
        for i in range(len(auto_spec_Pxx)):
            auto_spec_Pxx[i] /= 2

        # calculate the transfer function
        transfer_function = []
        for i in range(len(auto_spec_Pxx)):
            transfer_function.append(cr_spec_Pxy[i] / auto_spec_Pxx[i])

        # calculate the gain
        gain = []
        for i in range(len(transfer_function)):
            gain.append(np.abs(transfer_function[i]))

        # normalize the gain by its value at f=1 Hz
        self.normalized_gain = []
        for i in range(len(gain)):
            self.normalized_gain.append(gain[i] / gain[34])

        # calculate the phase
        self.phase = []
        for i in range(len(transfer_function)):
            self.phase.append(np.angle(transfer_function[i]))

        # plot the gain and phase as a function of frequency
        plt.figure()
        plt.plot(cr_spec_f, self.normalized_gain)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Gain")
        plt.title("Gain as a Function of Frequency")
        plt.show()

        plt.figure()
        plt.plot(cr_spec_f, self.phase)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Phase")
        plt.title("Phase as a Function of Frequency")
        plt.show()

        return
        
    """
    1. Take the average across trials of the binary vector and compute 
    its power spectrum. 
    2. Subtract the average binary vector from 
    each trial and compute the power spectrum. 
    3. Average these over the trials. 
    4. Compute the signal-to-noise ratio as a function of frequency. 
    5. Compare the shape of the gain and signal-to-noise ratio curves. 
    """
    def II_part_2_C_1(self):
        # step 1
        # create the averaged binary vector
        avg_bin_vec = np.array([0] * len(self.binary_bins[0]))
        for i in range(len(self.binary_bins)):
            avg_bin_vec = avg_bin_vec + np.array(self.binary_bins[i])
        avg_bin_vec /= len(self.binary_bins)
        # compute the power spectrum of the averaged binary vector
        spec_f, spec_Pxx = signal.welch(avg_bin_vec,
                                    window=signal.get_window(window="bartlett", Nx=2048, fftbins=False),
                                    noverlap=1024,
                                    return_onesided=True,
                                    fs=2000)
        
        # step 2
        # subtract the average binary vector from each trial
        # compute the power spectrum of each trial

        spec_f_lst, spec_Pxx_lst = [], []
        for i in range(len(self.binary_bins)):
            temp_spec_f, temp_spec_Pxx = signal.welch(self.binary_bins[i] - avg_bin_vec,
                                    window=signal.get_window(window="bartlett", Nx=2048, fftbins=False),
                                    noverlap=1024,
                                    return_onesided=True,
                                    fs=2000)
            spec_f_lst.append(temp_spec_f)
            spec_Pxx_lst.append(temp_spec_Pxx)

        # step 3
        # average the power spectra over the trials
        avg_spec_Pxx = np.array([0] * len(spec_Pxx_lst[0]))
        for i in range(len(spec_Pxx_lst)):
            avg_spec_Pxx = avg_spec_Pxx + spec_Pxx_lst[i]
        avg_spec_Pxx /= len(spec_Pxx_lst)

        # step 4
        # compute the signal-to-noise ratio as a function of frequency
        snr = []
        for i in range(len(avg_spec_Pxx)):
            snr.append(spec_Pxx[i] / avg_spec_Pxx[i])

        # step 5
        # compare the shape of the gain and signal-to-noise ratio curves
        plt.figure()
        plt.plot(spec_f, snr)
        plt.plot(spec_f, self.normalized_gain)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power")
        plt.title("Signal-to-Noise Ratio and Gain")
        plt.legend(["Signal-to-Noise Ratio", "Gain"])
        plt.show()

        return



        


if __name__ == "__main__":
    # parts I_A_1, I_A_2, I_A_3, I_B_1 were completed in the AnalysisOne init function
    A_ONE = AnalysisOne()
    # part I_B_2
    A_ONE.isi_histogram()

    # part I_B_3
    print("CV is: " + str(A_ONE.compute_cv()))
    
    # part I_B_4
    A_ONE.norm_isi_correlation_coefficients(200)

    print("Calculating AC bins which may take a while...")

    # part I_C_1 and I_C_2
    ac_bins = A_ONE.compute_ac_bins()
    ps_bins = A_ONE.compute_ps_bins()
    ps_bins.append(A_ONE.compute_poisson_firing(15.6))

    # plot all three bins for the power spectra
    plt.figure()
    plt.plot(ps_bins[0][0], ps_bins[0][1])
    plt.plot(ps_bins[1][0], ps_bins[1][1])
    plt.plot(ps_bins[2][0], ps_bins[2][1])
    plt.plot(ps_bins[3][0], ps_bins[3][1])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (Hz)")
    plt.title("Power Spectra")
    plt.legend(["1 ms", "0.5 ms", "0.1 ms", "Poisson"])
    plt.show()

    A_TWO = AnalysisTwo()

    # part II_A_1
    A_TWO.plot_raster()
    # part II_A_2
    A_TWO.plot_psth()
    # part II_B_1
    A_TWO.cross_correlation_calculations()
    # part II_B_2
    A_TWO.compute_cross_spectrum()
    # part II_C_1
    A_TWO.II_part_2_C_1()


   
