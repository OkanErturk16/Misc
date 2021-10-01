#%%
import numpy as np
from commpy.modulation import QAMModem, PSKModem
from commpy.filters import rrcosfilter
import matplotlib.pyplot as plt
import time
import scipy.signal as signal
from scipy.interpolate import interp1d
#%%

num_of_symbols = 50  # Number of symbols
numb_of_preamble = 100
# Create modulation. QAM16 makes 4 bits/symbol
zero_padding = np.zeros(20, dtype=np.complex64)
#barker = np.array([1,1,1,-1,-1,-1,1,-1,-1,1,-1], dtype=np.complex64)
barker = np.array([1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,0,1,-1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,1,1,1], dtype=np.complex64)
barker = np.tile(barker,1)
len_barker = len(barker)
alternate = np.zeros(numb_of_preamble, dtype= np.complex64)
for ii in range(len(alternate)):
    alternate[ii] = ii%2
mod_order = 16
snr = 0.5
mod1 = PSKModem(mod_order)
df   = 0.0
# Generate the bit stream for N symbols
sB = np.random.randint(0, 2, num_of_symbols*int(np.log2(mod_order)))

sQ = mod1.modulate(sB)
#%%
print(np.mean(np.abs(sQ)**2))

sQ = np.concatenate((zero_padding,barker,sQ))
sQ = sQ*np.exp(1j*np.pi*2*np.arange(0,len(sQ))*df)
noise = 1/np.sqrt(2.0*snr)*(np.random.randn(len(sQ)) + 1j*np.random.randn(len(sQ)))
sQ = sQ + noise

x_corr = np.correlate(sQ,np.conj(barker),'full')
plt.plot(np.abs(x_corr))
# %%

# %%
