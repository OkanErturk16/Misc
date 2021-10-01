#%%
import numpy as np
from commpy.modulation import QAMModem
from commpy.filters import rrcosfilter
import matplotlib.pyplot as plt
import time
import scipy.signal as signal
from scipy.interpolate import interp1d
#%%

num_of_symbols = 64  # Number of symbols
# Create modulation. QAM16 makes 4 bits/symbol
N = 8
T = 1/N
mod_order = 4
snr = 3
mod1 = QAMModem(mod_order)
sPSF = np.ones(N,dtype=np.complex64)
df   = 0.00000
# Generate the bit stream for N symbols
sB = np.random.randint(0, 2, num_of_symbols*int(np.log2(mod_order)))
# for ii in range(len(sB)):
#     sB[ii] = ii%2
# Generate N complex-integer valued symbols

sQ = mod1.modulate(sB)
sQ_os = np.zeros((len(sQ)-1)*N + 1, dtype=np.complex64)
sQ_os[::N] =sQ
 
qW = np.convolve(sPSF, sQ_os)
qW = qW*np.exp(1j*np.pi*2*np.arange(0,len(qW))*df)
noise = 1/np.sqrt(2.0*snr)*(np.random.randn(len(qW)) + 1j*np.random.randn(len(qW)))
qW = qW + noise
rk = np.convolve(sPSF, qW)
 
# plt.plot(np.real(rk))
# plt.plot(np.imag(rk))
    
rk_shift = np.roll(rk,1)
len_rk = len(rk_shift)
xk = np.abs(rk_shift)**2
k  = np.arange(0,len(xk))
Xm = -1/(2*np.pi)*np.angle(np.sum(xk*np.exp(-1j*2.0*np.pi*k/N)))
eps = (Xm*N)%N
print('EPS:',eps)
t_shift = np.arange(1,num_of_symbols-1)*N + eps
t       = np.arange(len(rk))
f2_real = interp1d(t, rk_shift.real, kind='cubic')
qW_real = f2_real(t_shift)
f2_imag = interp1d(t, rk_shift.imag, kind='cubic')
qW_imag = f2_imag(t_shift)
sample_points = qW_real + 1j*qW_imag
# plt.plot(t_shift,np.real(sample_points),'*')
# plt.plot(t,np.real(rk_shift))
plt.plot(sample_points.real, sample_points.imag,'*')
# %%
import cupy as cp
a = cp.zeros(int(1e6))
b = cp.zeros(int(1e6))
%timeit cp.concatenate((a,b))
# %%
