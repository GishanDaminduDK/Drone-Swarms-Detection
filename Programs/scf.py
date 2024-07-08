# derived from  https://github.com/avian2/spectrum-sensing-methods/blob/master/sensing/utils.py
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided
import torch



def sliding_window(x, w, s):
    """
    prepares the data matrix

    Args:
        x (numpy array): the input array
        w (int): window size
        s (int): stride 

    Returns:
        (numpy array): prepared input matrix

    """
    shape = (((x.shape[0] - w) // s + 1), w)
    strides = (x.strides[0]*s, x.strides[0])
    x_strided = as_strided(x, shape, strides)

    return x_strided


def scd_fam(x, Np, L, N=None, B=256):
    """
    Calculates the spectral correlation density

    Args:
        x (numpy array): the input array
        Np (int): Window size
        L (int): Seperation between window beginning 
        N (int): Size of input x (number of rows)
    """

    x = np.array(x)
    # input channelization
    xs = sliding_window(x, Np, L)

    if N is None:
        Pe = int(np.log2(xs.shape[0]))
        P = 2**Pe
        N = L*P
    else:
        P = N//L
    xs = xs[0:P, :].copy()

    # windowing
    w = np.hamming(Np)
    w /= np.sqrt(np.sum(w**2))
    xw = xs * np.tile(w, (P, 1))

    # first FFT
    XF1 = np.fft.fft(xw, axis=1)
    XF1 = np.fft.fftshift(XF1, axes=1)

    # calculating complex demodulates
    f = np.arange(Np)/Np - .5
    t = np.arange(P)*L

    f = np.tile(f, (P, 1))
    t = np.tile(t.reshape(P, 1), (1, Np))

    XD = XF1
    XD *= np.exp(-1j*2*np.pi*f*t)

    # calculating conjugate products, second FFT and the final matrix
    Sx = np.zeros((Np, 2*N), dtype=complex)
    Mp = N//Np//2

    for k in range(Np):
        for l in range(Np):
            XF2 = np.fft.fft(XD[:, k]*np.conjugate(XD[:, l]))
            XF2 = np.fft.fftshift(XF2)
            XF2 /= P

            i = (k+l) // 2  # spectral frequency component
            a = int(((k-l)/Np + 1) * N)
            Sx[i, a-Mp:a+Mp] = XF2[(P//2-Mp):(P//2+Mp)]

    f = np.absolute(Sx)
    my, mx = f.shape
    f = f[(my//2-B):(my//2+B), (mx//2-B):(mx//2+B)]
    return torch.tensor(f)

# return alpha profile of the SCD matrix
def alphaprofile(s):
    return np.amax(np.absolute(s), axis=1)

def spatialfft(input, mode = None):
    """ Performs Spatial FFT (n-point DFT) on input data (along axis = 1)
        Input: The rows of the input are the n-channels to be processed

        Mode: Default is fft. Setting as 'r' mode selects rfft (for real-valued input)

        Output: Complex-valued data. Output shape is same as the input shape    
    """
    if mode == None:
        spatial_data = np.fft.fft(input, axis = 1)
    elif mode == 'r':
        spatial_data = np.fft.rfft(input, axis = 1)

    return spatial_data

def scf_cube(inputs, Np=512, B=256, L=16, mode=None):

    spatial_data = spatialfft(inputs, mode=mode)
    assert spatial_data.shape[-1] == 2048, "SCF Input size should be 2048"

    outputs = []

    for i in range(spatial_data.shape[0]):
        f = scd_fam(spatial_data[i,:], Np, L, B=B)
        outputs.append(f)
    outputs = torch.stack(outputs, dim=0)

    return outputs

    
def scf_cube_old(inputs, node=None, Np=512, B=256, L=16, mode=None):

    outputs = []

    for i in range(inputs.shape[0]):
        f = scd_fam(inputs[i,:], Np, L, B=B)
        outputs.append(f)
        if node:
            node.scf_display.setText('Loading SCF'+'.'*(i+1))
    outputs = torch.stack(outputs, dim=0)

    return outputs



# compare with precomputed solution
if __name__ == "__main__":
    def audiotest():
        (Np, L, x, y) = np.load('audiosample.npy', allow_pickle=True)
        print("x.shape={}, Np={}, L={}".format(x.shape, Np, L))
        f = np.absolute(scd_fam(x, Np, L))
        err = np.linalg.norm(f - y)
        passfail = 'PASS' if err == 0.0 else 'FAIL'
        print("audiotest: {} (error={})".format(passfail, err))

    def bpsktest():
        # x is a bpsk + noise input
        x = np.load('../gen/noise_bpsk.npy')[0:1024]
        plotscd_fam(x, 256, 1, "BSPK")

    def main():
        import argparse
        parser = argparse.ArgumentParser(description='scf analysis')
        parser.add_argument("-t", action="store_true", help="time execution")
        args = parser.parse_args()
        if args.t:
            import timeit
            runs = 5
            print("Timing excecution over {} runs".format(runs))
            print(timeit.timeit("audiotest()", number=runs,
                  setup="from __main__ import audiotest"))
        else:
            audiotest()
        bpsktest()

    main()
