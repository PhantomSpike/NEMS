import scipy.signal as sps
import scipy as scp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_spectrogram(array, fs=None, ax=None, title=None, time_offset=0,
                     cmap=None):

    if not ax:
        ax = plt.gca()

    if fs is None:
        times = np.arange(0, array.shape[1])
    else:
        times = np.arange(0, array.shape[1])/fs-time_offset

    extent = [times[0], times[-1], 1, array.shape[0]]
    ax.imshow(array, origin='lower', interpolation='none',
              aspect='auto', extent=extent, cmap=cmap)
    ax.margins(x=0)

    # Override x-tic labels to display as real time
    # instead of time bin indices.
    #if fs is not None:
    #    locs = ax.get_xticks()
    #    newlabels = ["{:0.3f}".format(l/fs-time_offset) for l in locs]
    #    ax.set_xticklabels(newlabels)

    # TODO: Is there a way the colorbar can overlap part of the image
    # rather than shift it over?
    # cbar = plt.colorbar(fraction=0.05)
    # cbar.set_label('Amplitude')
    ax.set_xlabel('Time')
    ax.set_ylabel('Channel')
    if title:
        ax.set_title(title)


def spectrogram_from_signal(signal, title=None, ax=None):
    # TODO: How can the colorbar be scaled to match other signals?
    array = signal.as_continuous()
    plot_spectrogram(array, fs=signal.fs, title=title, ax=None)


def spectrogram_from_epoch(signal, epoch, occurrence=0, ax=None, title=None):
    if occurrence is None:
        return
    extracted = signal.extract_epoch(epoch)
    array = extracted[occurrence]
    plot_spectrogram(array, fs=signal.fs, ax=ax, title=title)
