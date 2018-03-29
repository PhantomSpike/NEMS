from functools import partial
import logging
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

from nems.plots.assemble import plot_layout
from nems.plots.heatmap import (weight_channels_heatmap, fir_heatmap,
                                strf_heatmap)
from nems.plots.scatter import plot_scatter
from nems.plots.spectrogram import spectrogram_from_epoch
from nems.plots.timeseries import timeseries_from_epoch
from nems.plots.histogram import pred_error_hist
import nems.modelspec as ms

log = logging.getLogger(__name__)


def plot_summary(rec, modelspecs, stimidx=0):
    '''
    Plots a summary of the modelspecs and their performance predicting on rec.
    '''
    if not modelspecs:
        raise ValueError('No modelspecs defined')

    if type(rec) is list:
        rec=rec[0]

    stim = rec['stim']
    resp = rec['respavg'] if 'respavg' in rec.signals else rec['resp']

    # SVD change: assume predictions have already been generated by
    #  nems.analysis.api.generate_prediction()
    pred=[rec['pred']]
    # Make predictions on the data set using the modelspecs
    #pred = [ms.evaluate(rec, m)['pred'] for m in modelspecs]

    sigs = [resp]
    sigs.extend(pred)

    # Example of how to plot a complicated thing:
    extracted = resp.extract_epoch('TRIAL')
    finite_trial = [np.sum(np.isfinite(x)) > 0 for x in extracted]
    occurrences, = np.where(finite_trial)
    if stimidx>len(occurrences)-1:
        stimidx=0
    occurrence = occurrences[stimidx]

    module_names=[m['fn'] for m in modelspecs[0]]


    def my_scatter_raw(idx, ax):
        plot_scatter(pred[idx], resp, ax=ax, title=rec.name)

    def my_scatter(idx, ax):
        plot_scatter(pred[idx], resp, ax=ax,
                     title="{0} r_test={1:.3f}".format(rec.name,modelspecs[0][0]['meta']['r_test']),
                     smoothing_bins=100)

    def my_spectro(ax):
        spectrogram_from_epoch(stim, 'TRIAL', ax=ax, occurrence=occurrence,
                               title=modelspecs[0][0]['meta']['cellid']+" - "+modelspecs[0][0]['meta']['modelname'])

    def my_timeseries(ax):
        timeseries_from_epoch(sigs, 'TRIAL', ax=ax, occurrences=occurrence)

    def my_strf(idx, ax):
        strf_heatmap(modelspecs[idx], ax=ax)

    def my_wc(idx, ax):
        weight_channels_heatmap(modelspecs[idx], ax=ax)

    def my_fir(idx, ax):
        fir_heatmap(modelspecs[idx], ax=ax)

    def my_nl(idx, ax):
        plot_scatter(pred[idx], resp, ax=ax,
                     title="{0} r_test={1:.3f}".format(rec.name,modelspecs[0][0]['meta']['r_test']),
                     smoothing_bins=100)

    def my_state(ax):
        """
        Plot state variables, pred and response across entire experiment

        Downsample by a factor of 5 first.
        """

        plt.sca(ax)

        r1=resp.as_continuous().T
        p1=pred[0].as_continuous().T
        nnidx=np.isfinite(p1)

        r1=scipy.signal.decimate(r1[nnidx],q=5,axis=0)
        p1=scipy.signal.decimate(p1[nnidx],q=5,axis=0)

        plt.plot(r1)
        plt.plot(p1)
        mmax=np.nanmax(p1)
        if 'state' in rec.signals.keys():
            for m in modelspecs[0]:
                if 'state_dc_gain' in m['fn']:
                    g=np.array(m['phi']['g'])
                    d=np.array(m['phi']['d'])
                s=",".join(rec["state"].chans)
                s=s+ " g="+np.array2string(g,precision=3)+" d="+np.array2string(d,precision=3)+""

            for i in range(1,rec['state'].shape[0]):
                d=rec['state'].as_continuous()[[i],:].T
                d=scipy.signal.decimate(d[nnidx],q=5,axis=0)
                d=d/np.nanmax(d)*mmax - mmax*1.1
                plt.plot(d)
            plt.text(0,2,s)

        plt.axis('tight')

    def my_hist(idx, ax):
        pred_error_hist(resp, pred[idx])

    def make_partials(fn, items):
        partials = [partial(fn, i) for i in range(len(items))]
        return partials

    if len(modelspecs) <= 10:

        plot_list=[[my_spectro],
                   [my_timeseries]]
        if any('fir' in n for n in module_names):
            plot_list.append(make_partials(my_strf, modelspecs))
        if any('nonlinearity' in n for n in module_names):
            plot_list.append([my_nl])
        if any('state' in n for n in module_names):
            plot_list.append([my_state])

        plot_list.append(make_partials(my_scatter, pred))
        #plot_list.append(make_partials(my_hist, modelspecs))

    else:
        # Don't plot the scatters/strfs when you have more than 10
        plot_list=[[my_spectro],[my_timeseries]]

    fig = plot_layout(plot_list)
    fig.tight_layout()
    return fig
