#!/usr/env/python

## Import General Tools
from pathlib import Path
from matplotlib import pyplot as plt

import numpy as np
from astropy import units as u
from astropy import stats
from astropy.table import Table, Column
from astropy.modeling import models, fitting
from astropy.visualization import (MinMaxInterval, PercentileInterval,
                                   ImageNormalize)
import ccdproc

from .core import *
from .. import KeckData, KeckDataList


def determine_read_noise(input, master_bias=None, plot=False, gain=None,
                         clippingsigma=5, clippingiters=3, trim=0,
                         plot_range_std=10,
                         ):
    '''
    Determine read noise from either a set of bias frames or from a single bias
    frame and a master bias.
    '''
    log.info(f'Determining read noise')
    if issubclass(type(input), KeckData)\
        and issubclass(type(master_bias), KeckData):
        bias0 = input
    elif type(input) == KeckDataList:
        log.info(f'  Checking that all inputs are BIAS frames')
        biases = KeckDataList([kd for kd in input.frames if kd.type() == 'BIAS'])
        log.info(f'  Found {biases.len} biases')
        bias0 = biases.pop()
        master_bias = make_master_bias(biases,
                                       clippingsigma=clippingsigma,
                                       clippingiters=clippingiters,
                                       trim=trim)
    else:
        raise KeckDataError(f'Input of type {type(input)} not understood')

    inst = (bias0.instrument).replace(' ', '_')
    obstime = bias0.obstime().replace(':', '').replace('/', '')
    readmode = bias0.readout_mode()
    readmode_str = '' if readmode is None else f"_{readmode}"

    diff = bias0.subtract(master_bias)
    npds = len(diff.pixeldata)
    log.info(f'  Determining read noise for each of {npds} extensions')
    read_noise = []
    for i in range(npds):
        ny, nx = diff.pixeldata[i].shape
        mean, median, std = stats.sigma_clipped_stats(
                      diff.pixeldata[i][trim:ny-trim,trim:nx-trim],
                      sigma=clippingsigma,
                      iters=clippingiters) * u.adu
        mode = get_mode(diff.pixeldata[i])
        log.debug(f'  Bias Diff (mean, med, mode, std) = {mean.value:.1f}, '\
                  f'{median.value:.1f}, {mode:d}, {std.value:.2f}')

        RN = std / np.sqrt(1.+1./master_bias.nbiases )
        log.info(f'  Read Noise is {RN:.2f} for extension {i+1}')
        read_noise.append(RN)

        # Generate Bias Plots
        if plot is not False:
            log.info(f'  Generating plot for: {bias0.filename()}, frame {i}')
            data = bias0.pixeldata[i].data[trim:ny-trim,trim:nx-trim]
            std = RN.value #np.std(data)
            if gain is not None:
                RN *= gain
            mode = get_mode(data)
            median = np.median(data)
            binwidth = int(20*std)
            binsize = 1
            med = np.median(data)
            bins = [x+med for x in range(-binwidth,binwidth,binsize)]
            norm = ImageNormalize(data, interval=PercentileInterval(98))

            plt.figure(figsize=(18,18))
            plt.subplot(2,1,1)
            plt.title(f"{inst}: {bias0.filename()}\n"
                      f"Read Noise = {RN:.2f}")
            N, bins, _ = plt.hist(data.ravel(), log=True, bins=bins,
                                  color='g', alpha=0.5, label='data')
            rn_envelope = models.Gaussian1D(mean=median, stddev=std,
                          amplitude=max(N))
            modelx = np.linspace(mode-plot_range_std*std, mode+plot_range_std*std, 100)
            modely = rn_envelope(modelx)
            plt.plot(modelx, modely, 'r-', alpha=0.5,
                     label=f'model (RN={RN:.2f})')
            plt.xlim(mode-plot_range_std*std, mode+plot_range_std*std)
            plt.ylim(1,max(N)*2)
            plt.xlabel('Value (ADU)')
            plt.ylabel('N Pix')
            plt.grid()
            plt.legend(loc='best')

            plt.subplot(2,1,2)
            plt.title(f"{bias0.filename()}, trim={trim:d}\n"
                      f"OBSTIME = {obstime}")
            plt.imshow(data, origin='lower', norm=norm, cmap='gray')

            plot_file = Path(f'read_noise_{inst}{readmode_str}_ext{i}.png')
            log.info(f'  Generating read noise plot: {plot_file}')
            plt.savefig(plot_file, bbox_inches='tight', pad_inches=0.10)

    return u.Quantity(read_noise)


def determine_dark_current(input, master_bias=None, plot=False,
                           nozero=False, usemedian=False,
                           clippingsigma=5, clippingiters=3, trim=0):
    '''
    Determine dark current from a set of dark frames and a master bias.
    '''
    log.info('Determining dark current')
    dark_frames = KeckDataList([kd for kd in input.frames if kd.type() == 'DARK'])
    log.info(f'  Found {dark_frames.len} dark frames')
    npds = len(dark_frames.frames[0].pixeldata)
    log.info(f'  Determining dark current for each of {npds} extensions')
    inst = (dark_frames.frames[0].instrument).replace(' ', '_')
    readmode = dark_frames.frames[0].readout_mode()
    readmode_str = '' if readmode is None else f"_{readmode}"

    # Get image statistics for each dark frame
    exptimes = []
    dark_means = []
    dark_medians = []
    for dark_frame in dark_frames.frames:
        exptimes.append(dark_frame.exptime())
        dark_diff = dark_frame.subtract(master_bias)
        dark_mean = [None] * npds
        dark_median = [None] * npds
        for i in range(npds):
            ny, nx = dark_diff.pixeldata[i].shape
            mean, med, std = stats.sigma_clipped_stats(
                       dark_diff.pixeldata[i][trim:ny-trim,trim:nx-trim],
                       sigma=clippingsigma, iters=clippingiters) * u.adu
            log.debug(f'  Bias Diff (mean, med, std) = {mean.value:.1f}, '\
                      f'{med.value:.1f}, {std.value:.2f}')
            dark_mean[i] = mean.value
            dark_median[i] = med.value
        dark_means.append(dark_mean)
        dark_medians.append(dark_median)

    log.info(f'  Obtained statistics for frames with {len(set(exptimes))} '
             f'different exposure times')
    dark_means = np.array(dark_means)
    dark_medians = np.array(dark_medians)

    # Fit Line to Dark Level to Determine Dark Current
    log.info(f"  Determining dark current from")
    exptime_ints = [int(t) for t in exptimes]
    for t in sorted(set(exptime_ints)):
        log.info(f"    {exptime_ints.count(t)} {t} second darks")
    DC = [None]*npds
    line = models.Linear1D(intercept=0, slope=0)
    if nozero is False:
        line.intercept.fixed = True
    fitter = fitting.LinearLSQFitter()
    for i in range(npds):
        if usemedian is False:
            dc_fit = fitter(line, exptimes, dark_means[:,i])
        else:
            dc_fit = fitter(line, exptimes, dark_medians[:,i])
        DC[i] = dc_fit.slope.value * u.adu/u.second
        log.info(f'  Dark Current is {DC[i]:.3f} for extension {i+1}')

        # Plot Dark Current Fit
        if plot is not False:
            log.info(f'  Generating plot for dark current for frame {i}')
            longest_exptime = max(exptimes)
            plt.figure(figsize=(12,6))
            plt.title('Dark Current')
            ax = plt.gca()
            ax.plot(exptimes, dark_means[:,i], 'ko', alpha=1.0,
                    label='mean count level in ADU')
            ax.plot([0, longest_exptime],
                    [dc_fit(0), dc_fit(longest_exptime)],
                    'k-', alpha=0.3,
                    label=f'dark current = {DC[i].value:.3f} ADU/s')
            plt.xlim(-0.02*longest_exptime, 1.10*longest_exptime)
            min_level = np.floor(min(dark_means[:,i]))
            max_level = np.ceil(max(dark_means[:,i]))
            plt.ylim(min([0,min_level]), 1.05*max_level)
            ax.set_xlabel('Exposure Time (s)')
            ax.set_ylabel('Dark Level (ADU)')
            ax.legend(loc='upper left', fontsize=10)
            ax.grid()

            plot_file = Path(f'dark_current_{inst}{readmode_str}_ext{i}.png')
            log.info(f'  Generating dark current plot: {plot_file}')
            plt.savefig(plot_file, bbox_inches='tight', pad_inches=0.10)

    return u.Quantity(DC)


def determine_gain(input, master_bias=None, read_noise=None, plot=False,
                   clippingsigma=5, clippingiters=3, trim=0,
                   aduthreshold=30000):
    '''
    Determine gain from a series of flats of varying exposure time.  This
    assumes a roughly constant illumination.
    '''
    log.info('Determining gain')
    flat_frames = KeckDataList([kd for kd in input.frames
                  if kd.type() in ['INTFLAT', 'FLAT']])
    npds = len(flat_frames.frames[0].pixeldata)

    # Get image statistics for each flat file
    names = ['file', 'exptime', 'exptime_int']
    dtypes = ['a100', 'f4', 'i4']
    names.extend( [f"mean{i}" for i in range(npds)] )
    dtypes.extend( ['f4']*npds )
    names.extend( [f"median{i}" for i in range(npds)] )
    dtypes.extend( ['f4']*npds )
    names.extend( [f"stddev{i}" for i in range(npds)] )
    dtypes.extend( ['f4']*npds )

    flat_table = Table(names=names, dtype=dtypes)
    log.info(f'Step 1: Get image statistics for {flat_frames.len} flat frames')
    log.info(f'Each frame has {npds} extensions')
    for flat_frame in flat_frames.frames:
        log.debug(f'Analyzing {flat_frame.filename()}')
        row = {'file': flat_frame.filename(),
               'exptime': flat_frame.exptime(),
               'exptime_int': int(flat_frame.exptime()),
               }
        bs_flat = flat_frame.subtract(master_bias)
        for i in range(npds):
            ny, nx = bs_flat.pixeldata[i].data.shape
            mean, med, std = stats.sigma_clipped_stats(
                       bs_flat.pixeldata[i][trim:ny-trim,trim:nx-trim],
                       sigma=clippingsigma, iters=clippingiters) * u.adu
            log.debug(f'  Flat (mean, med, std) = {mean.value:.1f}, '\
                      f'{med.value:.1f}, {std.value:.2f}')
            row[f'mean{i}'] = mean.value
            row[f'median{i}'] = med.value
            row[f'stddev{i}'] = std.value
        flat_table.add_row(row)

    bytime = flat_table.group_by('exptime_int')
    exptimes = sorted(set(flat_table['exptime_int']))
    ntimes = len(exptimes)
    filenames = [kd.filename() for kd in flat_frames.frames]
    npds = len(flat_frames.frames[0].pixeldata)
    signal = []
    variance = []
    signal_times = []
    log.info(f"Step 2: Analyze image differences at each of {ntimes} exposure times")
    for exptime in exptimes:
        exps = bytime.groups[bytime.groups.keys['exptime_int'] == exptime]
        nexps = len(exps)
        log.info(f'  Measuring statistics for {nexps} {exptime:d}s flats')
        for i in np.arange(0,nexps,2):
            if i+1 < nexps:
                flat_fileA = exps['file'][i]
                flat_fileB = exps['file'][i+1]
                log.debug(f"{flat_fileA} / {flat_fileB}")
                A = filenames.index(flat_fileA)
                B = filenames.index(flat_fileB)
                expA = flat_frames.frames[A]
                expB = flat_frames.frames[B]

                frame_signal = []
                frame_variance = []
                for j in range(npds):
                    meanA = exps[i][f'mean{j}']
                    meanB = exps[i+1][f'mean{j}']
                    ratio = meanA/meanB
                    log.debug(f'  Forming difference for extension {j} with scaling ratio {ratio:.4f}')
                    expB_scaled = expB.pixeldata[j].multiply(ratio)
                    diff = expA.pixeldata[j].subtract(expB_scaled)                
                    ny, nx = diff.data.shape
                    mean, med, std = stats.sigma_clipped_stats(
                                        diff.data[trim:ny-trim,trim:nx-trim],
                                        sigma=clippingsigma,
                                        iters=clippingiters) * u.adu
                    log.debug(f'    Signal Level = {meanA:.2f}')
                    log.debug(f'    Variance = {std.to(u.adu).value**2/2.:.2f}')
                    frame_variance.append(std.to(u.adu).value**2/2.)
                    frame_signal.append(meanA)
                signal.append(frame_signal)
                variance.append(frame_variance)
                signal_times.append(exps[i]['exptime'])
    signals = np.array(signal)
    signal_times = np.array(signal_times)
    variances = np.array(variance)

    gains = []
    log.info(f'Step 3: Fit gain and linearity for each extension')
    for j in range(npds):
        log.info(f"Fitting model to determine gain for extension {j}")
        signal = signals[:,j]
        variance = variances[:,j]
        ## Fit model to variance vs. signal
        ## var = RN^2 + 1/g S + k^2 S^2
        mask = np.array(np.array(signal) > aduthreshold)
        poly = models.Polynomial1D(degree=2, c0=read_noise[j].to(u.adu).value)
        poly.c0.fixed = True
        poly.c2.min = 0.0
        fitter = fitting.LevMarLSQFitter()
        y = np.array(variance)[~mask]
        x = np.array(signal)[~mask]
        gainfits = fitter(poly, x, y)
        # perr = np.sqrt(np.diag(fitter.fit_info['param_cov']))
        ksq = gainfits.c2.value
        # ksqerr = perr[1]
        # print('  k^2 = {:.2e} +/- {:.2e} e/ADU'.format(ksq, ksqerr))
        # print('  k^2 = {:.2e} e/ADU'.format(ksq))
        g = gainfits.c1**-1 * u.electron/u.adu
        # gerr = gainfits.c1**-2 * perr[0] * u.electron/u.adu
        log.info(f'  Gain = {g:.3f}')
        gains.append(g)

        ## Fit Linearity
        log.debug('  Fitting linear model to find linearity limit')
        line = models.Linear1D(intercept=0, slope=500)
        line.intercept.fixed = True
        fitter = fitting.LinearLSQFitter()
        x = np.array(signal_times)[~mask]
        y = np.array(signal)[~mask]
        linearity_fit = fitter(line, x, y)

        if plot is True:
            log.info('  Generating figures with flat statistics and gain fits')
            plt.figure(figsize=(12,12))
            for i,plottype in enumerate(['', '_log']):
                plt.subplot(2,1,i+1)
                ax = plt.gca()
                x = np.array(signal)[mask]
                y = np.array(variance)[mask]
                ax.plot(x, y, 'ko', alpha=0.3, markersize=5, markeredgewidth=0)
                x = np.array(signal)[~mask]
                y = np.array(variance)[~mask]
                if plottype == '_log':
                    ax.semilogx(x, y, 'ko', alpha=1.0, markersize=8, markeredgewidth=0)
                    sig_fit = np.linspace(min(signal), max(signal), 50)
                    var_fit = [gainfits(x) for x in sig_fit]
                    ax.semilogx(sig_fit, var_fit, 'k-', alpha=0.7,
                            label=f'Gain={g.value:.2f}e/ADU')
                else:
                    ax.plot(x, y, 'ko', alpha=1.0, markersize=8, markeredgewidth=0)
                    sig_fit = np.linspace(min(signal), max(signal), 50)
                    var_fit = [gainfits(x) for x in sig_fit]
                    ax.plot(sig_fit, var_fit, 'k-', alpha=0.7,
                            label=f'Gain={g.value:.2f}e/ADU')
                ax.set_ylabel('Variance')
                ax.set_xlabel('Mean Level (ADU)')
                ax.grid()
                ax.legend(loc='upper left', fontsize=10)

            obstime = flat_frames.frames[0].obstime().replace(':', '').replace('/', '')
            if obstime is None:
                plot_file = Path(f'gain_ext{i}.png')
            else:
                plot_file = Path(f'gain_ext{i}_{obstime}.png')
            plt.savefig(plot_file, bbox_inches='tight', pad_inches=0.10)

            log.info('  Generating figure with linearity plot')
            plt.figure(figsize=(12,12))
            for i,plottype in enumerate(['', '_log']):
                ax = plt.gca()
                time = np.array(signal_times)
                counts = np.array(signal)
                fit_counts = [linearity_fit(t) for t in time]
                y = (counts-fit_counts)/counts * 100.
                if plottype == '_log':
                    ax.semilogx(counts, y, 'ko', alpha=0.5, markersize=5, markeredgewidth=0)
                    ax.semilogx([min(counts), max(counts)], [0, 0], 'k-')
                else:
                    ax.plot(counts, y, 'ko', alpha=0.5, markersize=5, markeredgewidth=0)
                    ax.plot([min(counts), max(counts)], [0, 0], 'k-')
                ax.set_xlabel('Mean Level (ADU)')
                ax.set_ylabel('Signal Decrement (%) [(counts-fit)/counts]')
            #     plt.ylim(np.floor(min(decrements)), np.ceil(max(decrements)))
                ax.grid()
            if obstime is None:
                plot_file = Path(f'linearity_ext{i}.png')
            else:
                plot_file = Path(f'linearity_ext{i}_{obstime}.png')
            plt.savefig(plot_file, bbox_inches='tight', pad_inches=0.10)
    return u.Quantity(gains)

