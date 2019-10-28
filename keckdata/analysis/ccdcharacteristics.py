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


def determine_read_noise(input, master_bias=None, plot=False,
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
            std = np.std(data)
            mode = get_mode(data)
            binwidth = int(20*std)
            binsize = 1
            med = np.median(data)
            bins = [x+med for x in range(-binwidth,binwidth,binsize)]
            norm = ImageNormalize(data, interval=PercentileInterval(98))

            plt.figure(figsize=(18,18))
            plt.subplot(2,1,1)
            plt.title(bias0.filename())
            plt.imshow(data, origin='lower', norm=norm, cmap='gray')
            plt.subplot(2,1,2)
            plt.hist(data.ravel(), log=True, bins=bins, color='g', alpha=0.5)
            plt.xlim(mode-plot_range_std*std, mode+plot_range_std*std)
            plt.xlabel('Value (ADU)')
            plt.ylabel('N Pix')
            plt.grid()

            if plot is True:
                obstime = bias0.obstime().replace(':', '').replace('/', '')
                if obstime is None:
                    plot_file = Path(f'read_noise_ext{i}.png')
                else:
                    plot_file = Path(f'read_noise_ext{i}_{obstime}.png')
                log.info(f'  Generating read noise plot: {plot_file}')
                plt.savefig(plot_file, bbox_inches='tight', pad_inches=0.10)
            else:
                plt.show()

    return u.Quantity(read_noise)


def determine_dark_current(input, master_bias=None, plot=False,
                           clippingsigma=5, clippingiters=3, trim=0):
    '''
    Determine dark current from a set of dark frames and a master bias.
    '''
    log.info('Determining dark current')
    dark_frames = KeckDataList([kd for kd in input.frames if kd.type() == 'DARK'])
    log.info(f'  Found {dark_frames.len} dark frames')
    npds = len(dark_frames.frames[0].pixeldata)
    log.info(f'  Determining dark current for each of {npds} extensions')

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
#             dark_median[i] = med.value
        dark_means.append(dark_mean)
#         dark_medians.append(dark_median)

    log.info(f'  Obtained statistics for frames with {len(set(exptimes))} '
             f'different exposure times')
    dark_means = np.array(dark_means)
#     dark_medians = np.array(dark_medians)

    # Fit Line to Dark Level to Determine Dark Current
    log.info(f"  Determining dark current from")
    exptime_ints = [int(t) for t in exptimes]
    for t in set(exptime_ints):
        log.info(f"    {exptime_ints.count(t)} {t} second darks")
    DC = [None]*npds
    line = models.Linear1D(intercept=0, slope=0)
    line.intercept.fixed = True
    fitter = fitting.LinearLSQFitter()
    for i in range(npds):
        dc_fit_mean = fitter(line, exptimes, dark_means[:,i])
#         dc_fit_median = fitter(line, exptimes, dark_medians[:,i])
        DC[i] = dc_fit_mean.slope.value * u.adu/u.second
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
                    [dc_fit_mean(0), dc_fit_mean(longest_exptime)],
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

            if plot is True:
                obstime = dark_frames.frames[0].obstime().replace(':', '').replace('/', '')
                if obstime is None:
                    plot_file = Path(f'dark_current_ext{i}.png')
                else:
                    plot_file = Path(f'dark_current_ext{i}_{obstime}.png')
                log.info(f'  Generating dark current plot: {plot_file}')
                plt.savefig(plot_file, bbox_inches='tight', pad_inches=0.10)
            else:
                plt.show()

    return u.Quantity(DC)
