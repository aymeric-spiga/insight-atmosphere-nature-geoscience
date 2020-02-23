## A. Spiga, modified after code by Evgeniya Predybaylo

def wavelet(data,freq=20.,\
            filename=None,title=None,ymin=None,ymax=None,\
            offset = 0., facperiod = 1., unit = "seconds",\
            contrast=1., cutoff=None, s0fac=15.):

  ################################
  #if title is None:
  #    title = 'Wavelet Power Spectrum'
  ################################
  if filename is None:
      filename = "wavelet"
  ################################
  # freq in Hz
  dt = 1./freq
  ################################

  import numpy as np
  from waveletFunctions import wavelet, wave_signif
  import matplotlib.pylab as plt
  import matplotlib
  from mpl_toolkits.axes_grid1 import make_axes_locatable

  __author__ = 'Evgeniya Predybaylo'


  # WAVETEST Example Python script for WAVELET, using NINO3 SST dataset
  #
  # See "http://paos.colorado.edu/research/wavelets/"
  # The Matlab code written January 1998 by C. Torrence is modified to Python by Evgeniya Predybaylo, December 2014
  #
  # Modified Oct 1999, changed Global Wavelet Spectrum (GWS) to be sideways,
  #   changed all "log" to "log2", changed logarithmic axis on GWS to
  #   a normal axis.
  # ------------------------------------------------------------------------------------------------------------------
  
  # READ THE DATA
  sst = data
  
  #----------C-O-M-P-U-T-A-T-I-O-N------S-T-A-R-T-S------H-E-R-E------------------------------------------------------
  
  # normalize by standard deviation (not necessary, but makes it easier
  # to compare with plot on Interactive Wavelet page, at
  # "http://paos.colorado.edu/research/wavelets/plot/"
  variance = np.std(sst, ddof=1) ** 2
  sst = (sst - np.mean(sst)) / np.std(sst, ddof=1)
  n = len(sst)

  time = np.arange(len(sst)) * dt # construct time array
  #nn = 10000 ; time=np.linspace(0,nn,len(sst))
  #xlim = ([1870, 2000])  # plotting range
  #xlim=  ([0, nn])
  pad = 1  # pad the time series with zeroes (recommended)
  dj = 0.25  # this will do 4 sub-octaves per octave
  s0 = s0fac * dt  # 15 5 2 -- this says start at a scale of 6 months
  j1 = 48 / dj  # 24 ;; 12 / dj this says do 7 powers-of-two with dj sub-octaves each
  lag1 = 0.72  # lag-1 autocorrelation for red noise background
  mother = 'PAUL'
  
  # Wavelet transform:
  wave, period, scale, coi = wavelet(sst, dt, pad, dj, s0, j1, mother)
  power = (np.abs(wave)) ** 2  # compute wavelet power spectrum
  
  ## Significance levels: (variance=1 for the normalized SST)
  #signif = wave_signif(([1.0]), dt=dt, sigtest=0, scale=scale, lag1=lag1, mother=mother)
  #sig95 = signif[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])  # expand signif --> (J+1)x(N) array
  #sig95 = power / sig95  # where ratio > 1, power is significant
  
  ## Global wavelet spectrum & significance levels:
  #global_ws = variance * (np.sum(power, axis=1) / n)  # time-average over all times
  #dof = n - scale  # the -scale corrects for padding at edges
  #global_signif = wave_signif(variance, dt=dt, scale=scale, sigtest=1, lag1=lag1, dof=dof, mother=mother)
  
  ## Scale-average between El Nino periods of 2--8 years
  #avg = np.logical_and(scale >= 2, scale < 8)
  #Cdelta = 0.776  # this is for the MORLET wavelet
  #scale_avg = scale[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])  # expand scale --> (J+1)x(N) array
  #scale_avg = power / scale_avg  # [Eqn(24)]
  #scale_avg = variance * dj * dt / Cdelta * sum(scale_avg[avg, :])  # [Eqn(24)]
  #scaleavg_signif = wave_signif(variance, dt=dt, scale=scale, sigtest=2, lag1=lag1, dof=([2, 7.9]), mother=mother)
  
  
  
  ################################################""
  
  
  
  # cone-of-influence
  for ii in range(power.shape[1]):
     vec = period
     indy = np.where(vec >= coi[ii])
     for jjj in indy:
       power[jjj,ii] = np.nan
       #sig95[jjj,ii] = 0.

  if ymax is None:
      #######################################  
      ## define automatically lower bound
      try:
        limit = power[:,power.shape[1]/2]
        ymax = np.min(period[np.isnan(limit)])
      except:
        ymax = 3700.
  if ymin is None:
      #######################################  
      ## define automatically upper bound
      ymin = s0fac*dt #5.*dt
      #######################################

  #ymax = 20. # see coi
  #ymin = 0.8 # see 4 delta x
  
  
  import ppplot
  #ppplot.quickplot(sst)
  
  #ppplot.changefont(16)
  #fig = ppplot.figuref(x=16,y=6)
  #pl = ppplot.plot1d(fig=fig)
  #pl.f = sst
  #pl.make()


  ppplot.changefont(24)
  fig = ppplot.figuref(x=16,y=6)
  pl = ppplot.plot2d(fig=fig)
  pl.f = power #np.log(power)
  pl.x = offset + (time / facperiod)
  pl.y = period / facperiod
  pl.xlabel = 'time coordinate (%s)' % (unit)
  pl.ylabel = 'period (%s)' % (unit)
  if title is not None:
      pl.title = title
  pl.logy = True
  pl.invert = True
  pl.colorbar = "magma" #"hot"
  pl.showcb = False
  pl.div = 40
  
  import ppcompute
  pl.vmin = ppcompute.min(power)
  pl.vmax = ppcompute.max(power)/contrast
  
  pl.ymin = ymin
  pl.ymax = ymax
  pl.xmin = min(pl.x)
  pl.xmax = max(pl.x)
  #pl.nxticks = pl.xmax - pl.xmin + 1
  pl.make()

  import matplotlib.pyplot as plt
  if cutoff is not None:
      plt.plot([pl.xmin,pl.xmax], [cutoff,cutoff], color='y', linestyle='--') #, linewidth=2)

  plt.grid(which="both",linestyle="--",color="g")
  
  #import matplotlib.pyplot as plt
  #from matplotlib.ticker import FormatStrFormatter
  #
  #ax = plt.gca()
  #ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
  #plt.contour(time, period, sig95, [-99, 1], colors='g')
  


  ppplot.save(mode="png",filename=filename+"_wavelet")
  #ppplot.show()
  


  
  
  
  
  ###############################################
  
  
  
  
  
  
  
#  #------------------------------------------------------ Plotting
#  
#  #--- Plot time series
#  plt.figure(figsize=(18, 9))
#  plt.subplot(221)
#  plt.plot(time, sst)
#  plt.xlim(xlim[:])
#  plt.xlabel('km')
#  plt.ylabel('NINO3 SST (degC)')
#  plt.title('a) NINO3 Sea Surface Temperature (seasonal)')
#  plt.hold(False)
#  
#  # --- Plot 2--8 yr scale-average time series
#  plt.subplot(222)
#  plt.plot(time, scale_avg)
#  plt.xlim(xlim[:])
#  plt.xlabel('km')
#  plt.ylabel('Avg variance (degC^2)')
#  plt.title('d) 2-8 yr Scale-average Time Series')
#  plt.hold(True)
#  plt.plot(xlim, scaleavg_signif + [0, 0], '--')
#  plt.hold(False)
#  
#  #--- Contour plot wavelet power spectrum
#  plt3 = plt.subplot(223)
#  levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
#  CS = plt.contourf(time, period, np.log2(power), len(levels))  #*** or use 'contour'
#  im = plt.contourf(CS, levels=np.log2(levels))
#  plt.xlabel('km')
#  plt.ylabel('horinz wv (km)')
#  plt.title('b) NINO3 SST Wavelet Power Spectrum (in base 2 logarithm)')
#  plt.xlim(xlim[:])
#  # 95# significance contour, levels at -99 (fake) and 1 (95# signif)
#  plt.hold(True)
#  plt.contour(time, period, sig95, [-99, 1], colors='k')
#  # cone-of-influence, anything "below" is dubious
#  plt.plot(time, coi, 'k')
#  plt.hold(False)
#  # format y-scale
#  plt3.set_yscale('log', basey=2, subsy=None)
#  plt.ylim([np.min(period), np.max(period)])
#  ax = plt.gca().yaxis
#  ax.set_major_formatter(matplotlib.ticker.ScalarFormatter())
#  plt3.ticklabel_format(axis='y', style='plain')
#  plt3.invert_yaxis()
#  # set up the size and location of the colorbar
#  divider = make_axes_locatable(plt3)
#  cax = divider.append_axes("bottom", size="5%", pad=0.5)
#  plt.colorbar(im, cax=cax, orientation='horizontal')
#  
#  #--- Plot global wavelet spectrum
#  plt4 = plt.subplot(224)
#  plt.plot(global_ws, period)
#  plt.hold(True)
#  plt.plot(global_signif, period, '--')
#  plt.hold(False)
#  plt.xlabel('Power (degC^2)')
#  plt.ylabel('horinz wv (km)')
#  plt.title('c) Global Wavelet Spectrum')
#  plt.xlim([0, 1.25 * np.max(global_ws)])
#  # format y-scale
#  plt4.set_yscale('log', basey=2, subsy=None)
#  plt.ylim([np.min(period), np.max(period)])
#  ax = plt.gca().yaxis
#  ax.set_major_formatter(matplotlib.ticker.ScalarFormatter())
#  plt4.ticklabel_format(axis='y', style='plain')
#  plt4.invert_yaxis()
#  
#  plt.tight_layout()
#  
#  plt.show()
#  
#  # end of code
  
