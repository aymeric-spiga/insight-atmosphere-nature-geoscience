#### TOOL to analyse InSight data
#### A. Spiga 12/2018-12/2019
import sys
import numpy as np
#import matplotlib.pylab as mpl
#import matplotlib
#matplotlib.use('TkAgg')
#import matplotlib.pyplot as mpl
#import ppplot
import ppcompute
from scipy.ndimage.measurements import minimum_position
import time as timelib
import os.path
import datetime
import urllib2

########################################################################
########################################################################
publicpds = True  ## official PDS
secretlink = ""

###################################
###################################
datafolder = './' # change this by setting apss_lib.datafolder
datafolder = "/home/aspiga/data/InSight/api/mws_data"
# datafolder is not used in http mode
###################################
################################### 
npyfolder = "/home/aspiga/data/InSight/npy/" # change this by setting apss_lib.npyfolder
###################################
###################################
####
dict_name = {
  "PRE":"Pressure",
  "P_20Hz":"Pressure",
  "MAT":"Temperature -Y",
  "PAT":"Temperature +Y",
  "HWS":"Wind speed",
  "MHWS":"Wind speed -Y",
  "PHWS":"Wind speed +Y",
  "WD":"Wind dir",
  "MWD":"Wind dir -Y",
  "PWD":"Wind dir +Y",
  "PE5DHZ":"ESTA pressure",
  "SAC791":"West current",
  "SAC771":"East current"
}
dict_unit = {
  "PRE":"Pa",
  "P_20Hz":"Pa",
  "MAT":"K",
  "PAT":"K",
  "HWS":"m/s",
  "MHWS":"m/s",
  "PHWS":"m/s",
  "WD":"deg, 0N/90E",
  "MWD":"deg, 0N/90E",
  "PWD":"deg, 0N/90E",
  "PE5DHZ":"arbitrary unit",
  "SAC791":"Amp",
  "SAC771":"Amp"
}
###################################
################################### 
# tab with tuples (sol,Ls)
ttab = [\
(8,300),\
(57,330),\
(113,0),\
(174,30),\
(240,60),\
(306,90),\
(371,120),\
(431,150),\
(485,180),\
]
ttab15 = [\
(8,300),\
(32,315),\
(57,330),\
(85,345),\
(113,0),\
(143,15),\
(174,30),\
(207,45),\
(240,60),\
(273,75),\
(306,90),\
(339,105),\
(371,120),\
(402,135),\
(431,150),\
(459,165),\
(485,180),\
]
#soltab = np.arange(0,400)
#lstab = apss_lib.insight_sol2ls(soltab)
#for sol in soltab:
#    print sol,round(lstab[sol]%360)  
###################################
################################### 
verbose = True
def message(char):
  if verbose: 
    print "APSS_LIB: "+char
def exitmessage(char):
  message(char)
  sys.exit()
def datemessage():
  print datetime.datetime.today()
###################################
###################################
datemessage()
###################################
###################################
def insight_sol2ls(soltabin,forcecontinuity=True):
  solzero = 555
  return ppcompute.mars_sol2ls(soltabin+solzero,forcecontinuity=forcecontinuity)
###################################
###################################
def getdata(fifi,reload=False,download=False):
  ### --------------------------------------------------------------------------
  ### APSS_LIB.GETDATA
  ### --------------------------------------------------------------------------
  ### USE
  ### 1) apss_lib.getdata("twins_model_0302_01.csv")
  ### 2) path = "/home/aspiga/data/InSight/api/pds/files/twins_model_0302_01.csv"
  ###    apss_lib.getdata(path)
  ### 3) url = "https://atmos.nmsu.edu/PDS/data/PDS4/InSight/twins_bundle/data_derived/sol_0123_0210/twins_model_0204_01.csv"
  ###    apss_lib.getdata(url)
  ### --------------------------------------------------------------------------
  ### DESCRIPTION
  ### - this reads a file name input and outputs ndarray after reading the file
  ### - also works with a url link towards an online file (e.g. PDS)
  ### - a python binary file is saved as .npy for faster subsequent access
  ### - if .npy file is present, it is read directly
  ### - use reload=True to read the original file and recreate the .npy file
  ### --------------------------------------------------------------------------
  time0 = timelib.time()
  message("---------- GETDATA ----------")
  ### remove absolute path, or url, to create a name for Python binary file
  tmp = fifi.split('/')
  csv = tmp[-1]
  npyfile = npyfolder+'/'+csv[0:-3]+"npy"
  ### test if Python binary file is there
  ### if not, read the CSV file
  test = os.path.isfile(npyfile) 
  if test and not reload:
      message("using existing local Python binary file: "+npyfile)
      data = np.load(npyfile)
  else:
      try:
          if "http" in fifi:
              ### the two methods takes about the same time
              if download:
                  zefile = fifi # download then read
                  message("downloading and reading online file "+zefile)
              else:    
                  message("directly reading online file "+fifi)
                  zefile = urllib2.urlopen(fifi) # read directly online
          else:
              zefile = datafolder+'/'+fifi
              message("reading local file "+zefile)
          ######----------------------------------------------------------------
          ###### to avoid "float NaN to integer" exceptions
          ###### we use genfromtxt function with -9999 as missing value
          ###### then we convert -9999 to NaN for float fields
          mv = -9999 #np.nan
          ### to avoid mismatch in dtype depending on the availability of fields in file
          ### we provide dtype to genfromtxt so that all files are the same and maybe appended
          if "twins" in fifi:
              dtype = [('AOBT', '<f8'), ('SCLK', '<f8'), ('LMST', 'S18'), ('LTST', 'S14'), ('UTC', 'S22'), ('HWS', '<f8'), ('VERTICAL_WIND_SPEED', '?'), ('WD', '<f8'), ('WIND_FREQUENCY', '<f8'), ('WS_OPERATIONAL_FLAGS', '<i8'), ('MAT', '<f8'), ('BMY_AIR_TEMP_FREQUENCY', '<f8'), ('BMY_AIR_TEMP_OPERATIONAL_FLAGS', '<i8'), ('PAT', '<f8'), ('BPY_AIR_TEMP_FREQUENCY', '<f8'), ('BPY_AIR_TEMP_OPERATIONAL_FLAGS', '<i8')]
          else:
              dtype = None
          ###### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MAIN CALL
          data = np.genfromtxt(zefile,dtype=dtype,names=True,delimiter=',',filling_values=(mv))
          ###### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MAIN CALL
      except:
          message("File is not found or not readable")
          data = None
          np.save(npyfile,data)     
          message("saved a VOID Python binary file: " + npyfile)
          return data
      for name in data.dtype.names:
          try:
              w = np.where(data[name] == -9999)
              data[name][w] = np.nan  
          except:
              pass
              #message("missing value is -9999 and not NaN for "+name)
      ######----------------------------------------------------------------
      ######
      if "PRESSURE" in data.dtype.names:
          ### coming from ps_calib files
          ### ----
          ### PRESSURE ---> PRE
          data.dtype.names = ('AOBT', 'SCLK', 'LMST', 'LTST', 'UTC', \
                              'PRE', 'PRESSURE_FREQUENCY', \
                              'PRESSURE_TEMP', 'PRESSURE_TEMP_FREQUENCY')        
      if "HORIZONTAL_WIND_SPEED" in data.dtype.names:
          ### coming from twins_model files
          ### ----
          ### HORIZONTAL_WIND_SPEED --> HWS
          ### WIND_DIRECTION --> WD
          ### BMY_AIR_TEMP --> MAT
          ### BPY_AIR_TEMP --> PAT
          data.dtype.names = ('AOBT', 'SCLK', 'LMST', 'LTST', 'UTC', \
                              'HWS', 'VERTICAL_WIND_SPEED', \
                              'WD', 'WIND_FREQUENCY', 'WS_OPERATIONAL_FLAGS', \
                              'MAT', 'BMY_AIR_TEMP_FREQUENCY', 'BMY_AIR_TEMP_OPERATIONAL_FLAGS', \
                              'PAT', 'BPY_AIR_TEMP_FREQUENCY', 'BPY_AIR_TEMP_OPERATIONAL_FLAGS')
      ### when there is no wind data, HWS is read as a boolean
      if "HWS" in data.dtype.names:
          if data["HWS"].dtype == "bool":
              data["HWS"] = False
      #############""
      codes = avail_field(data)
      ### convert temperature to Kelvin
      #if "MAT" in codes:
      #  for cc in ["PAT","MAT"]:
      #    data[cc] = data[cc] + 273.15
      ## convert wind direction to interval [-180,180]
      if "WD" in codes:
          data["WD"] = shiftangle(data["WD"])
      if "PWD" in codes:
          data["PWD"] = shiftangle(data["PWD"])
      if "MWD" in codes:
          data["MWD"] = shiftangle(data["MWD"])
      ## save a Python binary
      np.save(npyfile,data)     
      message("saved a Python binary file: " + npyfile)
      message("got data in %i seconds" % (timelib.time() - time0))
  ## make LTST and LMST floats is not possible 
  ## because their dtype is already set to strings
  ## >>> use gettime 
  return data
###################################
################################### 
def distantpds(sol,var="PRE",event=False):
    ##----------------------------------
    ## - get link to download data
    ## - secret = False >>> official PDS
    ## - secret = True  >>> internal PDS
    ##----------------------------------
    ## generic PDS link
    if publicpds:
        pds_link = "https://atmos.nmsu.edu/PDS/data/PDS4/InSight/"
        ## get the right PDS release
        if sol <= 122:
            sol_link = "sol_0000_0122/"
        elif sol <= 210:
            sol_link = "sol_0123_0210/"
        else:
            exitmessage("NOT YET in PDS!")
    else:
        pds_link = secretlink
    ## get the right kind of link + file
    suffix = ""
    if event:
        suffix = "event"
    if var == "PRE":
        if publicpds:
            var_link = "ps_bundle/data_calibrated/"+sol_link
        else:
            var_link = ""
        #----
        var_link = var_link + "ps_calib"+suffix+"_%.04d_01.csv" % (sol)
    elif var in ["HWS","WD","MAT","PAT"]:
        if publicpds:
            var_link = "twins_bundle/data_derived/"+sol_link
        else:
            var_link = ""
        #----
        var_link = var_link + "twins_model"+suffix+"_%.04d_01.csv" % (sol)
    ## get the final url
    url = pds_link+var_link
    ## end
    return url
################################### data = apss_lib.getsol(17)
################################### data = apss_lib.getsol(302,var="PRE",distant=True,reload=True)
def getsol(sol,reload=False,var=None,distant=False,event=False):
  ## - if var is None 
  ##      --> MWS-style files with all variables
  ## - if var is not None 
  ##      --> PDS-style files with separate files
  suffix = ""
  if event:
      suffix = "event"
  if distant:
      fifi = distantpds(sol,var=var,event=event)
  else:
      if var is None:
          fifi = "%.5d-%.5d.csv" % (sol,sol)
      else:
          if var == "PRE":
              prefix = "ps_calib"+suffix
          elif var in ["HWS","WD","MAT","PAT"]:
              prefix = "twins_model"+suffix
          fifi = prefix+"_%.4d_01.csv" % (sol)
  return getdata(fifi,reload=reload)
################################### apss_lib.quick(17,"PRE")
def quick(sol,code):
  data = getsol(sol)
  mpl.plot(data['SCLK'],data[code],'b.')
  mpl.show()
################################### apss_lib.directpds(17,var="PRE")
def directpds(sol,var="PRE"):
  data = getsol(sol,var=var,distant=True)
  mpl.plot(data['SCLK'],data[var],'b.')
  mpl.show()
###################################
def get_not_missing(data,code):
  return np.isfinite(data[code])
###################################
def remove_missing(data,code):
  w = get_not_missing(data,code)
  return data[code][w],data["SCLK"][w]
###################################
def fill_missing(data,code,kind='linear',dt=None):
  ## !!! do not use dt > dt_file !!!
  ## if dt=None, possible de remplacer 
  ## data["PRE"],foo = apl.fill_missing(data,"PRE")
  import scipy.interpolate as spint
  ## get reference data
  w = get_not_missing(data,code) #np.isfinite(data[code])
  xref = data["SCLK"][w]
  yref = data[code][w]
  ## create interpolation function
  func = spint.interpolate.interp1d(xref, yref, kind=kind, fill_value="extrapolate")
  ## interpolate to full coordinate
  xnew = data["SCLK"]
  ## [or] to a made-up coordinate with step dt (seconds)
  if dt is not None:
   xmin = np.min(xnew)
   xmax = np.max(xnew)
   xnew = np.arange(xmin,xmax,dt)
  ## perform the interpolation
  interpolated = func(xnew)
  return interpolated, xnew
###################################
def getwhere(time,mint=None,maxt=None):
  ##-----------------
  ## get indices for a time interval
  ##-----------------
  idx = None
  if mint is not None and maxt is not None:
    idx = (time > mint)*(time < maxt)
  elif mint is not None:
    idx = (time > mint)
  elif maxt is not None:
    idx = (time < maxt)
  ##-----------------
  if idx is not None:
    w = np.where(idx)
  else:
    w = np.where(time >= 0) # i.e. all
  ##-----------------
  return w
###################################
def gettime(data,timetype):
  time = data[timetype]
  ## convert to float
  if timetype == "LTST":
    time = ltstfloat(time)
  elif timetype == "LMST":
    time = lmstfloat(time)
  return time
###################################
def reduced_data(data,mint=None,maxt=None,timetype='LTST'):
  ##-----------------
  ## get a data object reduced to the indicated time interval
  ##-----------------
  ## 1. get times (converted correctly)
  time = gettime(data,timetype)
  ## 2. get indices
  w = getwhere(time,mint=mint,maxt=maxt)
  ## 3. build new ndarray
  ref = data["SCLK"][w]
  dataout = np.ndarray(ref.shape, dtype=data.dtype)
  ## 4. fill new ndarray
  codes = data.dtype.names
  for code in codes:
    dataout[code] = data[code][w]
  return dataout
###################################
def avail_field(data):
  return data.dtype.names[4:]
###################################
def fill_missing_all(data,kind='linear'):
  ## data = fill_missing_all(data)
  ## -- or change name
  ## TBD: add LTST in float version (complex to do)
  codes = avail_field(data)
  for cc in codes:
    data[cc],foo = fill_missing(data,cc,kind=kind,dt=None)  
  return data
###################################
def ratiodd(data,mean=13.,std=1.8,code="PRE"):
  from scipy.stats import norm
  ## -----------------------------
  ## calculate a ratio
  ## to normalize vortex counts
  ## for incomplete sols
  ## ratio=1 for complete sol
  ## -----------------------------
  ## calculate integral of PDF
  ## for local times covered
  ## assumed gaussian
  ## -----------------------------
  ## use LMST because it has 
  ## decimals of seconds contrary to LTST
  ## ... but mean and std guessed from LTST
  x = lmstfloat(data['LMST'])
  try:
      ## select time coordinates 
      ## for which data points are not NaN
      prod = data[code]*x
      wherenan = np.isnan(prod)
      isok = (wherenan == False)
      x = x[isok]
  except:
      ## if something goes wrong, that means:
      ## - either PRE is not in file
      ## - or no valid values for PRE are found
      ## in that case return a ratio = 0
      return 0
  f_x = norm.pdf(x, mean, std)
  #w = np.isnan(data["PRE"])
  #f_x[w] = 0. # does not work
  ########################################
  integrand = f_x * np.gradient(x)
  ########################################
  ### <<<<< begin: hack
  ### remove possible peaks in integrand
  ### caused by cuts in temporal coverage
  #fac = 10.   # [trouble when ERPs are included, full sols are not = 1]
  #fac = 50.   # 14 0.792 // 37 0.658 // 22 0.906
  fac = 100.   # 14 0.792 // 37 0.674 // 22 0.906
  #fac = 1000. # 14 0.792 // 37 0.674 // 22 0.937
  w = np.where(integrand < fac*np.mean(integrand))
  integrand = integrand[w] ; x = x[w] ; f_x = f_x[w]
  ### >>>>> end: hack
  ########################################
  ratio = np.round(np.sum(integrand),decimals=3)
  #import matplotlib.pyplot as mpl
  #mpl.plot(x,f_x,'r.') ; mpl.show()
  #mpl.plot(x,integrand,'bo') ; mpl.show()
  ###from scipy.integrate import simps
  ###ratio = simps(f_x,x)
  return ratio
###################################
def ratioddsol(sol,mean=13.,std=1.8,reload=False,var=None,distant=False):
  ## combine GETSOL and RATIODD 
  ## with a possible exception raising if file not here
  try:
     data = getsol(sol,reload=reload,var=var,distant=distant)
     ratio = ratiodd(data,mean=mean,std=std)
  except:
     return 0
  return ratio
###################################
def to_utc(data,lt):
  ltst = data["LTST"]
  cltst = ltstfloat(ltst)
  w = np.where(cltst == lt)
  return data["UTC"][w][0][0:19]
###################################
def winfound(cltstpp,iii,winsearch=50):
  ## define interval in which not to search anymore
  ## -- winsearch in seconds
  whendrop = cltstpp[iii]
  func = np.abs(cltstpp-(whendrop-winsearch/3600.))
  wmin = minimum_position(func)[0]
  func = np.abs(cltstpp-(whendrop+winsearch/3600.))
  wmax = minimum_position(func)[0]
  return wmin,wmax
###################################
def smoothresample(data,code,ikind="linear",freq=20.,window=100,complete=False,reinterpolate=True):
    ########
    ## freq in Hz, window in seconds
    ## detrended, smoothed = smoothresample(data,"PRE")
    ########
    time0 = timelib.time()
    import scipy.interpolate as spint
    ## ----------------
    ## this hack allows to get smooth+detrend versions
    ## - when there is a cut in the middle of the sample
    ## - when there is a high-frequency ERP inserted
    ## - at a constant frequency (provided by freq, usually the highest)
    ## ----------------
    ## 1. get interpolation at hi-frequency 
    hif, hix = fill_missing(data,code,dt=1./freq, kind=ikind)
    ## 2. smooth and detrend this interpolation at high-frequency
    swin = window*freq
    hid, his = detrendsmooth(hif,swin)
    ## 3. re-interpolate results to original time series
    if reinterpolate:
        funcd = spint.interpolate.interp1d(hix, hid, kind=ikind, fill_value="extrapolate")
        funcs = spint.interpolate.interp1d(hix, his, kind=ikind, fill_value="extrapolate")
        if not complete:
            w = get_not_missing(data,code)
            xnew = data["SCLK"][w] 
        else:
            xnew = data["SCLK"]
        hid, his = funcd(xnew), funcs(xnew)
    dt = timelib.time() - time0
    if dt > 5.:
        message("smoothresample done in %i seconds" % (dt))
    if reinterpolate:
        ##########################
        ### remove all points within boundaries +/- window/2
        sec = xnew
        nn = sec[0]+(window/2)
        xx = sec[-1]-(window/2)
        q = getwhere(sec,maxt=nn)
        hid[q] = np.nan
        his[q] = np.nan
        q = getwhere(sec,mint=xx)
        hid[q] = np.nan
        his[q] = np.nan
        ##########################
    return hid, his
###################################
###################################
def remove_points(tab,sta,end,freq):  
  # end should not be 0
  st = +int(sta*freq)
  en = -np.max([int(end*freq),1])
  #print st,en,sta,end
  tab = tab[st:en]
  return tab
###################################
###################################
def treat_missing(rawstr,time): 
  # convert time in float (no gap)
  time = np.array([np.float(tt) for tt in time])
  # convert field in float (gap)
  data = np.zeros(np.shape(rawstr))
  ndata = (rawstr == '')
  ydata = (rawstr != '')
  data[ndata] = np.nan
  data[ydata] = [np.float(rr) for rr in rawstr[ydata]]
  # remove missing values because smooth
  data = data[ydata]
  time = time[ydata]
  return data,time,ydata
###################################
###################################
def detrendsmooth(data,window):
  ##########
  ### WINDOW IS IN NUMBER OF POINTS
  ##########
  ## d,s = detrendsmooth(data,window)
  datasmooth = ppcompute.smooth1d(data,window=int(window))
  nt = datasmooth.size
  ntt = data.size
  #print nt, ntt
  nn = (nt-ntt)/2
  datasmooth = datasmooth[nn:nt-nn-1]
  #print nn, datasmooth.size
  detrend = data - datasmooth
  return detrend, datasmooth
###################################
###################################
def ltstfloat(chartab,clock=None,indices=None):
  ####
  nt = chartab.size
  if nt == 1:
    chartab = np.array([chartab])
  ####
  if indices is None:
      ## there are two types of LTST entries
      ## one with simply the LTST time (MWS)
      ## one with sol number + LTST time (PDS)
      zelen = len(chartab[0])
      if zelen == 8:
          indices = [0,2,3,5,6,8]
      elif zelen == 14: 
          indices = [6,8,9,11,12,14]
      else:
          exitmessage("unknown LTST format!")
  ####
  tt = np.zeros((nt))
  cs = np.zeros((nt))
  n = 0
  for iii in np.arange(nt):
    ltst = chartab[iii]
    hh = float(ltst[indices[0]:indices[1]])
    mm = float(ltst[indices[2]:indices[3]])
    ss = float(ltst[indices[4]:indices[5]])
    tt[iii] = hh + mm/60. + ss/3600.
    ## LTST is only accurate to the second
    ## below the second level, if SCLK is given
    ## we reconstruct the value to add in cs
    ## NB: not perfect, sometimes two points overlap
    ## because not regular LTST <> SCLK
    if clock is not None:
      sps = getres(clock)
      ## we only correct if frequency does not change
      if sps != 0.:
        dt = 1./sps
        if tt[iii] == tt[iii-1]:
           n = n + 1
           cs[iii] = n*dt
        else:
           #print n #pour voir irregularites
           n = 0
  tt += cs/3600.
  return tt
#########
def lmstfloat(chartab):
  lmst = ltstfloat(chartab,indices=[6,8,9,11,12,18])
  try:
    # one point at the beginning might not be exactly midnight
    if lmst[0] > 23.9999: 
        lmst[0] = 0.
    # a couple points are the next day in the end
    # correct this to add 24 to those LMST
    # -- only works for complete sols
    w = np.where(np.diff(lmst) < 0)
    if len(w) > 1:
        ind = int(w[0]+1)
        lmst[ind:-1] = lmst[ind:-1] + 24.
        lmst[-1] = lmst[-1] + 24.
  except:
    pass
  return lmst
#########
def makedate(chartab):
  import datetime
  nt = chartab.size
  tt = []
  for iii in np.arange(nt):
    char = chartab[iii]
    yy = int(char[0:4]) #; print yy
    mm = int(char[5:7]) #; print mm
    dd = int(char[8:10]) #; print dd
    hh = int(char[11:13]) #; print hh
    mi = int(char[14:16]) #; print mi
    ss = int(char[17:19]) #; print ss
    tt.append(datetime.datetime(yy,mm,dd,hh,mi,ss))
  return tt
def namefile(sol,suffix):
  fifi = "%.5d-%.5d.csv" % (sol,sol)
  output = "sol%.5d%s_" % (sol,suffix)
  return fifi, output
def getparam(suffix):
  if suffix == "":
      #ltbounds = [07.,18.] # btw 07 and 08, GW false positives
      ltbounds = [08.,18.]
      #ltbounds = [11.,15.]
      ww = 500 # (in seconds) tradeoff, w 1000s some false positives, 200 good for embedded but not biggest
      ww = 1000 # w 1000s, the value of drop is more faithfully obtained, 
                # w 500s a bit underestimated (because drop starts to be appearing in smoothed series)
                # or even 2000s but then too much false positives...
  elif suffix == "evening":
      ltbounds = [18.,23.]
      ww = 1000 # 500 too small! 
  elif suffix == "night":
      ltbounds = [01.,06.]
      ww = 1000
  else:
      ltbounds = [00.,24.]
      ww = 1000
  return ltbounds, ww

def shiftangle(wd):
    ff = wd[np.isfinite(wd)] 
    w = np.where(ff > 180.)
    ff[w] = ff[w] - 360.
    wd[np.isfinite(wd)] = ff
    return wd

def getres(time):
  dt = np.round(np.diff(time),2)
  res = 1./dt
  #print res
  damin = np.min(res)
  damax = np.max(res)
  sps = damin
  if damin != damax:
     #print "mismatch of frequency in the sample", damin, damax
     #raise SystemExit(0)
     sps = 0.
  return sps

##############################
############################## 
def getpressure(data,suffix,window=None,ltset=None,code="PRE"):
  #### ************************
  #### first get data with data = getsol(sol)
  #### ************************
  #### smoothing window depends on suffix
  #### window = window (in seconds)
  #### -- set by suffix if None
  if window is None:
    ltbounds, window = getparam(suffix)
  else:
    ltbounds, foo = getparam(suffix)
  message("apply smoothing window %s seconds" % (window))
  if ltset is not None:
    ltbounds = ltset
  ####
  codes = avail_field(data)
  if code not in codes:
    message(code+" not in file")
    blk = np.array([])
    return blk, blk, blk, blk, blk, blk, 0.
  else:
    #ratio = ratiodd(data)
    ####
    pp,timepp = remove_missing(data,code)
    w = get_not_missing(data,code)
    ltstpp = data['LTST'][w]
    utcpp = data['UTC'][w]
    ####  
    try:
        ### do not use np.mean which does not know how to handle nan
        freq = ppcompute.max(data["PRESSURE_FREQUENCY"])
    except:
        freq = 2.  #20. to see ERPs #Hz
    #### !!! smoothresample breaks down if freq = nan
    dpp, spp = smoothresample(data,code,window=window,freq=freq)
    ####
    cltstpp = ltstfloat(ltstpp)
    idx = (cltstpp >= ltbounds[0])*(cltstpp <= ltbounds[1])
    w = np.where(idx)
    if len(w) == 0:
        exitmessage("problem with local time bounds")
    #zedate = makedate(utcpp) #???
    return pp[idx],dpp[idx],spp[idx],ltstpp[idx],timepp[idx],utcpp[idx]#,ratio
##############################
##############################
def studypressure(sol,field,suffix,utcpp,ltstpp,ltbounds=None,droplim = -0.3,ltnum=False,isplot=False,window=None):
  fifi, output = namefile(sol,suffix)
  if ltbounds is None:
    ltbounds, ww = getparam(suffix)
  else:
    ltboundsav = ltbounds
    ltbounds, ww = getparam(suffix)
    ltbounds = ltboundsav
  if window is not None:
    ww = window
  ###
  if not ltnum:
      cltstpp = ltstfloat(ltstpp)
  else:
      cltstpp = ltstpp
  ### for incomplete sols (e.g. not any daytime)
  try:
      test = np.max(cltstpp) < 8
  except:
      return None
  if isplot:
      ###
      ppplot.changefont(16)
      fig = ppplot.figuref(x=16,y=6)
      pl = ppplot.plot1d(fig=fig)
      ###
      w = np.isfinite(field)
      pl.f = field[w]
      pl.x = cltstpp[w]
      #pl.xlabel = 'elapsed seconds (first %0i seconds removed)' % (skip_start)
      #pl.xlabel = 'SCLK seconds'
      pl.xlabel = 'Mars LTST (hours)'
      pl.ylabel = "fluctuations from %s-s average" % (ww)
      #pl.legend = '%s (%s)' % (labpp,unipp)
      pl.legend = "Pressure (Pa)"
      if suffix == "":
        pl.ymin = -2.5
        pl.ymax = +1.0
      else:
        pl.ymin = -0.8
        pl.ymax = +0.8
      pl.xmin = ltbounds[0] 
      pl.xmax = ltbounds[1]
      #pl.xmin = np.max([ltbounds[0],cltstpp[0]])
      #pl.xmax = np.min([ltbounds[1],cltstpp[-1]])
      pl.nxticks = pl.xmax-pl.xmin+1
      pl.marker = ''
      pl.linestyle = '-'
      pl.fmt = '%.2f'
      #pl.title = "InSight/APSS, sol %i from %s to %s" % (sol,zedate[0],zedate[-1])
      pl.title = "InSight/APSS on sol %i" % (sol)
      #pl.x = zedate
      #pl.xdate = True
      pl.make()
      #########
  ## p < 0.3 pour les gros (sur le smooth direct) 
  ## gradient pour les petits (sur le detrend ?)
  indices = [] ; ddcount = [] ; drop = []
  search = np.empty_like(field) ; search[:] = field[:]
  fname = output+"DD_%.0f" % (-droplim*10)
  #print fname
  #import os.path
  #test = os.path.isfile(fname+".txt") 
  ######################
  ##### remove NaNs
  if not ltnum:
      w = np.isnan(search)
      search = search[~w]
      utcpp = utcpp[~w]
      cltstpp = cltstpp[~w]
  #######################
  test = False
  if (suffix == "") and (not test):
    countdd = 0 ; droptest = -9999.
    while droptest < droplim: 
      countdd += 1
      iii = minimum_position(search)
      droptest = round(search[iii],2)
      ### if one appears twice in a row, we stop the procedure
      ### this means we have reached limitations
      ### -- or if we simply have reached the limit
      condbreak = False
      if not ltnum:
          if len(indices) >= 2:
              if utcpp[iii[0]] == utcpp[indices[-1]]:
                  condbreak = True
      if droptest > droplim:
          condbreak = True
      if condbreak:
          #indices, ddcount, drop = indices[:-1], ddcount[:-1], drop[:-1]
          break
      else:
          indices.append(iii[0])
          ddcount.append(countdd)
          drop.append(search[iii])
          ## define interval in which not to search anymore
          wmin,wmax = winfound(cltstpp,iii)
          search[wmin:wmax] = 9999.
    ###
    dafile = open("./"+fname+".txt","w")
    for iii in range(len(indices)):
      try:
       ## patch for PDS-derived statistics
       ll = ltstpp[indices[iii]]
       if len(ll) > 8:
          ll = ll[6:]
      except:
       pass
      dafile.write('%2.2d %5.3f %s %s\n' % (ddcount[iii],drop[iii],ll,utcpp[indices[iii]]) )
      ddcount[iii]=utcpp[indices[iii]]
      #print cltstpp[indices[iii]]
    dafile.close()
    #########
    if isplot:
        pl.f = 0.8 - 0.2*search/9999.
        pl.x = cltstpp
        try:
            pl.legend = '%i drops $\delta P < %.1f$ Pa' % (ddcount[-1],droplim) 
        except:
            pass
        pl.make()
  if isplot:
      #fig.suptitle('InSight/APSS', fontsize=22)
      ppplot.save(filename=fname,mode="pdf",folder="./output/pdf_per_sol/")
  return indices, ddcount, drop, search

def dd_strongest(sol,suffix,data,spp,ddcount,indices,drop,ltstpp,timepp):
  fifi, output = namefile(sol,suffix)
  nx=3 #5
  ny=2
  cltstpp = ltstfloat(ltstpp)
  if suffix == "" and ddcount[-1] >= (nx*ny):
    ####
    fig = mpl.figure(figsize=(20,nx*3))
    axarr = fig.subplots(nx, ny)
    ####
    for col in [0,1]:
     for count in np.linspace(1,nx,dtype=int,num=nx):
      if col == 0: dacount = count-1
      elif col == 1: dacount = count+nx-1
      iii = indices[dacount]   
      imin,imax = winfound(cltstpp,iii,winsearch=150)
      axarr[count-1,col].plot(timepp[imin:imax],data[imin:imax])
      axarr[count-1,col].plot(timepp[imin:imax],spp[imin:imax])
      axarr[count-1,col].set_ylabel(r'$P$ (Pa)')
      axarr[count-1,col].set_xlabel(r'seconds')
      ## patch for PDS-derived statistics
      ll = ltstpp[iii]
      if len(ll) > 8:
          ll = ll[6:]
      titi = '%.4dDD%2.2d // %5.2f Pa // %s LTST' % (sol,ddcount[dacount],drop[dacount],ll)#,utcpp[iii])
      axarr[count-1,col].set_title(titi)   
    fig.tight_layout()
    ppplot.save(filename=output+"DDstrongest",folder='./output/pdf_per_sol_strongest/',mode="pdf")

#####################
#####################
def analyze_pressure(lastsol=400,soltab=None,sfxtab=None,recalculate=False,window=None,datatype="mws"):
    dafile = open("./sol.txt","a")
    print(soltab)
    tabddcount=[]
    tabdrop=[]
    listall = False
    if soltab is None:
      soltab = range(14,lastsol+1)
      #dafile = open("./sol.txt","a")
      listall = True
    if sfxtab is None:
      sfxtab = [""] 
      #sfxtab = ["","evening","night"]
    ##########
    for suffix in sfxtab:
        for sol in soltab: 

            print sol

            fifi,output = namefile(sol,suffix)
            fifi = datafolder+"/"+fifi
            fname = './output/txt_per_sol/'+output+"DD_3.txt"
            
            import os.path
            #test1 = os.path.isfile(fifi)
            if recalculate:
              test2 = False
              #message("recalculate pressure drops")
            else:
              test2 = os.path.isfile(fname)

            #### and what about night?
            #if not test1:
            #    #print "no file",fifi
            #    pass
            #elif test2:
            if test2:
                print "already done",fname
                pass
            else:
              try:
                if datatype == "mws":
                    message("FOUND FILE "+fifi+" TO PROCESS")
                    data = getsol(sol)
                elif datatype == "pds":
                    data = getsol(sol,var="PRE",distant=True)

                if suffix == "":
                  pp,dpp,spp,ltstpp,timepp,utcpp = getpressure(data,suffix,window=window)
                  
                  if len(pp) > 0:
                    message("calculating pressure drops for sol %i" % (sol))
                    indices, ddcount, drop, search = studypressure(sol,dpp,suffix,utcpp,ltstpp,window=window)
                    tabddcount = np.append(tabddcount,ddcount)
                    tabdrop = np.append(tabdrop,drop)
                    #dd_strongest(sol,suffix,pp,spp,ddcount,indices,drop,ltstpp,timepp)
                    ##### DONE AFTERWARDS NOW
                    #if listall and suffix == "":
                    #    #ratio = ratiodd(getsol(sol))
                    #    #print sol, ratio
                    #    #cltstp = ltstfloat(ltstpp)
                    #    #dhour = cltstp[-1]-cltstp[0]
                    #    ##ratio = round(ddcount[-1]/dhour,2)
                    #    ##dafile.write('%3.3d %2.2d %5.3f %5.3f %5.3f \n' % (sol, ddcount[-1], np.min(drop), round(dhour,2), ratio) )
                    #    dafile.write('%3.3d %5.3f \n' % (sol, ratio) )
                  else:
                    ### create a dummy file
                    #zefile = open(fname,"w")
                    #zefile.close()
                    pass
              except:
                pass
    if listall:
        dafile.close()
    return tabddcount,tabdrop

##########################
##########################
def plotvar(data,\
code=["HWS","WD","PRE","MAT"],\
name="plotraw",\
mint=None,maxt=None,\
timetype="LTST",\
utctitle=False,\
discrete=None,\
title=None,\
lim=None,\
freq=20.,\
window=None,\
nxticks=None,\
addsmooth=False,\
filename=None,\
perturbamp=None,\
fmt=None,\
isplot=True,\
mode="pdf",\
color=None,\
ymin=None,ymax=None,\
eventbounds=None):

  #####################
  ### ndarray data must have been loaded through e.g. data = getsol(sol)
  #####################
  ### -- utctitle=True to add UTC time as title
  ### -- discrete=True to show measurements as points
  ### -- window=NNN to detrend signal with NNN-s window
  ### -- addsmooth=True to show, instead of detrend, smoothed signal with NNN-s window
  ##------------------------------------------------
  ## 1. get times (converted correctly)
  time = gettime(data,timetype)
  ##------------------------------------------------
  ## 2. get indices
  w = getwhere(time,mint=mint,maxt=maxt)
  ##------------------------------------------------
  ## 3. reduce time
  time = time[w]
  ##------------------------------------------------
  ## 4. min/max
  xmin,xmax = np.min(time),np.max(time)
  xmin,xmax = np.max([mint,xmin]),np.min([maxt,xmax])
  ##------------------------------------------------
  ## 5. loop on variables
  for var in code:

    # select time interval
    field = data[var][w]

    ## if window is provided, smooth and detrend (on whole dataset)
    if window is not None:
        dpp, spp = smoothresample(data,var,freq=freq,window=window,complete=True)
        sfield = spp[w]
        if not addsmooth:
          field = dpp[w]
          if title is not None:
              title = title + " - detrended window %is" % (window)
        ###########################
        #### remove all points within boundaries +/- window/2
        #sec = data["SCLK"][w]
        #nn = sec[0]+(window/2)
        #xx = sec[-1]-(window/2)
        #q = getwhere(sec,mint=nn,maxt=xx)
        #time = time[q]
        #field = field[q]
        #sfield = sfield[q]
        ###########################

    ## only finite values
    ww = np.isfinite(field)

    ### remove for superpose
    pl = ppplot.plot1d()
    if isplot:
        ppplot.changefont(22)
        fig = ppplot.figuref(x=16,y=4)
        pl.fig = fig
    ###
    pl.f = field[ww]
    pl.x = time[ww]
    if timetype == "LMST":
        pl.xlabel = "Mars Local Mean Solar Time (hours)"
    elif timetype == "LTST":
        pl.xlabel = "Mars Local True Solar Time (hours)"
    else:
        pl.xlabel = "Mars "+timetype
    try:
      pl.ylabel = '%s (%s)' % (dict_name[var],dict_unit[var])
    except:
      pass
    if discrete is not None:
      pl.marker = discrete
      pl.linestyle = ''
    else:
      pl.marker = ''
      pl.linestyle = '-'
    if color is not None:
        pl.color = color
    pl.xmin = xmin
    pl.xmax = xmax
    pl.fmt = '%.1f'
    pl.nyticks = 8
    if perturbamp is not None:
        pl.ymin = -perturbamp
        pl.ymax = +perturbamp
        pl.nyticks = 8
        pl.fmt = '%.2f'
    if mint is None and maxt is None and timetype == "LMST":
        pl.xmin, pl.xmax, pl.nxticks = 0, 24, 24

    if ymin is not None:
        pl.ymin = ymin
    if ymax is not None:
        pl.ymax = ymax

    if nxticks is not None:
        pl.nxticks = nxticks

    if "WD" in var:
        pl.fmt = '%.0f'
    #elif "WS" in var:
    #    pl.fmt = '%.2f'
    if window is None:
        #if ("PRE" in var) or ("AT" in var) or ("DHZ" in var):
        if ("AT" in var) or ("DHZ" in var):
            pl.fmt = '%.0f'

    if fmt is not None:
        pl.fmt = fmt

    if title is None:
      if utctitle:
        pl.title = "%s to %s" % (to_utc(data,xmin),to_utc(data,xmax))
    else:
      pl.title = title

    if isplot:
        pl.make() 

    if eventbounds is not None:
      zey = [min(pl.f),max(pl.f)]
      pl.f = zey 
      pl.x = [eventbounds[0],eventbounds[0]]
      pl.color = "m"
      pl.linestyle = "--"
      pl.marker = ""
      pl.make()
      pl.f = zey
      pl.x = [eventbounds[1],eventbounds[1]]
      pl.color = "m"
      pl.linestyle = "--"
      pl.marker = ""
      if isplot:
          pl.make()

    if lim is not None:
      pl.f = lim + field[ww]*0.
      pl.color = "r"
      pl.linestyle = "-"
      pl.marker = ""
      if isplot:
          pl.make()    

    zevar = var
    if window is not None: 
     if addsmooth:
      zevar = "s"+zevar
      pl.f = sfield[ww]
      pl.color = "r"
      pl.linestyle = "--"
      pl.marker = ""
      pl.legend = "smoothed with window %i s" % (window)
      if isplot:
          pl.make()
     else:
      zevar = "d"+zevar 

    if isplot:
        if filename is not None:
            ppplot.save(filename=filename+"_"+zevar,mode=mode)

  fff = pl.f #field[ww]
  ttt = pl.x #time[ww]
  return fff,ttt

##########################
##########################
def do_wavelet(data,code="PRE",freq=1,window=1000,mint=18,maxt=23,timetype='LTST',addsmooth=False,filename=None,perturbamp=None,title=None,tmin=None,tmax=None):

    import wavelet

    ## get data over reduced interval
    datalim = reduced_data(data,mint=mint,maxt=maxt,timetype=timetype)

    ## plot smooth and detrend
    ## -- time axis from data is used
    plotvar(datalim,code=[code],mint=mint,maxt=maxt,timetype=timetype,window=window,addsmooth=addsmooth,filename=filename,perturbamp=perturbamp,title=filename)

    ## resample and smooth
    ## -- time axis from data not used
    ## do not reinterpolate back 
    ## to avoid frequency problems (e.g. ERPs) 
    ## ... because wavelet expect evenly spaced sample
    dpp,spp = smoothresample(datalim,code,ikind="linear",freq=freq,window=window,reinterpolate=False)
    
    ## remove boundaries corresponding to window size
    lim = int(window*freq)/2
    dpp = dpp[lim:-lim]
    
    ## perform wavelet analysis
    if title is None:
        title = "Wavelet Power Spectrum "+code
    if filename is None:
        filename = code
    wavelet.wavelet(dpp,freq=freq,filename=filename+"_wvl",title=title,ymin=tmin,ymax=tmax)

##########################
##########################
def spectra(data,freq,kind="power"):
    ps = np.abs(np.fft.fft(data))**2    # Pa^2/Hz # atmospheric analysis
    asp = np.abs(np.fft.fft(data))      # Pa/Hz**-1/2 # instrument sensitivity
    time_step = 1. / float(freq)
    norm = 2.*time_step/data.size
    ps = ps*norm
    asp = asp*np.sqrt(norm)
    freqs = np.fft.fftfreq(data.size, time_step)
    idx = np.argsort(freqs)
    if kind == "power":
        return ps[idx],freqs[idx]
    elif kind == "amplitude":
        return asp[idx],freqs[idx]

##########################
##########################
def showfreq(data,timetype="LTST",filename=None):
    import ppplot
    ppplot.changefont(18)
    fig = ppplot.figuref(x=20,y=10)
    pl = ppplot.plot1d(fig=fig)
    pl.linestyle, pl.marker, pl.fmt = "", ".", "%.1f"
    pl.xlabel, pl.ylabel = timetype, "frequency (Hz)"
    pl.xmin, pl.xmax, pl.nxticks = 0, 24, 24
    pl.logy = True
    pl.ymin, pl.ymax = 0.01,100   
    for code in ["PRE","MHWS"]:
      try:
        w = np.isfinite(data[code])
        ff = 1./np.gradient(data["SCLK"][w])
        tt = gettime(data,timetype)[w]
        pl.f, pl.x = ff, tt
        pl.legend = code
        pl.make()
      except:
        pass
    if filename is not None:
        ppplot.save(filename=filename,mode="png")

##########################
##########################
def getspectra(data,mint=None,maxt=None,timetype='LTST',code='PRE',swin=500,freq=20,smth=None):
    interm = reduced_data(data,mint=mint,maxt=maxt,timetype='LTST')
    interm,foo = remove_missing(interm,code)
    interm,foo = detrendsmooth(interm,swin*freq) 
    sp,fq = spectra(interm,freq)
    if smth is not None:
        foo,sp = detrendsmooth(sp,smth)
        foo,fq = detrendsmooth(fq,smth)
    return sp,fq

##########################
########################## TBF: problems with outputs
def multisol(solini=66,solsol=20,\
             code="HWS",\
             timetype="LMST",\
             freq=0.1,win=None,detrend=False,\
             ymin=None,ymax=None,\
             compute=None,\
             ttinter=[[0,24]],\
             tcinter=None,\
             redpoint=None,\
             title=None,\
             reload=False,\
             filename=None,\
             nxticks=None,\
             mode="pdf",\
             pds=False,\
             addls=False,\
             marker=None,\
             fmt=None):
    outy = np.array([])
    outx = np.array([])
    import ppplot
    ppplot.changefont(20)
    ppplot.changemarkersize(7)
    fig = ppplot.figuref(x=16,y=4)
    pl = ppplot.plot1d(fig=fig)
    pl.ymin = ymin ; pl.ymax = ymax ; pl.nyticks = 8
    pl.xmin = solini ; pl.xmax = solini+solsol-1 
    if nxticks is None:
        pl.nxticks = np.min([solsol,25])
    else:
        pl.nxticks = nxticks
    pl.color = "b"
    pl.linestyle = "" 
    pl.xlabel = "InSight sol (with fraction = "+timetype+"/24)"
    #pl.xlabel = "InSight sol"
    pl.ylabel = '%s (%s)' % (dict_name[code],dict_unit[code])
    if compute is not None:
        if compute == "max":
            pl.marker = "v" ; pl.fmt = "%.1f"
            pl.ylabel = "MAX "+pl.ylabel
        elif compute == "min":
            pl.marker = "^" ; pl.fmt = "%.1f"
            pl.ylabel = "MIN "+pl.ylabel
        elif compute == "mean":
            pl.marker = "o" ; pl.fmt = "%.1f"
            pl.ylabel = "MEAN "+pl.ylabel
        elif compute == "std":
            pl.marker = "s" ; pl.fmt = "%.2f"  
            pl.ylabel = "STD "+pl.ylabel
    elif win is not None:
        if detrend:
            pl.marker = "," ; pl.fmt = "%.1f"
        else:
            pl.marker = "" ; pl.fmt = "%.1f"
            pl.linestyle = "-"     
    else:
        pl.marker = "," ; pl.fmt = "%.1f"

    if marker is not None:
        pl.marker = marker

    if fmt is not None:
        pl.fmt = fmt
    if title is not None:
        pl.title = title
    for sol in range(solini,solini+solsol):
        try:
            ## get data
            if pds:
                data = getsol(sol,var=code,distant=True,reload=reload)
                if code == "PRE":
                    ratio = ratiodd(data,mean=13.,std=1.8)
                else:
                    ratio = 1. #ratiodd(data,mean=13.,std=1.8,code="HWS") 
            else:
                data = getsol(sol,reload=reload)       
                ratio = ratiodd(data,mean=13.,std=1.8)
            #try:
            #    ratio = ratiodd(data,mean=13.,std=1.8)
            #except:
            #    ratio = -1.0
            #ratio = ratiodd(data,mean=13.,std=1.8)

            if ratio > 0.7:

              ## get time
              time = gettime(data,timetype) 

              ## reduce time because we use complete=False to avoid missing data
              nm = get_not_missing(data,code)
              time = time[nm]

              dtime = np.max(time)-np.min(time)
              if dtime > 20.:
            
                ## smoothresample (all day to minimize the adverse impact of cuts)
                if win is not None:
                  dpp,spp = smoothresample(data,code,freq=freq,window=win,complete=False)
                else:
                  dpp,spp = data[code],data[code]
                  dpp = dpp[nm]
                  spp = spp[nm]
       
                ## loop to plot night.....evening sequence for each sol
                count = 0
                for tt in ttinter:            
                               
                  w = getwhere(time,mint=tt[0],maxt=tt[1])

                  pl.f = spp[w]
                  if detrend:
                      pl.f = dpp[w]
                  pl.x = sol + time[w]/24.

                  if compute is not None:
                      if compute == "std":
                          pl.f = np.std(pl.f)
                      elif compute == "mean":
                          pl.f = np.mean(pl.f)
                      elif compute == "max":
                          pl.f = np.max(pl.f)
                      elif compute == "min":
                          pl.f = np.max(pl.f)
                      pl.x = np.mean(pl.x)                     

                  if compute is not None and redpoint is not None:
                      #dasol = np.round(pl.x,decimals=1)
                      #yaa = np.round(sol+(redpoint/24.),1)
                      #if dasol == yaa  or dasol == yaa+1:
                      if tt[0] == redpoint:
                          pl.color = "r"
                      else:
                          pl.color = "b"

                  if compute is not None and tcinter is not None:
                      pl.color = tcinter[count]

                  outy = np.append(outy,pl.f)        
                  outx = np.append(outx,pl.x) 
                  pl.make()      
                  count = count+1

        except:
            pass


    if addls:
      for tt in ttab:
        sol,ls = tt
        if pl.xmin < sol < pl.xmax:
            mm = np.max(pl.f) ; mn = np.min(pl.f)
            pl.x, pl.f = [sol,sol], [pl.ymin,pl.ymax]
            pl.marker, pl.linestyle = "", "--"
            pl.color = 'm'    
            pl.ax.text(sol,pl.ymax*1.05,r'$L_s=%i^{\circ}$'%(ls),\
                    color = pl.color,horizontalalignment='center',verticalalignment='center',\
                    fontsize=18)
            #pl.ax = add_stripes(pl.ax)
            pl.make()


    if filename is not None:
        ppplot.save(filename=filename+"_"+code,mode=mode)

    return outy,outx
###################################
################################### to add gaps
def add_stripes(axes, alph = 0.05, col = "grey"):
        axes.axvspan(0, 15, alpha=alph, color=col)
        axes.axvspan(30, 33, alpha=alph, color=col)
        axes.axvspan(92, 95, alpha=alph, color=col)
        axes.axvspan(107, 113, alpha=alph, color=col)
        axes.axvspan(120, 123, alpha=alph, color=col)
        axes.axvspan(171, 173, alpha=alph, color=col)
        axes.axvspan(189, 190, alpha=alph, color=col)
        axes.axvspan(216, 218, alpha=alph, color=col)
        axes.axvspan(231, 233, alpha=alph, color=col)
        axes.axvspan(246, 247, alpha=alph, color=col)
        axes.axvspan(266, 267, alpha=alph, color=col)
        axes.axvspan(269, 285, alpha=alph, color=col)
        axes.axvspan(345, 346, alpha=alph, color=col)
        return axes


