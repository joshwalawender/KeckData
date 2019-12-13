from . import *


##-------------------------------------------------------------------------
## HIRES
##-------------------------------------------------------------------------
class HIRESData(KeckData):
    """Class to represent HIRES data.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.instrument = 'HIRES'
        self.xgap = 50
        self.ygap = 50

    def verify(self):
        """Verifies that the data which was read in matches an expected pattern
        """
        if len(self.headers) != 4:
            raise IncorrectNumberOfExtensions("header", "4", self)
        if len(self.pixeldata) not in [1, 2, 3]:
            raise IncorrectNumberOfExtensions("pixel", "1, 2, or 3", self)
        if len(self.tabledata) != 0:
            raise IncorrectNumberOfExtensions("table", "0", self)

    def type(self):
        if self.get('OBSTYPE').upper() == 'BIAS' and float(self.get('DARKTIME')) < 2:
            return 'BIAS'
        elif self.get('OBSTYPE').upper() == 'DARK':
            return 'DARK'
        elif self.get('OBSTYPE').upper() == 'INTFLAT':
            return 'INTFLAT'
        else:
            return None

    def filename(self):
        return f"{self.get('OUTFILE')}{int(self.get('FRAMENO')):04d}.fits"

    def exptime(self):
        """Return the exposure time in seconds.
        """
        if self.type() in ['BIAS', 'DARK']:
            return float(self.get('DARKTIME'))
        else:
            return float(self.get('EXPTIME'))

    def obstime(self):
        return self.get('DATE', None)



##-------------------------------------------------------------------------
## LRIS Blue
##-------------------------------------------------------------------------
class LRISBlueData(KeckData):
    """Class to represent LRIS Blue data.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.instrument = 'LRISb'
        self.xgap = 50
        self.ygap = 50
        self.fixed = False

    def fixme(self):
        """LRIS Blue does not scale DETSEC in the header by binning like other
        instruments do, so we need to fix that here.
        """
        if self.fixed is False:
            binx, biny = self.binning()
            if binx > 1 or biny > 1:
                for i,pd in enumerate(self.pixeldata):
                    badDETSEC = split_fits_section( self.pixeldata[i].header.get('DETSEC') )
                    x1 = np.ceil(badDETSEC['x1']/binx)
                    x2 = x1 + np.floor((badDETSEC['x2']-badDETSEC['x1'])/binx)
                    y1 = np.ceil(badDETSEC['y1']/biny)
                    y2 = y1 + np.floor((badDETSEC['y2']-badDETSEC['y1'])/binx)
                    newDETSEC = f"[{x1:.0f}:{x2:.0f},{y1:.0f}:{y2:.0f}]"
                    print(pd.data.shape)
                    print(self.pixeldata[i].header.get('DETSEC'))
                    print(newDETSEC)
                    print()
                    self.pixeldata[i].header.set('DETSEC', newDETSEC)
        self.fixed = True

    def verify(self):
        """Verifies that the data which was read in matches an expected pattern
        """
        pass

    def type(self):
        if self.get('OBSTYPE').upper() == 'BIAS':
            return 'BIAS'
        elif self.get('OBSTYPE').upper() == 'DARK':
            return 'DARK'
        elif self.get('OBSTYPE').upper() == 'INTFLAT':
            return 'INTFLAT'
        else:
            return None

    def filename(self):
        return f"{self.get('OUTFILE')}{int(self.get('FRAMENO')):04d}.fits"

    def exptime(self):
        """Return the exposure time in seconds.
        """
        return float(self.get('TTIME'))

    def obstime(self):
        return self.get('DATE', None)


##-------------------------------------------------------------------------
## LRIS Red
##-------------------------------------------------------------------------
class LRISRedData(KeckData):
    """Class to represent LRIS Red data.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.instrument = 'LRISr'
        self.xgap = 50
        self.ygap = 50

    def type(self):
        if self.get('OBSTYPE').upper() == 'BIAS':
            return 'BIAS'
        elif self.get('OBSTYPE').upper() == 'DARK':
            return 'DARK'
        elif self.get('OBSTYPE').upper() == 'INTFLAT':
            return 'INTFLAT'
        else:
            return None

    def filename(self):
        return f"{self.get('OUTFILE')}{int(self.get('FRAMENO')):04d}.fits"

    def exptime(self):
        """Return the exposure time in seconds.
        """
        return float(self.get('TTIME'))

    def obstime(self):
        return self.get('DATE', None)


##-------------------------------------------------------------------------
## DEIMOS
##-------------------------------------------------------------------------
class DEIMOSData(KeckData):
    """Class to represent DEIMOS data.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.instrument = 'DEIMOS'
        self.xgap = 50
        self.ygap = 50
