from datetime import datetime
from .core import *


##-------------------------------------------------------------------------
## VYSOS20
##-------------------------------------------------------------------------
class VYSOS20(KeckData):
    """Class to represent VYSOS20 data.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.instrument = 'V20'

    def verify(self):
        """Verifies that the data which was read in matches an expected pattern
        """
        if len(self.headers) != 1:
            raise IncorrectNumberOfExtensions("header", "1", self)
        if len(self.pixeldata) != 1:
            raise IncorrectNumberOfExtensions("pixel", "1", self)
        if len(self.tabledata) > 0:
            raise IncorrectNumberOfExtensions("table", "0", self)

    def type(self):
        obsmode = self.get('IMAGETYP')
        translator = {'Light Frame': 'OBJECT',
                      'Bias Frame': 'BIAS',
                      'Dark Frame': 'DARK',
                      'FLAT': 'FLAT'
                      }
        return translator.get(obsmode.strip(), None)

    def filter(self):
        """Return the image filter.
        """
        filter = self.get('FILTER').strip()
        return filter

    def filename(self):
        return self.fitsfilename

    def exptime(self):
        """Return the exposure time in seconds.
        """
        return float(self.get('EXPTIME'))

    def obstime(self):
        return datetime.strptime(self.get('DATE-OBS'), '%Y-%m-%dT%H:%M:%S')

    def verify(self):
        pass

    def gain(self):
        headergain = self.get('GAIN', None)
        return float(headergain) if headergain is not None else None


##-------------------------------------------------------------------------
## VYSOS5
##-------------------------------------------------------------------------
class VYSOS5(KeckData):
    """Class to represent VYSOS20 data.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.instrument = 'V5'

    def verify(self):
        """Verifies that the data which was read in matches an expected pattern
        """
        if len(self.headers) != 1:
            raise IncorrectNumberOfExtensions("header", "1", self)
        if len(self.pixeldata) != 1:
            raise IncorrectNumberOfExtensions("pixel", "1", self)
        if len(self.tabledata) > 0:
            raise IncorrectNumberOfExtensions("table", "0", self)

    def type(self):
        obsmode = self.get('IMAGETYP').strip()
        translator = {'Light Frame': 'OBJECT',
                      'LIGHT': 'OBJECT',
                      'Bias Frame': 'BIAS',
                      'BIAS': 'BIAS',
                      'Dark Frame': 'DARK',
                      'Dark': 'DARK',
                      'DARK': 'DARK',
                      'FLAT': 'FLAT'
                      }
        return translator.get(obsmode.strip(), None)

    def filter(self):
        """Return the image filter.
        """
        filter = 'PSi'
        return filter

    def filename(self):
        return self.fitsfilename

    def exptime(self):
        """Return the exposure time in seconds.
        """
        return float(self.get('EXPTIME'))

    def obstime(self):
        return datetime.strptime(self.get('DATE-OBS'), '%Y-%m-%dT%H:%M:%S')

    def verify(self):
        pass

    def gain(self):
        headergain = self.get('GAIN', None)
        return float(headergain) if headergain is not None else None
