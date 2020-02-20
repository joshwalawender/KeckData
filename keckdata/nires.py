from .core import *


##-------------------------------------------------------------------------
## NIRES SPEC
##-------------------------------------------------------------------------
class NIRESData(KeckData):
    """Class to represent NIRES data.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.instrument = 'NIRES SPEC'

    def verify(self):
        """Verifies that the data which was read in matches an expected pattern
        """
        pass

    def type(self):
        obstype = self.get('OBSTYPE')
        return obstype.strip().upper()

    def filename(self):
        KOAID = self.get('KOAID', None)
        if KOAID is None:
            return f"{self.get('DATAFILE')}.fits"
        return KOAID

    def exptime(self):
        """Return the exposure time in seconds.
        """
        exptime = float(self.get('ITIME')) * int(self.get('COADDS'))
        return exptime

    def obstime(self):
        date = self.get('DATE-OBS', mode=str)
        time = self.get('UTC', mode=str)
        return f"{date}T{time}"

    def readout_mode(self):
        nreads = self.get('NUMREADS', mode=int)
        translator = {2: 'CDS', 32: 'MCDS16'}
        try:
            mode = translator[nreads]
        except KeyError:
            mode = f'NREADS = {nreads}'
        return mode

    def minitime(self):
        nreads = self.get('NUMREADS', mode=int)
        readtime = self.get('RDITIME', mode=float)
        minitime = np.ceil(nreads/2 * readtime)
        return minitime
