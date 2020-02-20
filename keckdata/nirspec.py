from .core import *


##-------------------------------------------------------------------------
## NIRSPEC SPEC
##-------------------------------------------------------------------------
class NIRSPECData(KeckData):
    """Class to represent NIRSPEC data.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.instrument = 'NIRSPEC SPEC'

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
            return self.get('DATAFILE')
        return KOAID

    def exptime(self):
        """Return the exposure time in seconds.
        """
        exptime = float(self.get('TRUITIME')) * int(self.get('COADDONE'))
        return exptime

    def obstime(self):
        date = self.get('DATE-OBS', mode=str)
        time = self.get('UTC', mode=str)
        return f"{date}T{time}"

    def readout_mode(self):
        nreads = self.get('READDONE', mode=int)
        translator = {2: 'CDS', 16: 'MCDS16'}
        try:
            mode = translator[nreads]
        except KeyError:
            mode = f'NREADS = {nreads}'
        return mode

    def minitime(self):
        nreads = self.get('READDONE', mode=int)
        readtime = self.get('READTIME', mode=float)
        minitime = np.ceil(nreads/2 * readtime)
        return minitime
