from .core import *


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
