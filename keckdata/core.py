#!/usr/env/python

## Import General Tools
from pathlib import Path, PosixPath

import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.nddata import NDData, StdDevUncertainty, CCDData
from astropy.table import Table


##-------------------------------------------------------------------------
## Handle FITS Section Strings in Headers
##-------------------------------------------------------------------------
def split_fits_section(fitssec):
    '''Split a fits section string such as "[100:300,600:900]" to a dictionary.
    '''
    xr, yr = fitssec.strip('[').strip(']').split(',')
    x1, x2 = xr.split(':')
    y1, y2 = yr.split(':')
    cutout = {'x1': min([ int(x1), int(x2) ]),
              'x2': max([ int(x1), int(x2) ]),
              'y1': min([ int(y1), int(y2) ]),
              'y2': max([ int(y1), int(y2) ]),
              'xreverse': (x1 > x2),
              'yreverse': (y1 > y2),
              }
    return cutout


##-------------------------------------------------------------------------
## KeckData Exceptions
##-------------------------------------------------------------------------
class KeckDataError(Exception):
    """Base class for exceptions in this module."""
    pass


class IncompatiblePixelData(KeckDataError):
    """Raise when trying to operate on multiple KeckData
    objects which have incompatible pixeldata.
    """
    def __init__(self, message):
        msg = f"KeckData objects have incompatible pixeldata. {message}"
        super().__init__(msg)


class IncorrectNumberOfExtensions(KeckDataError):
    """Raise when verify method fails for a specific instrument.
    """
    def __init__(self, datatype, expected, kd):
        msg = (f"Incorrect number of {datatype} entries.  "
               f"Expected {expected} for {type(kd)}")
        super().__init__(msg)


class UnableToParseInstrument(KeckDataError):
    """Raise when failing to parse INSTRUME header keyword.
    """
    def __init__(self, message):
        msg = f"Unable to determine instrument. {message}"
        super().__init__(msg)


##-------------------------------------------------------------------------
## KeckData Classes
##-------------------------------------------------------------------------
class KeckData(object):
    """Our data model.
    
    Attributes:
    pixeldata -- a list of CCDData objects containing pixel values.
    tabledata -- a list of astropy.table.Table objects
    headers -- a list of astropy.io.fits.Header objects
    """
    def __init__(self, *args, **kwargs):
        self.pixeldata = []
        self.tabledata = []
        self.headers = []
        self.instrument = None
        self.xgap = 0
        self.ygap = 0

    def verify(self):
        """Method to check the data against expectations. For the 
        KeckData class this simply passes and does nothing, but
        subclasses for specific instruments can populate this
        with appropriate tests.
        """
        pass

    def add(self, kd2):
        """Method to add another KeckData object to this one and return
        the result.  This uses the CCDData object's add method and
        simply loops over all elements of the pixeldata list.
        """
        if issubclass(type(kd2), KeckData):
            if len(self.pixeldata) != len(kd2.pixeldata):
                raise IncompatiblePixelData
            for i,pd in enumerate(self.pixeldata):
                self.pixeldata[i] = pd.add(kd2.pixeldata[i])
        elif type(kd2) in [float, int]:
            self.pixeldata[i] = pd.add(kd2)
        return self

    def subtract(self, kd2):
        """Method to subtract another KeckData object to this one
        and return the result.  This uses the CCDData object's
        subtract method and simply loops over all elements of
        the pixeldata list.
        """
        if issubclass(type(kd2), KeckData):
            if len(self.pixeldata) != len(kd2.pixeldata):
                raise IncompatiblePixelData
            for i,pd in enumerate(self.pixeldata):
                self.pixeldata[i] = pd.subtract(kd2.pixeldata[i])
        elif type(kd2) in [float, int]:
            self.pixeldata[i] = pd.subtract(kd2)
        return self

    def multiply(self, kd2):
        """Method to multiply another KeckData object by this one
        and return the result.  This uses the CCDData object's
        multiply method and simply loops over all elements of
        the pixeldata list.
        """
        if issubclass(type(kd2), KeckData):
            if len(self.pixeldata) != len(kd2.pixeldata):
                raise IncompatiblePixelData
            for i,pd in enumerate(self.pixeldata):
                self.pixeldata[i] = pd.multiply(kd2.pixeldata[i])
        elif type(kd2) in [float, int]:
            self.pixeldata[i] = pd.multiply(kd2)
        return self

    def get(self, kw, mode=None):
        """Method to loop over all headers and get the specified keyword value.
        Returns the first result it finds and doe not check for duplicate
        instances of the keyword in subsequent headers.
        """
        for hdr in self.headers:
            val = hdr.get(kw, None)
            if val is not None:
                if mode is not None:
                    assert mode in [str, float, int, bool]
                    if mode is bool and type(val) is str:
                        if val.strip().lower() == 'false':
                            return False
                        elif val.strip().lower() == 'true':
                            return True
                    elif mode is bool and type(val) is int:
                        return bool(int(val))
                    # Convert result to requested type
                    try:
                        return mode(val)
                    except ValueError:
                        print(f'WARNING: Failed to parse "{val}" as {mode}')
                        return val
                else:
                    return val
        return None

    def type(self):
        """Return the image type.
        
        BIAS, DARK, INTFLAT, ARC, FLAT, FLATOFF, OBJECT
        """
        return None

    def exptime(self):
        """Return the exposure time in seconds.
        """
        return self.get('EXPTIME')

    def binning(self):
        """Return the exposure binning as an (x, y) tuple.
        """
        binning_str = self.get('BINNING')
        bx, by = binning_str.split(',')
        return ( int(bx.strip()), int(by.strip()) )

    def filename(self):
        """Return a string with the filename.
        
        Returns KOAID if that is present (i.e. this file has been processed
        through KOA).  Otherwise return the DATAFILE value.
        """
        KOAID = self.get('KOAID', None)
        if KOAID is None:
            DATAFILE = f"{self.get('DATAFILE', None)}.fits"
            return DATAFILE
        else:
            return KOAID

    def obstime(self):
        """Return a string with the UT date and time of the observation.
        """
        return self.get('DATE', None)

    def iraf_mosaic(self, fordisplay=True, zero=True, xgap=None, ygap=None):
        '''Using the DETSEC and DATASEC keywords in the header, form a mosaic
        version of the data with all pixeldata arrays combined in to a single
        CCDData object.
        '''
        binx, biny = self.binning()
        if xgap is None:
            xgap = int(self.xgap/binx)
        if ygap is None:
            ygap = int(self.ygap/biny)
        CCDs = {}
        for i,pd in enumerate(self.pixeldata):
            CCDNAME = pd.header.get('CCDNAME')
            EXTNAME = pd.header.get('EXTNAME')
            DETSEC = pd.header.get('DETSEC')
            DATASEC = pd.header.get('DATASEC')
            if CCDNAME not in CCDs.keys():
                CCDs[CCDNAME] = {'CCDNAME': CCDNAME,
                                 'EXTNAMES': [EXTNAME],
                                 'DETSECS': [DETSEC],
                                 'DATASECS': [DATASEC],
                                 'PDid': [i],
                                }
            else:
                CCDs[CCDNAME]['EXTNAMES'].append(EXTNAME)
                CCDs[CCDNAME]['DETSECS'].append(DETSEC)
                CCDs[CCDNAME]['DATASECS'].append(DATASEC)
                CCDs[CCDNAME]['PDid'].append(i)

        # Form an intermediate CCDSEC which is the position of the data
        # within each CCD (i.e. combine the amps)
        for CCD in CCDs.keys():
            ccd_sec = None
            for j,extname in enumerate(CCDs[CCD]['EXTNAMES']):
                DETSEC = split_fits_section(CCDs[CCD]['DETSECS'][j])
                if ccd_sec is None:
                    ccd_sec = {'x1': DETSEC['x1'],
                               'x2': DETSEC['x2'],
                               'y1': DETSEC['y1'],
                               'y2': DETSEC['y2'],
                               }
                else:
                    ccd_sec['x1'] = min([ ccd_sec['x1'],  DETSEC['x1'] ])
                    ccd_sec['x2'] = max([ ccd_sec['x2'],  DETSEC['x2'] ])
                    ccd_sec['y1'] = min([ ccd_sec['y1'],  DETSEC['y1'] ])
                    ccd_sec['y2'] = max([ ccd_sec['y2'],  DETSEC['y2'] ])
            CCDs[CCD]['CCDSEC'] = ccd_sec

        # Figure out the grid parameters of the chips in the "detector" focal plane
        chips = []
        for CCD in CCDs.keys():
            CCDSEC = CCDs[CCD]['CCDSEC']
            chips.append( [CCD,
                           int(np.ceil(CCDSEC['x1']/binx)),
                           int(np.ceil(CCDSEC['x2']/binx)),
                           int(np.ceil(CCDSEC['y1']/biny)),
                           int(np.ceil(CCDSEC['y2']/biny)),
                          ] )
        chips.sort(key=lambda c: c[3])
        chips.sort(key=lambda c: c[1])

        x1s = sorted( list( set( [c[1] for c in chips] ) ) )
        y1s = sorted( list( set( [c[3] for c in chips] ) ) )
        chip_grid = (len(x1s), len(y1s))
        ngrid = len(x1s) * len(y1s)
        assert ngrid == len(CCDs)

#         if fordisplay is True: print(f"Scaling each FITS extension to the same mean")
#         if zero is True: print(f"Zeroing background level")

        # Using the CCDSEC info, form the data for each CCD chip
        unit = set([pd.unit for pd in self.pixeldata]).pop()
        meanlv = None
        for CCD in CCDs.keys():
            CCDSEC = CCDs[CCD]['CCDSEC']
            ccd_size_y = int((CCDSEC['y2'] - CCDSEC['y1'] + 1)/biny)
            ccd_size_x = int((CCDSEC['x2'] - CCDSEC['x1'] + 1)/binx)
            CCDs[CCD]['data'] = CCDData(data=np.zeros((ccd_size_y, ccd_size_x)), unit=unit )
            for j,extname in enumerate(CCDs[CCD]['EXTNAMES']):
                PDid = CCDs[CCD]['PDid'][j]
                DETSEC = split_fits_section(CCDs[CCD]['DETSECS'][j])
                DATASEC = split_fits_section(CCDs[CCD]['DATASECS'][j])
                imagesection = self.pixeldata[PDid][DATASEC['y1']-1:DATASEC['y2'], DATASEC['x1']-1:DATASEC['x2']]
                if DETSEC['xreverse'] is True:
                    imagesection.data = np.fliplr(imagesection.data)
                if DETSEC['yreverse'] is True:
                    imagesection.data = np.flipud(imagesection.data)
                if fordisplay is True:
                    if meanlv is None:
                        meanlv = np.percentile(imagesection.data, 0.1) if zero is False else 0
                    imagesection -= np.percentile(imagesection.data, 0.1) - meanlv
                DETSEC['x2'] -= (CCDSEC['x1']-1)
                DETSEC['x1'] -= (CCDSEC['x1']-1)
                DETSEC['y2'] -= (CCDSEC['y1']-1)
                DETSEC['y1'] -= (CCDSEC['y1']-1)
                CCDs[CCD]['data'].data[DETSEC['y1']-1:DETSEC['y2'], DETSEC['x1']-1:DETSEC['x2']] = imagesection.data

        # Assemble the "detector" mosaic
        for i,chip in enumerate(chips):
            CCD, CCDx1, CCDx2, CCDy1, CCDy2 = chip
            gridxpos = x1s.index(CCDx1)
            gridypos = y1s.index(CCDy1)
            CCDSEC = CCDs[CCD]['CCDSEC']
            CCDx1 += (2*gridxpos-1)*xgap if gridxpos > 0 else 0
            CCDx2 += (2*gridxpos+1)*xgap if gridxpos < len(x1s)-1 else (2*gridxpos)*xgap
            CCDy1 += (2*gridypos-1)*ygap if gridypos > 0 else 0
            CCDy2 += (2*gridypos+1)*ygap if gridypos < len(y1s)-1 else (2*gridypos)*ygap
            chips[i] = [CCD, CCDx1, CCDx2, CCDy1, CCDy2, gridxpos, gridypos]

        xmax = max([chip[2] for chip in chips])
        ymax = max([chip[4] for chip in chips])
        mosaic = CCDData(data=np.zeros((ymax, xmax)), unit=unit )
        for i,chip in enumerate(chips):
            CCD, CCDx1, CCDx2, CCDy1, CCDy2, gridxpos, gridypos = chip
            MOSx1 = CCDx1+xgap-1 if gridxpos > 0 else CCDx1-1
            MOSx1 = max( [MOSx1, 0] ) # Ensure we're above 0
            MOSx2 = MOSx1 + CCDs[CCD]['data'].data.shape[1]
            MOSy1 = CCDy1+ygap-1 if gridypos > 0 else CCDy1-1
            MOSy1 = max( [MOSy1, 0] ) # Ensure we're above 0
            MOSy2 = MOSy1 + CCDs[CCD]['data'].data.shape[0]
            mosaic.data[MOSy1:MOSy2,MOSx1:MOSx2] = CCDs[CCD]['data'].data
    
        self.mosaic = mosaic
        return self.mosaic


##-------------------------------------------------------------------------
## Get HDU Type
##-------------------------------------------------------------------------
def get_hdu_type(hdu):
    """Function to examine a FITS HDU object and determine its type.  Returns
    one of the following strings:
    
    'header' -- This is a PrimaryHDU or ImageHDU with no pixel data.
    'pixeldata' -- This is a PrimaryHDU or ImageHDU containing pixel data.
    'uncertainty' -- This is a pixeldata HDU which is associated with the
                     uncertainty data written by either CCDData or KeckData.
    'mask' -- This is a pixeldata HDU which is associated with the mask
              data written by either CCDData or KeckData.
    'tabledata' -- This is a TableHDU type HDU.
    """
    if type(hdu) in [fits.PrimaryHDU, fits.ImageHDU] and hdu.data is None:
        # This is a header only HDU
        return 'header'
    elif type(hdu) in [fits.PrimaryHDU, fits.ImageHDU] and hdu.data is not None:
        # This is a pixel data HDU
        extname = hdu.header.get('EXTNAME', '').strip()
        if extname == 'MASK':
            # This is a mask HDU
            return 'mask'
        elif extname == 'UNCERT':
            # This is an uncertainty HDU
            return 'uncertainty'
        else:
            # This must be pixel data
            return 'pixeldata'
    elif type(hdu) == fits.TableHDU:
            # This is table data
            return 'tabledata'


##-------------------------------------------------------------------------
## KeckData Reader
##-------------------------------------------------------------------------
def fits_reader(file, defaultunit='adu', datatype=None, verbose=False):
    """A reader for KeckData objects.
    
    Currently this is a separate function, but should probably be
    registered as a reader similar to fits_ccddata_reader.
    
    Arguments:
    file -- The filename (or pathlib.Path) of the FITS file to open.

    Keyword arguments:
    defaultunit -- If the BUNIT keyword is unable to be located or
                   parsed, the reader will assume this unit.  Defaults
                   to "adu".
    datatype -- The output datatype.  Defaults to KeckData, but could
                be a subclass such as MOSFIREData.  The main effect of
                this is that it runs the appropriate verify method on
                the data.
    """
    if type(file) is not Path:
        file = Path(file).expanduser()

    try:
        hdul = fits.open(file, 'readonly')
    except FileNotFoundError as e:
        print(e.msg)
        raise e
    except OSError as e:
        print(e.msg)
        raise e
    # Determine instrument
    if datatype is None:
        instrument = None
        while instrument is None:
            for hdu in hdul:
                if hdu.header.get('INSTRUME') is not None:
                    instrument = hdu.header.get('INSTRUME')
        if instrument is None:
            raise UnableToParseInstrument

        if instrument[:5] == 'HIRES':
            from .visible import HIRESData
            datatype = HIRESData
        elif instrument.strip() == 'LRISBLUE':
            from .visible import LRISBlueData
            datatype = LRISBlueData
        elif instrument.strip() == 'LRIS':
            from .visible import LRISRedData
            datatype = LRISRedData
        elif instrument[:6] == 'DEIMOS':
            from .visible import DEIMOSData
            datatype = DEIMOSData
        elif instrument.strip() == 'MOSFIRE':
            from .infrared import MOSFIREData
            datatype = MOSFIREData
        else:
            print(f'Using generic KeckData object for "{instrument}"')
            datatype = KeckData

    # Loop though HDUs and read them in as pixel data or table data
    kd = datatype()
    while len(hdul) > 0:
        if verbose: print('Extracting HDU')
        hdu = hdul.pop(0)
        kd.headers.append(hdu.header)
        hdu_type = get_hdu_type(hdu)
        if verbose: print(f'  Got HDU type = {hdu_type}')
        if hdu_type == 'header':
            pass
        elif hdu_type == 'tabledata':
            kd.tabledata.append(Table(hdu.data))
        elif hdu_type == 'pixeldata':
            # Check the next HDU
            mask = None
            uncertainty = None
            if len(hdul) > 0:
                next_type = get_hdu_type(hdul[0])
                if next_type == 'mask':
                    mask = hdul[0].data
                elif next_type == 'uncertainty':
                    uncertainty = hdul[0].data
            if len(hdul) > 1:
                next_type2 = get_hdu_type(hdul[1])
                if next_type2 == 'mask':
                    mask = hdul[1].data
                elif next_type2 == 'uncertainty':
                    uncertainty = hdul[1].data               
            # Sanitize "ADU per coadd" BUNIT value
            if hdu.header.get('BUNIT') == "ADU per coadd":
                hdu.header.set('BUNIT', 'adu')
            # Populate the CCDData object
            c = CCDData(hdu.data, mask=mask, uncertainty=uncertainty,
                        meta=hdu.header,
                        unit=hdu.header.get('BUNIT', defaultunit),
                       )
            kd.pixeldata.append(c)
    if verbose: print(f'Read in {len(kd.headers)} headers, '
                      f'{len(kd.pixeldata)} sets of pixel data, '
                      f'and {len(kd.tabledata)} tables')
    kd.verify()
    return kd


##-------------------------------------------------------------------------
## KeckDataList
##-------------------------------------------------------------------------
class KeckDataList(object):
    """An object to manage lists of KeckData objects.
    
    Attributes:
    kds -- a list of KeckData objects.
    len -- the number of KeckData objects in the list
    kdtype -- the type of KeckData object contained in the list.
              e.g. `KeckData`, `HIRESData`, `MOSFIREData`
    """
    def __init__(self, input, verbose=False):
        assert type(input) == list
        self.frames = []
        for item in input:
            if type(item) == str:
                p = Path(str)
                if p.exists():
                    try:
                        kd = fits_reader(p, verbose=verbose)
                        self.frames.append(kd)
                    except:
                        pass
            elif type(item) in [Path, PosixPath]:
                if item.exists():
                    try:
                        kd = fits_reader(item, verbose=verbose)
                        self.frames.append(kd)
                    except:
                        print(f"WARNING: Unable to read: {item}")
                        raise
                else:
                    print(f"WARNING: File not found: {item}")
            elif issubclass(type(item), KeckData):
                self.frames.append(item)

        self.len = len(self.frames)

        # Verify all input object have the same number of pixeldata arrays
        pixeldata_lengths = set([len(kd.pixeldata) for kd in self.frames])
        if len(pixeldata_lengths) > 1:
            raise IncompatiblePixelData(
                  'Input files have insconsistent pixel data ')

        # Determine which KeckData type this is
        kdtypes = set([type(kd) for kd in self.frames])
        if len(kdtypes) > 1:
            raise IncompatiblePixelData(
                  'Input KeckData objects are not all of the same type: {kdtypes}')
        self.kdtype = kdtypes.pop()

    def pop(self):
        '''Return one object from the list and remove it from the list.
        '''
        self.len -= 1
        return self.frames.pop()
        
