from astropy.io import fits
from astropy.table import Table
hdul = fits.open('apogee.fits')
data = hdul[1].data
#table_data = Table(data)
#print(table_data)
#print(dir(data))
#hdul.info()
#dir(hdul) = ['__str__', _data', '_file', '_read_all', '_read_next_hdu',  'fileinfo', 'filename', 'fromfile', 'fromstring', 'index', 'index_of', 'info', 'readall',  'writeto']
#hdul[0] = ['__dict__', '__str__', '_bitpix', '_get_raw_data', '_header',  'data', 'filebytes', 'fileinfo', 'fromstring', 'header',  'writeto']

#hdul[1]
'''
TTYPE1  = 'APSTAR_ID'          /                                                
TTYPE2  = 'TARGET_ID'          /                                                
TTYPE3  = 'ASPCAP_ID'          /
TTYPE7  = 'LOCATION_ID'        /
TTYPE15 = 'RA      '           /                                                
TTYPE16 = 'DEC     '           /
TTYPE24 = 'SNR     '           /
TTYPE37 = 'RV_TEFF '           /                                                
TTYPE38 = 'RV_LOGG '           /                                                
TTYPE39 = 'RV_FEH  '           /
TTYPE59 = 'TEFF    '           /                                                
TTYPE60 = 'LOGG    '           /
TTYPE63 = 'TEFF_ERR'           /                                                
TTYPE64 = 'LOGG_ERR'           /
TTYPE75 = 'FE_H    '           /
TTYPE90 = 'FE_H_ERR'           /     
'''

'''
hdul[1].columns
ColDefs(
    name = 'APSTAR_ID'; format = '45A'
    name = 'TARGET_ID'; format = '34A'
    name = 'ASPCAP_ID'; format = '44A'
    name = 'FILE'; format = '34A'
    name = 'APOGEE_ID'; format = '18A'
    name = 'LOCATION_ID'; format = 'I'
    name = 'RA'; format = 'D'
    name = 'DEC'; format = 'D'
    name = 'SNR'; format = 'E'
    name = 'RV_TEFF'; format = 'E'
    name = 'RV_LOGG'; format = 'E'
    name = 'RV_FEH'; format = 'E'
    name = 'TEFF'; format = 'E'
    name = 'LOGG'; format = 'E'
    name = 'PARAM_M_H'; format = 'E'
    name = 'PARAM_ALPHA_M'; format = 'E'
    name = 'TEFF_ERR'; format = 'E'
    name = 'LOGG_ERR'; format = 'E'
    name = 'FE_H'; format = 'E'

)
'''
