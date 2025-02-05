import numpy as np
import sys
sys.path.append('/home/vittorioinaf/Documenti/Programmi/Python')
from meanerr2 import meanerr2
from biweight import biweight

def magmed(mag2, err2=None, median_or_mean=False, do_biweight=0, biweight_sigma=3.):
     
    # do_biweight
    # 1: SÌ, CON MEDIA E SIGMA BEVINGTON
    # 2: SÌ, CON MEDIANA E SIGMA MEDIANA PESATA
    
    if err2 is None:
        flux=10.**(-mag2/2.5)
        magmeanflux = -2.5 *np.log10(sum(flux)/len(flux))

        return magmeanflux
    
    else:
  
        if do_biweight != 0:
            ind_biweight = biweight(mag2,biweight_sigma,do_biweight)
            mag2 = mag2[ind_biweight]
            err2 = err2[ind_biweight]

        flux = 10**(-mag2/2.5)
        errflux = flux*np.log(10)*.4*err2
        
        sss = meanerr2(flux,errflux)
        
        if ~median_or_mean:
            meanflux = sss[4]
            errmeanflux = sss[5]
        else:
            meanflux = sss[0]
            errmeanflux = sss[1]

        magmeanflux = -2.5 * np.log10(meanflux)
        error_on_flux = errmeanflux*2.5/(np.log(10)*meanflux)

        return [magmeanflux,error_on_flux]
