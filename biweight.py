import numpy as np
from meanerr2 import meanerr2

def biweight(array_to_clip,sigmasoglia,method=2,silent=False):

# method
#; 1: Si', CON MEDIA E SIGMA BEVINGTON
#; 2: Si', CON MEDIANA E SIGMA MEDIANA PESATA
    
    esclusi=1 #;valore iniziale per iterare almeno una volta
    sss=0 #;contatore iterazioni
    
    n_iniziale=len(array_to_clip)
    ind_nonesclusi=np.arange(0,n_iniziale-1)
    temp_array_to_clip=array_to_clip
    ind_nonesclusi_old=np.arange(0,n_iniziale)
    
    while (esclusi != 0):
        n_nonesclusi_inizio=len(temp_array_to_clip)
        #xmean,xsigma,xmedian,xsigmamedian=res=meanerr2(temp_array_to_clip,np.repeat(1.,n_nonesclusi_inizio))[[0,2,4,5]]
        xmean,_,xsigma,_,xmedian,xsigmamedian,_,_=meanerr2(temp_array_to_clip,np.repeat(1.,n_nonesclusi_inizio))
           #togliere [ind_nonesclusi] e le righe sopra e sotto per non clippare piu'
          
        if method == 1: 
            media_biweight=xmean
            sigma_biweight=xsigma
        elif method == 2:
            media_biweight=xmedian
            sigma_biweight=xsigmamedian
        
        ind_nonesclusi=(np.where(abs(temp_array_to_clip - media_biweight) <= sigmasoglia*sigma_biweight))[0]
        nonesclusi=len(ind_nonesclusi)
        
        #print range(n_nonesclusi_inizio)
        #print set(range(n_nonesclusi_inizio))
        #print ind_nonesclusi
        #print set(ind_nonesclusi)
        
        ind_esclusi=list(set(range(n_nonesclusi_inizio)) - set(ind_nonesclusi))
        esclusi=len(ind_esclusi)
        
        #print ind_nonesclusi_old
        
        ind_nonesclusi_absolute=ind_nonesclusi_old[ind_nonesclusi]
        ind_nonesclusi_old=ind_nonesclusi_absolute
        
        temp_array_to_clip=temp_array_to_clip[ind_nonesclusi]

        if silent:
            print(' sigma='+str(xsigmamedian)+'; rimasti: '+str(nonesclusi)+'/'+str(n_iniziale))
            print(media_biweight)
            print(sigma_biweight)
            print(temp_array_to_clip)
            #ind_nonesclusi_old=ind_nonesclusi
        sss=sss+1
    
    #ind_esclusi_absolute=cgsetdifference(indgen(n_iniziale),ind_nonesclusi_absolute)
    #print(array_to_clip[ind_nonesclusi_absolute])
    
    return(ind_nonesclusi_absolute)

def biweight_fit(xfit, array_to_clip, errors, order, sigmasoglia, 
                 option1=0, silent=True):
    
    esclusi=1 #valore iniziale per iterare almeno una volta
    sss=0 #contatore iterazioni

    n_iniziale = len(array_to_clip)
    ind_nonesclusi = np.arange(n_iniziale)
    ind_nonesclusi_old = np.arange(n_iniziale)
    temp_array_to_clip = array_to_clip
    temp_xfit = xfit

    while esclusi != 0:
        
        resk = np.polyfit(temp_xfit, temp_array_to_clip, order, w=1/np.asarray(errors)**2)

        yfit = resk[-1]
        for deg in np.arange(order):
            yfit += resk[deg] * temp_xfit**(order-deg)
            
        xmean,_,xsigma,_,xmedian,xsigmamedian,_,_ = meanerr2(temp_array_to_clip - yfit, 
                                                             np.asarray(errors))

    #     option1 serve nel caso in cui abbiamo un fit periodico o qualcosa
    #     che fitta una curva di luce. Il criterio di esclusione cambia e
    #     si tiene conto dell'ampiezza della funzione fittante. Vengono
    #     quindi rigettati quei punti che stanno troppo lontani rispetto all'ampiezza
    #     della funzione fittante.

        if option1 == 1:
            threshold=0.33*(max(yfit)-min(yfit))
        else:
            threshold=sigmasoglia*xsigmamedian
        
        ind_nonesclusi=(np.where(abs(temp_array_to_clip - yfit) <= threshold))[0]
        
#         print(temp_array_to_clip)
#         print(xmedian)
#         print(yfit)
#         print('---')
        
#         print(np.mean(temp_array_to_clip - xmedian))
#         print(np.mean(temp_array_to_clip - yfit))
#         print(len(ind_nonesclusi))
        
        ind_nonesclusi_absolute = ind_nonesclusi_old[ind_nonesclusi]
        esclusi = len(ind_nonesclusi_old) - len(ind_nonesclusi_absolute)
        ind_nonesclusi_old=ind_nonesclusi_absolute
        
        temp_array_to_clip = temp_array_to_clip[ind_nonesclusi]
        errors = errors[ind_nonesclusi]
        temp_xfit = temp_xfit[ind_nonesclusi]
        
        if not silent: 
            print(' sigma='+str('{:9.6f}'.format(xsigmamedian))+'; rimasti: '+
                  str(len(ind_nonesclusi))+'/'+str(n_iniziale))
        sss = sss + 1
        
        print(resk)
    
#     ind_nonesclusi=where(abs([temp_array_to_clip-ynew] - xmedian) le threshold,nonesclusi,compl=ind_esclusi,ncompl=esclusi)
    
    
    return {'ind': ind_nonesclusi_absolute, 'resk': resk, 'yfit': yfit, 
           'x_clipped': temp_xfit, 'y_clipped': temp_array_to_clip, 'xsigmamedian': xsigmamedian}
