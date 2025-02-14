# Fonction de sous-échantillonnage régulier et transformation des données.

import re
import numpy as np
import pandas as pd

def frust_nrm(ds, N):
    '''
    Cette fonction prend en entrée un dataset (Opset) tabata et produit une série de matrices de données renormalisées et rééchantillonnées avec un nombre de point donné.

    Entrées :
        * ds - Opset
        * N  - Taille du rééchantillonnage.

    Sorties :
        * L  - liste de matrices de taille Nxr où r est le nombre de signaux de l'opset ds.

    *Warning*
        Pas encore de stockage temporaire. Ce sera peut-être nécessaire en fonction du nombre de vols.
    '''

    cols = [re.sub('\[.*\]','', col) for col in ds.df.columns]
    R = {c : [] for c in cols}
    rec = []
    for df in ds:
        rec.append(df.index.name)
        t0 = df.index[0].value
        t1 = df.index[-1].value
        t = pd.to_datetime(np.linspace(t0,t1,N))
        for V,C in zip(cols,df.columns):
            x = df[C].values
            # Etape de normalisation.
            x0 = np.min(x)
            x1 = np.max(x)
            xn = (x-x0)/(x1-x0)
            # Etape d'interpolation.
            y = np.interp(t,df.index,xn)
            R[V].append(y)
    
    # Transformation en tables.
    for V in cols:
        R[V] = pd.DataFrame(data=np.vstack(R[V]).T, columns=rec)

    return R

def compute_points(df):
    '''
    Calcule les points de changement de phase.
    '''
    y = df['ALT[m]'].values
    n = len(y)
    t = np.arange(0,n)
     # Début de montée.
    t1 = np.max(np.argwhere((y<y[0]+10) & (t<n/3)))
    # Fin de la descente
    t4 = np.min(np.argwhere((y<y[-1]+10) & (t>2*n/3)))
    # Fin de la montée.
    
    dy = np.append(np.diff(y),0)
    k = 3
    dyc = np.convolve(dy,np.ones(2*k+1)/(2*k+1))
    dyc = dyc[k:-k]
    mx = np.max(y)
    tx = np.argwhere(y==mx)
    txm = (tx[0]+tx[-1])/2
    # Fin de la montée.
    t2 = np.min(np.argwhere((y>mx-mx/10) & (dyc<2.5) & (t>t1) & (t<txm)))
    # Début de la descente.
    t3 = np.max(np.argwhere((y>mx-mx/10) & (dyc>-2.5) & (t<t4) & (t>txm)))

    return t1,t2,t3,t4

def compute_all_sizes(ds):
    '''
    Calcule tous les points de changement de phase.
    '''
    P = []
    for df in ds:
        t1,t2,t3,t4 = compute_points(df)
        P.append([t1,t2-t1,t3-t2,t4-t3,len(df)-t4])
    return np.array(P)

def frust_phase_nrm(ds, NL):
    '''
    FRUST_PHASE_NRM - Synchronysation par rééchantillonnage maximal par phase.

    Entrées :
        * ds  - Opset
        * NL  - Taille du rééchantillonnage par phase.

    Sorties :
        * L  - liste de matrices de taille Nxr où r est le nombre de signaux de l'opset ds.

    *Warning*
        Pas encore de stockage temporaire. Ce sera peut-être nécessaire en fonction du nombre de vols.
    '''

    cols = [re.sub('\[.*\]','', col) for col in ds.df.columns]
    R = {c : [] for c in cols}
    rec = []
    for df in ds:
        rec.append(df.index.name)
        I = compute_points(df)
        I = [0] + list(I) + [len(df)-1]
        for V,C in zip(cols,df.columns):
            x = df[C].values
            # Etape de normalisation.
            x0 = np.min(x)
            x1 = np.max(x)
            xn = (x-x0)/(x1-x0)
            # Etape d'interpolation.
            Y = np.array(xn[0])
            for i in range(len(I)-1):
                t0 = df.index[I[i]].value
                t1 = df.index[I[i+1]].value
                t = pd.to_datetime(np.linspace(t0,t1,NL[i]))
                y = np.interp(t,df.index[I[i]:I[i+1]],xn[I[i]:I[i+1]])
                Y = np.hstack([Y,y[1:]])
            R[V].append(Y)

    # Transformation en tables.
    for V in cols:
        R[V] = pd.DataFrame(data=np.vstack(R[V]).T, columns=rec)

    return R
