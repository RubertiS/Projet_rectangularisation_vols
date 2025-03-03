import numpy as np
import pandas as pd
import re
import ruptures as rpt

def compute_points(df, n_points, method = "ruptures", col = "ALT[m]"):
    '''
    Fonction qui détecte les points de changement de phase pour un vol donné.
    
    Arguments :
        df : pd.DataFrame, contenant les données du vol
        n_points : int, nombre de points à détecter (n_points + 1 phases)
        method : string, méthode pour détecter les points de changement de phase
    
    Returns :
        change_points : numpy array, contenant les points de changement de phase.
    '''
    altitude = df[col].values  
    d_altitude = np.gradient(altitude)
    if method == "ruptures" :       
        algo = rpt.Binseg(model="rbf").fit(np.column_stack((altitude, d_altitude)))
        change_points = [0] + algo.predict(n_bkps=n_points)

    if method == "trivial":
        change_points = [0] + [len(df)-1]

    if method == "ptrivial":
        change_points = [0] + [int(np.min(np.argwhere(altitude>=0.5))), int(np.max(np.argwhere(altitude>=0.5)))] + [len(df)-1]
    return change_points

def synchronize_flights(ds, method="ruptures", n_points=6, subset_fraction=0.1, col="ALT[m]"):

    """
    Synchronisation des vols par rééchantillonnage basé sur les phases détectées.

    Arguments :
        ds : list of pd.DataFrame - Liste des vols à synchroniser.
        method : str - Méthode de détection des phases (ex: "ruptures").
        n_points : int - Nombre de points de changement de phase à détecter.
        subset_fraction : float - Fraction de données à conserver.
        col : str - Colonne utilisée pour la détection des phases.

    Returns :
        R - Dictionnaire de DataFrames synchronisés.
    """

    ds = [df.iloc[::max(1, int(1 / subset_fraction))].copy() for df in ds] 
    cols = [re.sub(r'\[.*\]', '', col) for col in ds[0].columns]
    R = {c: [] for c in cols}
    rec = []

    all_phase_lengths = []
    all_change_points = []
    for df in ds:
        change_points = compute_points(df, n_points,method=method, col=col)
        all_change_points.append(change_points)
        phase_lengths = np.diff(change_points)  
        all_phase_lengths.append(phase_lengths)

    NL = np.round(np.mean(np.vstack(all_phase_lengths), axis=0)).astype(int)
    for df, change_points in zip(ds, all_change_points):
        rec.append(df.index.name)
        for V, C in zip(cols, df.columns):
            x = df[C].values
            Y = []

            for i in range(len(change_points) - 1):
                t0, t1 = change_points[i], change_points[i + 1]

                if t1 > len(x) - 1:
                    t1 = len(x) - 1

                new_t = np.linspace(t0, t1, NL[i])

                y_interp = np.interp(new_t, np.arange(t0, t1 + 1), x[t0:t1 + 1])
                Y.append(y_interp)

            R[V].append(np.hstack(Y))

    
    for V in cols:
        R[V] = pd.DataFrame(data=np.vstack(R[V]).T, columns=rec)
    
    return R, all_change_points,ds
