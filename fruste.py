# Fonction de sous-échantillonnage régulier et transformation des données.

import re
import numpy as np
import pandas as pd
import tabata as tbt

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
        t = pd.to_datetime(np.linspace(t0, t1, N))
        
        for V, C in zip(cols, df.columns):
            x = df[C].values
            
            # Vérifier si les données sont numériques
            if np.issubdtype(x.dtype, np.number):  # Vérifie si les données sont numériques
                # Etape de normalisation.
                x0 = np.max(x)
                x1 = np.min(x)
                xn = (x - x0) / (x1 - x0)  # Normalisation des données
                # Etape d'interpolation.
                y = np.interp(t, df.index, xn)
                R[V].append(y)
            else:
                # Si les données ne sont pas numériques, ajouter une valeur par défaut ou ignorer.
                print(f"Les données de la colonne {C} ne sont pas numériques.")
                R[V].append(np.zeros(N))  # Utilisation de zeros si les données ne sont pas numériques
    
    # Transformation en tables.
    for V in cols:
        R[V] = pd.DataFrame(data=np.vstack(R[V]).T, columns=rec)

    return R


