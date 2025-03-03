import pandas as pd
import numpy as np
from tabata.opset import Opset
import pandas as pd
from scipy.signal import find_peaks

def clean_and_normalize_data(input_file, output_file):
    if input_file == output_file:
        raise ValueError("Le fichier d'entrée et le fichier de sortie ne peuvent pas être les mêmes.")

    ds = Opset(input_file)
    print(f"Taille du df avant traitement : {len(ds)}")

    clean_ds = Opset(output_file)  
    clean_ds.clean()

    for idx, df in enumerate(ds.iterator()):
        # Détection de l'unité et conversion en mètres si nécessaire
        if "ALT[m]" in df.columns:
            altitude = df["ALT[m]"]
        elif "ALT [ft]" in df.columns:
            altitude = df["ALT [ft]"] * 0.3048
            df["ALT[m]"] = altitude
        else:
            print(f"Pas de colonne d'altitude détectée pour Vol #{idx}, passage au suivant.")
            continue

        if max(altitude) <= 1000:
            print(f"Vol exclu (altitude max ≤ 1000) : Vol #{idx}")
            continue

        # Vérifier si l'altitude finale est au-dessus de 500 m
        if altitude.iloc[-1] > 3000:
            print(f"Vol exclu (altitude finale > 500m) : Vol #{idx}")
            continue

        # Vérification de la stabilité en croisière (30% à 70% du vol)
        start_idx = int(0.4 * len(altitude))
        end_idx = int(0.6 * len(altitude))
        alt_croisière = altitude.iloc[start_idx:end_idx]

        # Trouver les creux dans cette zone
        minima_indices, _ = find_peaks(-alt_croisière, prominence=100)

        for idx_min in minima_indices:
            altitude_min = alt_croisière.iloc[idx_min]

            # Vérifier si l'altitude minimale descend sous 60% de l'altitude max
            if altitude_min < 0.7 * max(altitude):
                # Vérifier s'il y a une remontée d'au moins 500 m après ce creux
                if (alt_croisière.iloc[idx_min:] > altitude_min + 1000).any():
                    print(f"Vol exclu (chute brutale suivie d'une remontée en croisière) : Vol #{idx}")
                    continue

        if isinstance(df.index, pd.DatetimeIndex):
            x = df.index
            t = (x - x[0]).total_seconds()

            # Vérification de la régularité de l'horodatage
            dt = np.diff(x) / np.timedelta64(1, 's')
            dt_clean = dt[~np.isnan(dt)]  # Suppression des NaN

            if len(dt_clean) > 0 and any(dt_clean != dt_clean[0]):
                name = df.index.name
                median_dt = np.median(dt_clean)
                df.index = pd.date_range(start=x[0], periods=len(df), freq=pd.Timedelta(seconds=median_dt))
                df.index.name = name  

        # Normalisation de l'altitude
        min_val = altitude.min()
        max_val = altitude.max()
        df["ALT_norm"] = (altitude - min_val) / (max_val - min_val)

        # Détection et suppression des faux atterrissages
        minima_indices, _ = find_peaks(-altitude, prominence=100)
        has_fake_landing = any(
            idx_min != 0 and idx_min != len(altitude) - 1 and 
            altitude.iloc[idx_min] < 500 and 
            (altitude.iloc[idx_min:] > altitude.iloc[idx_min] + 1000).any()
            for idx_min in minima_indices
        )

        if has_fake_landing:
            print(f"Vol exclu (faux atterrissage détecté) : Vol #{idx}")
            continue
        
        clean_ds.put(df)

    print(f"Taille du df après traitement : {len(clean_ds)}")
    print(f"Qualité des données : {100*len(clean_ds)/len(ds):.2f}%")
    print(f"Fichier nettoyé sauvegardé sous {output_file}")
