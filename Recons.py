import numpy as np
import re

def reconstruct_flight_phased(aligned_dfs, original_flight, change_points):
    """
    Reconstruit un vol original à partir des données alignées tout en respectant les phases.

    Arguments :
        aligned_dfs : dict of pd.DataFrame - Dictionnaire contenant les données synchronisées.
        original_flight : pd.DataFrame - Données originales du vol (avant synchronisation).
        change_points : list - Points de changement de phase du vol original.

    Returns :
        reconstructed_df : pd.DataFrame - Vol reconstruit avec toutes les variables.
    """
    reconstructed_df = original_flight.copy()

    for col in original_flight.columns:  # Boucle sur chaque variable
        col_clean = re.sub(r'\[.*\]', '', col)  # Nettoyage du nom de la colonne
        if col_clean in aligned_dfs:  # Vérifie si la variable est présente dans les données synchronisées
            aligned_series = aligned_dfs[col_clean].iloc[:, 0]  # Sélectionne le vol correspondant dans sync_flights

            # On initialise la colonne avec NaN
            reconstructed_df[col] = np.nan  

            # Reconstruction phase par phase
            for i in range(len(change_points) - 1):
                t0, t1 = change_points[i], change_points[i + 1]

                if t1 > len(reconstructed_df) - 1:  
                    t1 = len(reconstructed_df) - 1  # Empêche d'aller hors limites

                if i >= len(aligned_series):
                    continue  # Évite les erreurs d'index hors limites

                # Indices alignés correspondant à cette phase
                phase_start = int(i * (len(aligned_series) / (len(change_points) - 1)))
                phase_end = int((i + 1) * (len(aligned_series) / (len(change_points) - 1)))

                aligned_phase = aligned_series.iloc[phase_start:phase_end]

                if len(aligned_phase) < 2:
                    continue  # Évite les erreurs avec trop peu de points

                # Interpolation pour remettre aux indices originaux
                new_t = np.linspace(t0, t1, len(aligned_phase))
                reconstructed_df[col].iloc[t0:t1] = np.interp(
                    np.arange(t0, t1),
                    new_t,
                    aligned_phase.values
                )

    return reconstructed_df
