import numpy as np
import pandas as pd
from fastdtw import fastdtw
from dtaidistance import dtw
import matplotlib.pyplot as plt

def calculate_dtw(reference, target, value_col="ALT[m]"):
    """
    Calcule le chemin d'alignement entre la série temporelle de référence et celle de la cible via DTW.

    Args:
        reference (pd.DataFrame): Données de référence.
        target (pd.DataFrame): Données de la cible.
        value_col (str): La colonne utilisée pour le DTW.

    Returns:
        tuple: (distance, path), où `distance` est la distance DTW et `path` est le chemin d'alignement.
    """
    # Extraction des valeurs comme séries numériques
    ref_values = reference[value_col].values
    tgt_values = target[value_col].values
    
    # Calcul de DTW
    #distance, path = fastdtw(ref_values, tgt_values, radius = 1, dist = 2)
    
    distance, paths = dtw.warping_paths(reference[value_col], target[value_col])
    path = dtw.best_path(paths)
    
    return distance, path


def align_with_dtw(reference, target, path):
    """
    Réaligne la cible sur la référence uniquement temporellement en utilisant le chemin DTW.
    Les valeurs originales de la cible ne sont pas modifiées.

    Args:
        reference (pd.DataFrame): Données de référence.
        target (pd.DataFrame): Données de la cible.
        path (list): Le chemin DTW (indices correspondants entre la référence et la cible).

    Returns:
        pd.DataFrame: La cible alignée sur la temporalité de la référence.
    """
    # Indices numériques pour la référence
    ref_indices = np.arange(len(reference))

    # Créer un DataFrame aligné
    aligned_target = pd.DataFrame(index=ref_indices, columns=target.columns)

    # Remplir les lignes alignées selon le chemin DTW
    for ref_idx, tgt_idx in path:
        aligned_target.loc[ref_idx] = target.iloc[tgt_idx]

    # Remplir les valeurs manquantes si nécessaire
    aligned_target = aligned_target.fillna(method="ffill").fillna(method="bfill")
    return aligned_target



def plot_dtw_alignment(reference, target, aligned_target, value_col="ALT[m]"):
    """
    Visualise le résultat de l'alignement temporel DTW.

    Args:
        reference (pd.DataFrame): Données de référence.
        target (pd.DataFrame): Données de la cible originale.
        aligned_target (pd.DataFrame): Données de la cible alignée temporellement.
        value_col (str): La colonne utilisée pour le tracé.
    """
    # Indices numériques
    ref_indices = np.arange(len(reference))
    tgt_indices = np.arange(len(target))
    aligned_indices = np.arange(len(aligned_target))

    plt.figure(figsize=(12, 6))

    # Tracer les séries temporelles
    plt.plot(ref_indices, reference[value_col], label="Référence", linewidth=2)
    plt.plot(tgt_indices, target[value_col], label="Cible originale", linestyle="--", alpha=0.7)
    plt.plot(aligned_indices, aligned_target[value_col], label="Cible alignée (temporel)", linewidth=1.5)

    plt.title("Alignement temporel DTW")
    plt.xlabel("Indices numériques")
    plt.ylabel(value_col)
    plt.legend()
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()
