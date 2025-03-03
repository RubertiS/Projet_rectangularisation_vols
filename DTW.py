import numpy as np
import pandas as pd
from fastdtw import fastdtw
from dtaidistance import dtw
import matplotlib.pyplot as plt
from tslearn.barycenters import dtw_barycenter_averaging
from scipy.interpolate import interp1d
from dtaidistance import dtw_visualisation as dtwvis

def calculate_dtw(reference, target, col="ALT[m]",sens =1):
    """
    Calcule le chemin d'alignement entre la série temporelle de référence et celle de la cible via DTW.

    Args:
        reference (pd.DataFrame): Données de référence.
        target (pd.DataFrame): Données de la cible.
        col (str): La colonne utilisée pour le DTW.

    Returns:
        tuple: (distance, path), où `distance` est la distance DTW et `path` est le chemin d'alignement.
    """
    
    ref_values = reference[col].values
    tgt_values = target[col].values
    if sens !=1 :
        ref_values = ref_values[::-1]
        tgt_values = tgt_values[::-1]
    # Calcul de DTW
    #distance, path = fastdtw(ref_values, tgt_values, radius = 1, dist = 2)
    distance, paths = dtw.warping_paths(ref_values, tgt_values,window = int(0.02*max(len(target),len(reference))))

    #distance, paths = dtw.warping_paths(reference[col], target[col],window = 1)
    path = dtw.best_path(paths)
    
    #dtwvis.plot_warpingpaths(ref_values,tgt_values,paths,path)

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
    ref_indices = np.arange(len(reference))

    aligned_target = pd.DataFrame(index=ref_indices, columns=target.columns)

    for ref_idx, tgt_idx in path:
        aligned_target.loc[ref_idx] = target.iloc[tgt_idx]

    aligned_target = aligned_target.ffill().bfill()
    return aligned_target



def plot_dtw_alignment(reference, target, aligned_target, col="ALT[m]"):
    """
    Visualise le résultat de l'alignement temporel DTW.

    Args:
        reference (pd.DataFrame): Données de référence.
        target (pd.DataFrame): Données de la cible originale.
        aligned_target (pd.DataFrame): Données de la cible alignée temporellement.
        col (str): La colonne utilisée pour le tracé.
    """
    ref_indices = np.arange(len(reference))
    tgt_indices = np.arange(len(target))
    aligned_indices = np.arange(len(aligned_target))

    plt.figure(figsize=(8, 4))

    plt.plot(ref_indices, reference[col], label="Référence", linewidth=2)
    plt.plot(tgt_indices, target[col], label="Cible originale", linestyle="--", alpha=0.7)
    plt.plot(aligned_indices, aligned_target[col], label="Cible alignée (temporel)", linewidth=1.5)

    plt.title("Alignement temporel DTW")
    plt.xlabel("Indices numériques")
    plt.ylabel(col)
    plt.legend()
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()
