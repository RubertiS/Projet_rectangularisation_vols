# Importation des bibliothèques nécessaires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fastdtw import fastdtw
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
#from tslearn.barycenters import dtw_barycenter_averaging
from scipy.interpolate import interp1d


def dtw_distance(series1, series2, window_ratio=0.02):
    """
    Calcule la distance DTW entre deux séries temporelles avec une contrainte de Sakoe-Chiba.
    """
    window = int(window_ratio * max(len(series1), len(series2)))
    distance, _ = dtw.warping_paths(series1, series2, window=window)
    return distance

def dtw_path(series1, series2, window_ratio=0.02):
    """
    Retourne le chemin optimal DTW entre deux séries temporelles avec une contrainte de Sakoe-Chiba.
    """
    window = int(window_ratio * max(len(series1), len(series2)))
    _, paths = dtw.warping_paths(series1, series2, window=window)
    return dtw.best_path(paths)

def choose_reference_curve(series_list):
    """
    Sélectionne la courbe de référence parmi une liste de séries temporelles par minimisation de la distance DTW.
    """
    min_distance = float('inf')
    best_ref = None
    
    for ref in series_list:
        total_distance = sum(dtw_distance(ref, s) for s in series_list if not np.array_equal(ref, s))
        if total_distance < min_distance:
            min_distance = total_distance
            best_ref = ref
    
    return best_ref

def calculate_dtw(reference, target, col="ALT[m]", sens=1, window_ratio=0.02):
    """
    Calcule la distance DTW et le chemin d'alignement entre la série de référence et la cible.
    """
    ref_values = reference[col].values[::-1] if sens != 1 else reference[col].values
    tgt_values = target[col].values[::-1] if sens != 1 else target[col].values
    
    path = dtw_path(ref_values, tgt_values, window_ratio)
    return dtw_distance(ref_values, tgt_values, window_ratio), path

def align_with_dtw(reference, target, path):
    """
    Aligne la série cible sur la temporalité de la référence.
    """
    ref_indices = np.arange(len(reference))
    aligned_target = pd.DataFrame(index=ref_indices, columns=target.columns)
    
    for ref_idx, tgt_idx in path:
        aligned_target.loc[ref_idx] = target.iloc[tgt_idx]
    
    return aligned_target.ffill().bfill()

def interpolate_series(series, new_length):
    """
    Interpolation linéaire pour adapter la série à une nouvelle longueur.
    """
    x_old = np.linspace(0, 1, len(series))
    x_new = np.linspace(0, 1, new_length)
    interpolator = interp1d(x_old, series, kind='linear', fill_value='extrapolate')
    return interpolator(x_new)

def plot_dtw_alignment(reference, target, aligned_target, col="ALT[m]", show_warping=False):
    """
    Visualise l'alignement DTW des séries temporelles.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(reference[col], label="Référence", linewidth=2)
    plt.plot(target[col], label="Cible originale", linestyle="--", alpha=0.7)
    plt.plot(aligned_target[col], label="Cible alignée", linewidth=1.5)
    plt.title("Alignement temporel DTW")
    plt.xlabel("Indices")
    plt.ylabel(col)
    plt.legend()
    plt.grid(alpha=0.5)
    plt.show()
    
    if show_warping:
        dtwvis.plot_warpingpaths(reference[col].values, target[col].values, 
                                 dtw.warping_paths(reference[col].values, target[col].values)[1],
                                 dtw.best_path(dtw.warping_paths(reference[col].values, target[col].values)[1]))
