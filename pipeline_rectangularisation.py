
# Projet : Rectangularisation des signaux d'altitude

## Étape 1 : Chargement des données
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error

def load_altitude_from_record(h5_path, key):
    with h5py.File(h5_path, 'r') as f:
        group = f[key]
        cols = [c.decode() for c in group['block0_items'][:]]
        values = group['block0_values'][:]
        df = pd.DataFrame(values, columns=cols)
    return df['ALT[m]'].values

## Étape 2 : Normalisation et métriques de synchronisation

def normalize_signal(signal, target_length=200):
    x_original = np.linspace(0, 1, len(signal))
    f = interp1d(x_original, signal, kind='linear')
    x_new = np.linspace(0, 1, target_length)
    return f(x_new)

def compute_mad_to_median(signals_matrix):
    median_curve = np.median(signals_matrix, axis=0)
    mad = np.mean(np.abs(signals_matrix - median_curve), axis=1)
    return np.mean(mad)

def compute_envelope_width(signals_matrix):
    return np.mean(np.max(signals_matrix, axis=0) - np.min(signals_matrix, axis=0))

## Étape 3 : Rectangularisation simple

def rectangularize_simple(signal):
    x = np.linspace(0, 1, len(signal))
    f = interp1d([x[0], x[-1]], [signal[0], signal[-1]])
    return f(x)

## Étape 4 : Visualisation des signaux

def plot_superposed_signals(S, title="Signaux normalisés", show_median=True):
    plt.figure(figsize=(12, 5))
    for s in S:
        plt.plot(s, color='gray', alpha=0.3)
    if show_median:
        median_curve = np.median(S, axis=0)
        plt.plot(median_curve, color='red', linewidth=2, label='Médiane')
    plt.title(title)
    plt.xlabel("Temps normalisé")
    plt.ylabel("Altitude (m)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

## Étape 5 : Pipeline d'exécution

h5_path = "AFL1EB_cleaned_final.h5"
record_keys = [f"record_{i:02d}" for i in range(5)]

list_of_signals = [load_altitude_from_record(h5_path, key) for key in record_keys]
S = np.array([normalize_signal(sig) for sig in list_of_signals])
list_of_signals_rect = [rectangularize_simple(sig) for sig in list_of_signals]
S_rect = np.array([normalize_signal(sig) for sig in list_of_signals_rect])

mad_sync = compute_mad_to_median(S)
env_sync = compute_envelope_width(S)
mad_sync_rect = compute_mad_to_median(S_rect)
env_sync_rect = compute_envelope_width(S_rect)

print("————————————————————————————————————————")
print(f"Synchronisation ORIGINALE (MAD à la médiane) : {mad_sync:.2f} m")
print(f"Synchronisation ORIGINALE (enveloppe moyenne) : {env_sync:.2f} m")
print()
print(f"Synchronisation RECTANGULARISÉE (MAD à la médiane) : {mad_sync_rect:.2f} m")
print(f"Synchronisation RECTANGULARISÉE (enveloppe moyenne) : {env_sync_rect:.2f} m")
print("————————————————————————————————————————")

plot_superposed_signals(S, title="Signaux normalisés — Originaux")
plot_superposed_signals(S_rect, title="Signaux normalisés — Rectangularisés")
