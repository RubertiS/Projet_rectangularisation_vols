import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from scipy.ndimage import gaussian_filter1d

def align_without_phases(reference, target):
    """
    Aligne les données de la cible sur la référence sans tenir compte des phases.
    
    Parameters:
        - reference (pd.DataFrame): Données de référence.
        - target (pd.DataFrame): Données de la cible à aligner.
        
    Returns:
        - pd.DataFrame: Données alignées de la cible.
    """
    ref_indices = np.linspace(0, 1, len(reference))
    tgt_indices = np.linspace(0, 1, len(target))
    
    interpolated_target = pd.DataFrame({
        col: np.interp(ref_indices, tgt_indices, target[col])
        for col in target.columns if target[col].dtype != 'object'  
    }, index=reference.index)
    
    return interpolated_target


def preprocess_data(data, subset_fraction=0.05, sigma=2):
    """
    Prétraite les données : sous-échantillonnage, filtrage gaussien, calcul des dérivées et extraction des colonnes.
    """
    # Sous-échantillonnage
    data = data.iloc[::int(1 / subset_fraction), :]

    # Filtrage gaussien pour réduire le bruit
    alt = gaussian_filter1d(data["ALT[m]"].values, sigma=sigma)
    vz = gaussian_filter1d(data["Vz[m/s]"].values, sigma=sigma)
    tas = gaussian_filter1d(data["TAS[m/s]"].values, sigma=sigma)
    #tisa = gaussian_filter1d(data["Tisa[K]"].values, sigma=sigma)
    #f = gaussian_filter1d(data["F[N]"].values, sigma=sigma)

    # Calcul des dérivées
    d_alt = np.gradient(alt)
    d2_alt = np.gradient(d_alt)  # Dérivée de l'altitude
    d_vz = np.gradient(vz)    # Dérivée de la vitesse verticale
    d_tas = np.gradient(tas)  # Dérivée de la vitesse sol

    # Retourner les observations sous forme de tableau
    #observations = np.column_stack([alt, vz, tas, d_alt, d_vz, d_tas])
    observations = np.column_stack([alt, gaussian_filter1d(d_alt,1), d2_alt])

    return data, observations

def create_hmm_model(n_states, random_state=42):
    """
    Crée un modèle HMM avec une matrice de transition contrainte.
    """
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=1000,
        tol=1e-4,
        init_params="mcs",
        random_state=random_state,
    )

    transition_matrix = np.zeros((n_states, n_states))
    for i in range(n_states - 1):
        transition_matrix[i, i] = 0.7  # Rester dans la même phase
        transition_matrix[i, i + 1] = 0.3  # Passer à la phase suivante
    transition_matrix[-1, -1] = 1.0  # Phase finale : rester dans "Atterrissage"
    #transition_matrix +=1e-10
    transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

    model.transmat_ = transition_matrix
    return model

def detect_phases(data, model, states):
    """
    Associe les phases détectées par le modèle HMM aux données.
    """
    observations = np.column_stack([
        data["ALT[m]"], 
        #data["Vz[m/s]"], 
        #data["TAS[m/s]"], 
        np.gradient(data["ALT[m]"].values),
        np.gradient(np.gradient(data["ALT[m]"].values)),
        #np.gradient(data["Vz[m/s]"].values),  # Dérivée de la vitesse verticale
        #np.gradient(data["TAS[m/s]"].values),  # Dérivée de la vitesse sol
    ])
    hidden_states = model.predict(observations)
    data = data.copy()
    data.loc[:, "Phase"] = [states[state] for state in hidden_states]

    #data["Phase"] = [states[state] for state in hidden_states]
    return data

def plot_phases_on_flight(data, phases_col="Phase", value_col="ALT[m]", title="Phases détectées par vol"):
    """
    Visualise les phases détectées sur un vol donné.
    """
    unique_phases = data[phases_col].unique()
    phase_colors = {phase: color for phase, color in zip(unique_phases, plt.cm.tab10.colors)}

    x_indices = np.arange(len(data))
    plt.figure(figsize=(8, 4))
    for phase in unique_phases:
        phase_mask = data[phases_col] == phase
        plt.plot(
            x_indices[phase_mask],
            data.loc[phase_mask, value_col],
            label=phase,
            color=phase_colors[phase],
            linewidth=2,
        )

    plt.xlabel("Index numérique")
    plt.ylabel(value_col)
    plt.title(title)
    plt.legend(title="Phases", loc="best")
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()


def align_by_phase(reference, target, states):
    """
    Aligne les données de la cible sur celles de la référence phase par phase.
    """
    aligned_data = []
    for phase in states:
        ref_phase = reference[reference["Phase"] == phase]
        tgt_phase = target[target["Phase"] == phase]

        if len(ref_phase) > 1 and len(tgt_phase) > 1:
            ref_indices = np.linspace(0, 1, len(ref_phase))
            tgt_indices = np.linspace(0, 1, len(tgt_phase))

            interpolated_phase = pd.DataFrame({
                col: np.interp(ref_indices, tgt_indices, tgt_phase[col])
                for col in tgt_phase.columns if col != "Phase"
            }, index=ref_phase.index)

            interpolated_phase["Phase"] = phase
            aligned_data.append(interpolated_phase)

    return pd.concat(aligned_data).sort_index()


def plot_alignment(reference, target, aligned_data):
    """
    Trace les données de la référence, de la cible et de la cible alignée.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(target)), target["ALT[m]"], label="Cible avant alignement (ALT[m])", linestyle='--', alpha=0.7)
    plt.plot(range(len(reference)), reference["ALT[m]"], label="Référence (ALT[m])", alpha=0.7)
    plt.plot(range(len(aligned_data)), aligned_data["ALT[m]"], label="Cible après alignement (ALT[m])", alpha=0.9)
    plt.legend()
    plt.xlabel("Indices numériques")
    plt.ylabel("ALT[m]")
    plt.title("Alignement de la cible avec la référence")
    plt.grid()
    plt.tight_layout()
    plt.show()

    import numpy as np

def calculate_sync_error(reference, aligned_target, value_col="ALT[m]", metric="RMSE"):
    """
    Calcule l'erreur de synchronisation entre la référence et la cible alignée.
    
    Args:
        reference (pd.DataFrame): Données de référence.
        aligned_target (pd.DataFrame): Données de la cible alignée.
        value_col (str): La colonne sur laquelle calculer l'erreur.
        metric (str): La métrique utilisée pour l'erreur ("RMSE" ou "MAE").
    
    Returns:
        float: L'erreur de synchronisation calculée.
    """
    if value_col not in reference.columns or value_col not in aligned_target.columns:
        raise ValueError(f"La colonne '{value_col}' doit exister dans les deux ensembles de données.")
    
    ref_values = reference[value_col].values
    tgt_values = aligned_target[value_col].values
    
    if len(ref_values) != len(tgt_values):
        raise ValueError("Les longueurs des données de référence et de la cible alignée ne correspondent pas.")
    
    if metric == "RMSE":
        error = np.sqrt(np.mean((ref_values - tgt_values) ** 2))
    elif metric == "MAE":
        error = np.mean(np.abs(ref_values - tgt_values))
    else:
        raise ValueError(f"Metric '{metric}' non reconnue. Utilisez 'RMSE' ou 'MAE'.")
    
    return error
