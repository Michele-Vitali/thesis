import numpy as np
from scripts.expert import settings


def process_osc_action(raw_action: np.ndarray, previous_action: np.ndarray = None) -> np.ndarray:
    """
    Centralizza la logica di clipping, binarizzazione del gripper e smoothing (rampa).
    Assicura che la raccolta dati HDF5 e l'inferenza OpenVLA siano matematicamente identiche.
    """
    # 1. Check input format.
    action = np.asarray(raw_action, dtype=float).reshape(-1).copy()
    if action.shape != (7,) or not np.all(np.isfinite(action)):
        raise ValueError(f"Azione non valida ricevuta: {action}")

    # 2. Clip the actions.
    action[0:3] = np.clip(action[0:3], -settings.max_position_action, settings.max_position_action)
    action[3:6] = np.clip(action[3:6], -settings.max_rotation_action, settings.max_rotation_action)

    # 3. Discretize gripper action.
    action[-1] = 1.0 if action[-1] >= 0.5 else 0.0

    '''
    This is robot controller's work, not my pipeline's!

    # 4. Smoothing / Ramp.
    delta = action - previous_action
    
    # Limit the variation on the first 6 DoF.
    delta[:6] = np.clip(delta[:6], -settings.max_action_delta, settings.max_action_delta)
    
    actual_action = previous_action + delta
    
    # Gripper does not get smoothed.
    actual_action[-1] = action[-1] 
    '''

    return action