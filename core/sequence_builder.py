import numpy as np

def create_sequences(X, y, time_steps):
    """Create overlapping sequences for LSTM training.

    Parameters
    ----------
    X : np.ndarray or pandas.DataFrame
        Feature matrix of shape (samples, features).
    y : np.ndarray or list
        Target vector of shape (samples,).
    time_steps : int
        Number of time steps per sequence.

    Returns
    -------
    X_seq : np.ndarray
        Array of shape (num_sequences, time_steps, num_features).
    y_seq : np.ndarray
        Corresponding targets for each sequence.
    """
    # Ensure numpy arrays
    if not isinstance(X, np.ndarray):
        X = X.values
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    num_samples, num_features = X.shape
    sequences = []
    targets = []
    # We want the target to be the label of the LAST packet in the sequence
    for i in range(num_samples - time_steps + 1):
        seq = X[i : i + time_steps]
        target = y[i + time_steps - 1] # Label of the last element in the sequence
        sequences.append(seq)
        targets.append(target)
        
    X_seq = np.stack(sequences)
    y_seq = np.array(targets)
    return X_seq, y_seq
