import numpy as np

def get_possible_displacements(
    max_displacement,
    n_displacements,
    has_zero_displacement = True
):
    if n_displacements <= 1:
        n_displacements = 1
        x_displacements = np.array([0])
        y_displacements = np.array([0])
        return n_displacements, x_displacements, y_displacements
    
    n_gridpoints = int(1 + np.floor(np.sqrt(n_displacements)))
    continue_flag = True
    X_: np.ndarray
    Y_: np.ndarray
    while continue_flag:
        x_ = np.linspace(- max_displacement, + max_displacement, n_gridpoints)
        y_ = np.linspace(- max_displacement, + max_displacement, n_gridpoints)
        X_, Y_ = np.meshgrid(x_, y_)
        R_ = np.sqrt(X_ ** 2 + Y_ ** 2)
        indices = np.where(R_ <= max_displacement)
        if np.size(indices) >= n_displacements:
            continue_flag = False
            break
        n_gridpoints = n_gridpoints + 1
    n_displacements = indices[0].size
    x_displacements = X_[indices]
    y_displacements = Y_[indices]
    # if not n_gridpoints % 2:
    ## if not already present, add zero displacement
    if has_zero_displacement and not np.any(np.logical_and(x_displacements == 0.0, y_displacements == 0.0)):
        x_displacements = np.concatenate((np.array([0]), x_displacements))
        y_displacements = np.concatenate((np.array([0]), y_displacements))
        n_displacements += 1

    return n_displacements, x_displacements, y_displacements
