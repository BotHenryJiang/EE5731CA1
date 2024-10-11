import numpy as np

def construct_W(points1, points2):
    """
    Construct linear system W for the epipolar constraint
    """
    W = []
    for (x, y, _), (x_prime, y_prime, _) in zip(points1, points2):
        W.append([
            x_prime*x, x_prime*y, x_prime,
            y_prime*x, y_prime*y, y_prime,
            x, y, 1
        ])
    return np.array(W)

def compute_fundamental_matrix(points1, points2, T1, T2):
    # Ensure points are in homogeneous coordinates
    assert points1.shape[1] == 3 and points2.shape[1] == 3, "Points must be in homogeneous coordinates"

    # Construct matrix W for the equation Wf = 0
    W = construct_W(points1, points2)

    # Perform SVD on W
    _, _, Vt = np.linalg.svd(W)
    F = Vt[-1].reshape(3, 3)

    # Enforce rank 2 constraint on F
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ Vt

    # Denormalize the fundamental matrix
    F = T2.T @ F @ T1

    # Normalize F to have unit norm
    F = F / np.max(np.abs(F))

    return F

# Compute the fundamental matrix using normalized keypoints
F = compute_fundamental_matrix(normalized_keypoints_1, normalized_keypoints_2, T1, T2)

# Print the fundamental matrix
print("Computed Fundamental Matrix F:")
print(F)