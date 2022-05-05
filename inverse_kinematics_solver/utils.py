import numpy as np

def degrees2rad(degrees):
    return np.pi * degrees / 180.0

def rad2degrees(rad):
    return 180.0 * rad / np.pi

def compute_theta(p_next, p_prev):
    delta_p = p_next - p_prev
    return np.arctan2(delta_p[1], delta_p[0])

def compute_angles_from_trajectory(trajectory):
    # forward kinematics
    angles = []
    for idx in range(trajectory.shape[0]):
        theta0 = rad2degrees(compute_theta(trajectory[idx][1], trajectory[idx][0]))
        theta1 = rad2degrees(compute_theta(trajectory[idx][2], trajectory[idx][1]))
        theta2 = rad2degrees(compute_theta(trajectory[idx][3], trajectory[idx][2]))
        angles.append([theta0, theta1, theta2])
    angles = np.array(angles)

    deltas = []
    for idx in range(1, len(angles)):
        deltas.append(angles[idx] - angles[idx - 1])
    return np.array(deltas)