import numpy as np

def degrees2rad(degrees):
    return np.pi * degrees / 180.0

def rad2degrees(rad):
    return 180.0 * rad / np.pi

def compute_theta(p_next, p_prev):
    delta_p = p_next - p_prev
    return np.arctan2(delta_p[1], delta_p[0])
