"""
Coordination transformation
"""

import numpy as np
def cartesion_to_spherical(x, y, z):
    r3 = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)
    theta = np.arccos(z/r3)
    return r3, theta, phi


def cylindrical_to_cartesion(r, phi, z):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return (x, y, z)


def theta_to_eta(theta):
    return -np.log(np.tan(0.5*theta))
