import numpy as np
import matplotlib.pyplot as plt
import json
import os


def dist(loc1, loc2):
    return np.sqrt((loc2[0] - loc1[0]) ** 2 + (loc2[1] - loc1[1]) ** 2)

def mvee(points, tol = 0.001, jitter=0.3):
    """
    Find the minimum volume ellipse.
    Return A, c where the equation for the ellipse given in "center form" is
    (x-c).T * A * (x-c) = 1
    """
    points = np.asmatrix(points, dtype=float)
    # Jitter
    points += np.random.uniform(-jitter, jitter, size=points.shape)
    N, d = points.shape
    Q = np.column_stack((points, np.ones(N))).T
    err = tol+1.0
    u = np.ones(N)/N
    while err > tol:
        # assert u.sum() == 1 # invariant
        X = Q * np.diag(u) * Q.T
#         invertible = la.cond(X) < 1/sys.float_info.epsilon
        M = np.diag(Q.T * la.inv(X) * Q)
        jdx = np.argmax(M)
        step_size = (M[jdx]-d-1.0)/((d+1)*(M[jdx]-1.0))
        new_u = (1-step_size)*u
        new_u[jdx] += step_size
        err = la.norm(new_u-u)
        u = new_u
    c = u*points
    A = la.inv(points.T*np.diag(u)*points - c.T*c)/d    
    return np.asarray(A), np.squeeze(np.asarray(c))

def min_bounding_ellipse(points, jitter=0, pad=1.0, min_radius=1.0):
    A, center = mvee(points, tol=0.0001, jitter=jitter)
    U, D, V = la.svd(A)

    # x, y radii.
    rx, ry = 1./np.sqrt(D)
    # Major and minor semi-axis of the ellipse.
    dx, dy = 2 * rx, 2 * ry
    a, b = max(dx, dy), min(dx, dy)

    e = np.sqrt(a ** 2 - b ** 2) / a

    arcsin = -1. * np.rad2deg(np.arcsin(V[0][0]))
    arccos = np.rad2deg(np.arccos(V[0][1]))

    angle = arccos if arcsin > 0. else -1. * arccos

    # Floor to avoid collapsed ellipse
    a = max(a + pad, min_radius)
    b = max(b + pad, min_radius)            
    
    return center, a, b, angle