import numpy as np
import matplotlib.pyplot as plt

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    # v1_u = unit_vector(v1)
    # v2_u = unit_vector(v2)
    angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
    if (angle > np.pi):
        angle -= 2 * np.pi
    elif (angle <= -np.pi):
        angle += 2 * np.pi
    return angle

def rotate_vector(vector, theta):
    """ Rotates a vector by theta radians. """
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        vector[0]*c - vector[1]*s,
        vector[0]*s + vector[1]*c
    ])

def Vector(x, y):
    return np.array([x, y])

def plot_vector(u, v, c):
    """ u: position
        v: direction
        c: color 
    """
    plt.arrow(u[0], u[1], v[0], v[1], head_width=0.2, head_length=0.2, fc=c, ec=c)

fig, ax = plt.subplots()
plt.xlim(-5, 5)
plt.ylim(-5, 5)
ax.set_aspect('equal')

o = Vector(0, 0)
u = Vector(1, 0)
plot_vector(o, u, 'r')
v = rotate_vector(u, 0)
plot_vector(o, v, 'g')

print(angle_between(u, v) * 180/np.pi)

plt.show()