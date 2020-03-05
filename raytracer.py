import numpy as np
import matplotlib.pyplot as plt

DEBUG = True

"""
Vectors are (z, y)
    y
    ^
    |
 ---|------> z
    |
"""

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
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

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

class System:
    """ Collection of interfaces within a bounding box.

    Args  
    interfaces (list): list of interfaces  
    left (float): left most coordinate of system  
    right (float): right most coordinate of system  
    top (float): top most coordinate of system  
    bottom (float): bottom most coordinate of system  
    """

    def __init__(self, interfaces, left, right, top, bottom):
        self.interfaces = interfaces
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

class Ray:
    """ A ray is a straight line representing the path travelled by light in a medium.

    Args  
    u (vector): displacement vector  
    v (vector): orientation vector  
    """

    def __init__(self, u, v, system):
        self.system = system
        self.u = u # displacement
        self.v = v # direction

    def propagate(self):
        """ Propogates the ray until either it reaches an interface, or leaves the system boundaries.

        Returns  
        reached_boundary (bool): True if ray hits system boundary  
        
        If reached boundary:
            next_ray (ray): ray on the other side of the interface
        Else:
            intersection_point (vector): intersection at boundary
        """

        # Check for any intersections
        intersections = []
        for interface in self.system.interfaces:
            intersection = interface.intersect(self)
            
            if intersection[0] and intersection[1] > 1e-4:
                intersections.append(intersection)
        
        # No intersections => ray hits boundary
        if len(intersections) == 0:
            # Calculate position at which ray would hit boundary
            t = np.ndarray(4)
            t[0] = (self.system.left - self.u[0]) / self.v[0]
            t[1] = (self.system.right - self.u[0]) / self.v[0]
            t[2] = (self.system.top - self.u[1]) / self.v[1]
            t[3] = (self.system.bottom - self.u[1]) / self.v[1]
            print(t)
            # t = t[np.where(t >= 0)]
            return True, self.u + self.v*min(t)

        print(intersections)

        # Get closest intersection
        first_intersection = sorted(intersections, key=lambda x:x[1])[0]

        t = first_intersection[1]
        normal = first_intersection[2]
        refractive_ratio = first_intersection[3]
        entering = first_intersection[4]

        if DEBUG:
            x = self.u[0]+t*self.v[0]
            y = self.u[1]+t*self.v[1]
            plt.arrow(x, y, normal[0], normal[1], head_width=0.2, head_length=0.2, fc='r', ec='r')
            plt.arrow(x, y, -normal[0], -normal[1], head_width=0.2, head_length=0.2, fc='b', ec='b')
            plt.arrow(x, y, self.v[0], self.v[1], head_width=0.2, head_length=0.2, fc='g', ec='g')
            plt.plot(x, y, 'k>' if entering else 'k<')

        print(entering)
        print(normal)

        new_u = self.u + self.v * t

        theta1 = np.pi - angle_between(normal, self.v) if entering else angle_between(normal, self.v) - np.pi
        
        print(angle_between(normal, self.v)*180/np.pi)

        theta2 = refractive_ratio * theta1 if entering else theta1 / refractive_ratio
        
        if normal[1] < 0:
            print("Rotate down")
            new_v = rotate_vector(-normal, theta2)
        else:
            print("Rotate up")
            new_v = rotate_vector(-normal, -theta2)

        return False, Ray(new_u, new_v, self.system)

class CircularInterface:
    """ Circular interface between two mediums.

    Args  
    R (float): radius  
    c (vector): centre point  
    n1 (float):  refractive index towards the negative z direction  
    n2 (float):  refractive index towards the positive z direction
    """

    def __init__(self, R, c, n1, n2):
        self.R = R   # radius
        self.c = c   # centre point
        self.n1 = n1 # n towards the negative z direction
        self.n2 = n2 # n towards the positive z direction

    def intersect(self, ray):
        """ Calculates if the ray intersects this interface.
        
        Args  
        ray: ray object
        
        Returns  
        intersect (bool): True if ray intersects  
        t (float): parameter of ray at intersection  
        normal (vector): vector normal to interface at intersection
        refractive_ratio (float): n1/n2
        entering (bool): True if entering the lens
        """
        p = ray.u - self.c
        delta = np.dot(p, ray.v)**2 + self.R**2 - np.dot(p, p)

        if delta < 1e-4:
            return False, None, None

        t = np.ndarray(2)

        t[0] = -np.dot(p, ray.v) + np.sqrt(delta)
        t[1] = -np.dot(p, ray.v) - np.sqrt(delta)
        
        if DEBUG:
            plt.plot(ray.u[0]+t[0]*ray.v[0], ray.u[1]+t[0]*ray.v[1], 'rx' if t[0] > 0 else 'kx')
            plt.plot(ray.u[0]+t[1]*ray.v[0], ray.u[1]+t[1]*ray.v[1], 'rx' if t[1] > 0 else 'kx')

        t = t[np.where(t > 0)]
        if len(t) == 0:
            return False, None, None

        t = min(t)
        
        if DEBUG:
            plt.plot(ray.u[0]+t*ray.v[0], ray.u[1]+t*ray.v[1], 'gx')

        intersection_point = ray.u + ray.v * t
        normal = -unit_vector(self.c - intersection_point)
        
        if np.abs(angle_between(normal, ray.v)) > np.pi/2:
            entering = True
        else:
            entering = False

        return True, t, normal, self.n1 / self.n2, entering

if __name__ == "__main__":
    
    left=0
    right=10
    bottom=-6
    top=6
    
    i1 = CircularInterface(50, Vector(53, 0), 1, 1.5)
    i2 = CircularInterface(50, Vector(-46, 0), 1.5, 1)

    system = System([i1, i2], left, right, bottom, top)

    q = [(Ray(Vector(0, y), Vector(1, 0), system), [np.array([0, y])]) for y in np.linspace(-3,3, 10)]
    paths = []

    fig, ax = plt.subplots()

    while len(q) > 0:
        e = q.pop()
        stop, x = e[0].propagate()
        if stop:
            e[1].append(x)
            paths.append(e[1])
        else:
            e[1].append(x.u)
            q.append((x, e[1]))

    lens1 = plt.Circle(xy=i1.c, fill=False, radius=i1.R, ls='-', lw=1)
    lens2 = plt.Circle(xy=i2.c, fill=False, radius=i2.R, ls='-', lw=1)
    ax.add_patch(lens1)
    ax.add_patch(lens2)
    
    for path in paths:
        p = np.asarray(path).T
        plt.plot(p[0], p[1])

    plt.xlim(left, right)
    plt.ylim(bottom, top)
    ax.set_aspect('equal')

    plt.show()