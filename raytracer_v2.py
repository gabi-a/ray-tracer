from sympy import Symbol, Eq, symbols
from sympy.vector import CoordSys3D, Del
from sympy.functions import sign, sqrt
from sympy.solvers.solveset import solveset_real
import numpy as np
import matplotlib.pyplot as plt

#==================================#
#          Helper functions
#==================================#

coords = CoordSys3D('C')

def unit_vector(v):
    return v / np.linalg.norm(v)

def vector(x, y, z):
    return np.array((x, y, z))

_n_x, _n_y, _n_z = symbols('nx, ny, nz')
_n = _n_x * coords.i + _n_y * coords.j + _n_z * coords.k
_r1_x, _r1_y, _r1_z = symbols('r1x, r1y, r1z')
_r1 = _r1_x * coords.i + _r1_y * coords.j + _r1_z * coords.k
_n1n2 = Symbol('n1n2')

_det = 1 - (_n1n2**2) * (_n.cross(_r1).magnitude()**2)
_det_eval = Symbol('det_eval')
_r_t = _n1n2 * _r1 + (sign(_n.dot(_r1)) * sqrt(_det_eval) - _n1n2 * (_n.dot(_r1))) * _n

#==================================#
#           Base classes
#==================================#

class Ray:

    def __init__(self, u, v, system):
        self.u = u
        self.v = v
        self.system = system

    def propagate(self):
        
        # Find all intersections
        intersections = [interface.intersect(self) for interface in system]
        intersections = list(filter(None, intersections))

        if len(intersections) == 0:
            print("This ray didn't intersect anything! Check system boundaries...")
            print(f"Ray paramterisation: {self.u} + {self.v} t")
            quit()

        # Select first intersection
        # i.e. minimum t for which f(u+vt) = 0 where f(x) = 0 defines the interface
        intersection = min(intersections, key=lambda x:x.t)

        u = self.u + self.v * intersection.t
        # print("Intersection occured with surface described by:")
        # print("\t", intersection)
        # print("at:")
        # print("\t", u)
        
        if intersection.is_boundary:
            # print("Hit boundary")
            return True, u
        
        # Compute refraction
        n1n2 = intersection.refractive_ratio(self.u)
        n_hat = intersection.normal_at(u)

        # print(f"r1 = {self.v}")
        # print(f"n1n2 = {n1n2}")
        # print(f"n_hat = {n_hat}")

        det = _det.subs([
            (_r1_x, self.v[0]),
            (_r1_y, self.v[1]),
            (_r1_z, self.v[2]),
            (_n_x, n_hat.dot(coords.i)),
            (_n_y, n_hat.dot(coords.j)),
            (_n_z, n_hat.dot(coords.k)),
            (_n1n2, n1n2)
        ])

        if det < 0:
            print("Total internal refraction (unimplemented)")
            return False, None

        r_t = _r_t.subs([
            (_det_eval, det),
            (_r1_x, self.v[0]),
            (_r1_y, self.v[1]),
            (_r1_z, self.v[2]),
            (_n_x, n_hat.dot(coords.i)),
            (_n_y, n_hat.dot(coords.j)),
            (_n_z, n_hat.dot(coords.k)),
            (_n1n2, n1n2)
        ])

        v = vector(
            float(r_t.dot(coords.i)),
            float(r_t.dot(coords.j)),
            float(r_t.dot(coords.k)), 
        )

        return False, Ray(u, v, system)


class Interface:

    def __init__(self, _f=None, n1=1, n2=1, is_boundary=False):
        self._f = _f
        self.n1 = n1
        self.n2 = n2
        self.is_boundary = is_boundary

        delop = Del()
        self._grad_f = delop.gradient(self._f, True).normalize()
    
    def normal_at(self, v):
        return self._grad_f.subs([(coords.x, v[0]),
                                  (coords.y, v[1]),
                                  (coords.z, v[2])])

    def __repr__(self):
        return f"f(x,y,z) = "+self._f.__str__()+" = 0"

    def refractive_ratio(self, u):
        inside = float(self._f.subs([(coords.x, u[0]), 
                                     (coords.y, u[1]), 
                                     (coords.z, u[2])])) < 0
        
        return self.n2 / self.n1 if inside else self.n1 / self.n2

    def intersect(self, ray):
        _t = Symbol('t')
        g = self._f.subs([(coords.x, ray.u[0] + ray.v[0]*_t), 
                          (coords.y, ray.u[1] + ray.v[1]*_t), 
                          (coords.z, ray.u[2] + ray.v[2]*_t)])

        sol = solveset_real(Eq(g, 0), _t)
        
        try:
            roots = np.asarray(list(sol), dtype=float)
        except:
            return None

        if len(roots) == 0:
            return None

        roots = roots[np.where(roots > 0)]

        if len(roots) == 0:
            return None

        self.t = np.min(roots)

        return self

#==================================#
#      Boundary interfaces
#==================================#

class YZ_BoundaryInterface(Interface):
    """
    Boundary interface in the y-z plane.
    """

    def __init__(self, x):
        super().__init__(_f = coords.x - x, is_boundary=True)

class XZ_BoundaryInterface(Interface):
    """
    Boundary interface in the x-z plane.
    """

    def __init__(self, y):
        super().__init__(_f = coords.y - y, is_boundary=True)

class XY_BoundaryInterface(Interface):
    """
    Boundary interface in the x-y plane.
    """

    def __init__(self, z):
        super().__init__(_f = coords.z - z, is_boundary=True)

#==================================#
#      Refractive interfaces
#==================================#

class CylindricalInterface(Interface):
    """ 
    Cylindrical interface oriented along the x-axis.
    (z-c_z)^2 + (y-c_y)^2 = r^2
    """

    def __init__(self, c, r, n1, n2):
        super().__init__(_f = (coords.z - c[2])**2 + (coords.y - c[1])**2 - r**2, 
                         n1=n1, n2=n2)

class ParabolicInterface(Interface):
    """
    Parabolic interface oriented along the x-axis.
    z = a*y^2+b
    """

    def __init__(self, a, b, n1, n2):
        super().__init__(_f = a*coords.y**2 + b - coords.z, 
                         n1=n1, n2=n2)


#==================================#
#           Test system
#==================================#

if __name__ == '__main__':
        
    system = [XY_BoundaryInterface(0),
              XY_BoundaryInterface(10),
              XZ_BoundaryInterface(-5),
              XZ_BoundaryInterface(5),
              ParabolicInterface(0.2, 2, 1, 1.2),
              CylindricalInterface(vector(0, 0, 28), 20, 1, 1)]

    paths = []
    q = [([vector(0, y, 0), ], Ray(vector(0, y, 0), unit_vector(vector(0, 0, 1)), system)) for y in np.linspace(-3, 3, 20)]

    while len(q) > 0:
        print(f"There are {len(q)} rays in the queue.", end="\r")
        elem = q.pop()
        hit_boundary, next = elem[1].propagate()
        if hit_boundary:
            elem[0].append(next)
            paths.append(elem[0])
        else:
            elem[0].append(next.u)
            q.append((elem[0], next))
        
    for path in paths:
        p = np.asarray(path).T
        plt.plot(p[2], p[1])
        plt.plot(p[2], p[1], 'kx')

    plt.xlim(0, 10)
    plt.ylim(-5, 5)
    plt.show()

