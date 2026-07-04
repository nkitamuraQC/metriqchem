import jax
import jax.numpy as jnp
import numpy as np

def g_ij(x, y, z, a=1.0):
    """Returns the spatial metric g_{ij} (3x3 matrix)"""
    r = jnp.sqrt(x**2 + y**2 + z**2)
    fac = a / (r**2 * (r - a))
    return jnp.array([
        [1 + fac * x**2,     fac * x * y,     fac * x * z],
        [fac * y * x,        1 + fac * y**2,  fac * y * z],
        [fac * z * x,        fac * z * y,     1 + fac * z**2],
    ])

def g_ij_up(x, y, z, a=1.0):
    """Returns the inverse metric g^{ij} (3x3 matrix)"""
    r = jnp.sqrt(x**2 + y**2 + z**2)
    fac = a / r**3
    return jnp.array([
        [1 - fac * x**2,     -fac * x * y,    -fac * x * z],
        [-fac * x * y,       1 - fac * y**2,  -fac * y * z],
        [-fac * x * z,       -fac * y * z,    1 - fac * z**2],
    ])

def g_ij_det(x, y, z, a=1.0):
    """Returns the determinant of the spatial metric g_{ij} (3x3 matrix)"""
    r = np.sqrt(x**2 + y**2 + z**2)
    fac = a / (r**2 * (r - a))
    matrix = np.array([
        [1 + fac * x**2,     fac * x * y,     fac * x * z],
        [fac * y * x,        1 + fac * y**2,  fac * y * z],
        [fac * z * x,        fac * z * y,     1 + fac * z**2],
    ])
    return np.linalg.det(matrix)

def g_ij_det2(x, y, z, a=1.0):
    """Returns the gradient of the determinant of the spatial metric g_{ij} (3x3 matrix)"""
    r = jnp.sqrt(x**2 + y**2 + z**2)
    fac = a / (r**2 * (r - a))
    matrix = jnp.array([
        [1 + fac * x**2,     fac * x * y,     fac * x * z],
        [fac * y * x,        1 + fac * y**2,  fac * y * z],
        [fac * z * x,        fac * z * y,     1 + fac * z**2],
    ])
    return jnp.linalg.det(matrix)


# Jacobian of g_ij_up and g_ij_det2 with respect to x, y, z
dgup_dx = jax.jacobian(g_ij_up, argnums=0)
dgup_dy = jax.jacobian(g_ij_up, argnums=1)
dgup_dz = jax.jacobian(g_ij_up, argnums=2)
dgdet_dx = jax.jacobian(g_ij_det2, argnums=0)
dgdet_dy = jax.jacobian(g_ij_det2, argnums=1)
dgdet_dz = jax.jacobian(g_ij_det2, argnums=2)
