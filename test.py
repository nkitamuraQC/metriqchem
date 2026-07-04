import jax
import jax.numpy as jnp
from metriqchem.curvature import g_ij, g_ij_up
from metriqchem.free_elec_gas import FreeElecGasMetric
from pyscf import gto, dft, scf
import numpy as np

def test_curvature():
    # 動作確認
    x0, y0, z0 = 1.0, 2.0, 3.0
    a = 1.0
    print("g_ij =\n", g_ij(x0, y0, z0, a))
    print("g^ij =\n", g_ij_up(x0, y0, z0, a))
    
    # g_ij @ g^ij ≈ 単位行列 になることを確認
    print("g_ij @ g^ij =\n", g_ij(x0, y0, z0, a) @ g_ij_up(x0, y0, z0, a))
    return

def test_free_elec_gas_metric():
    mol = gto.Mole()
    mol.atom = [
        ['GHOST-H', (0.0, 0.0, 0.0)],
        ['GHOST-H', (1.0, 0.0, 0.0)],
        ['GHOST-H', (0.0, 1.0, 0.0)],
        ['GHOST-H', (0.0, 0.0, 1.0)],
    ]
    mol.basis = 'sto-3g'
    mol.charge = -4  # 電子数 = 2
    mol.spin = 0
    mol.build()

    mf = scf.RHF(mol)
    h1e = mf.get_hcore(mol)
    e_tb = np.linalg.eigvals(h1e)  # ハミルトニアンの固有値を計算して確認

    nx, ny, nz = 5, 5, 5
    a = 0.1
    free_elec_gas = FreeElecGasMetric(mol, a=a)
    e_metric = free_elec_gas.run(nx=nx, ny=ny, nz=nz)
    print("Eigenvalues of the core Hamiltonian:", e_tb)
    print("Free electron gas energy@a={}:".format(a), e_metric)
    return

if __name__ == "__main__":
    test_curvature()
    test_free_elec_gas_metric()
