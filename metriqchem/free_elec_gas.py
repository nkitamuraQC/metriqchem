from pyscf import gto, dft
import numpy as np
import jax

from metriqchem.curvature import (
    g_ij, g_ij_up, g_ij_det,
    dgup_dx, dgup_dy, dgup_dz,
    dgdet_dx, dgdet_dy, dgdet_dz
)

class FreeElecGasMetric:
    def __init__(self, refmol: gto.Mole, a = 1.0):
        """
        Parameters
        ----------
        refmol : gto.Mole
            reference molecule object from PySCF.
        a : float, optional
            value of the constant term in the metric. Default is 1.0.
        """
        self.refmol = refmol
        self.coords = None
        self.weights = None
        self.ao_value = None
        self.ao_grad = None
        self.ao_hess = None
        self.constant = a  # 定数項の初期値を設定
        self.g_ij_values = None  # リーマン計量の初期値を設定
        self.g_ij_det_values = None  # リーマン計量の行列式の初期値を設定
        self.g_ij_up_values = None  # 逆リーマン計量の初期値を設定
        self.g_ij_grad_values = None  # リーマン計量の勾配の初期値を設定
        self.divergence_check = False  # 発散チェックの初期値を設定

    def build_grid(self, level=0):
        """
        Generates a grid and sets the coordinates and weights.

        Parameters
        ----------
        level : int, optional
            Grid resolution level (0-9, higher means denser). Default is 0.
        """
        grids = dft.gen_grid.Grids(self.refmol)
        grids.level = level
        grids.build()
        self.coords = grids.coords   # (Ngrid, 3) grid point coordinates
        self.weights = grids.weights # (Ngrid,)  Weights for each grid point
        print(f"Grid built with {self.coords.shape[0]} points at level {level}.")
        return
    
    def make_manual_grids(self, nx=4, ny=4, nz=4, x_range=(0, 1), y_range=(0, 1), z_range=(0, 1)):
        """
        Generates a manual grid based on specified ranges and number of points in each direction.

        Parameters
        ----------
        nx, ny, nz : int
            Number of grid points in x, y, z directions.
        x_range, y_range, z_range : tuple
            Range for each direction (min, max).
        """
        x = np.linspace(x_range[0], x_range[1], nx)
        y = np.linspace(y_range[0], y_range[1], ny)
        z = np.linspace(z_range[0], z_range[1], nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        self.coords = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        self.weights = np.ones(self.coords.shape[0])  
        print(f"Manual grid built with {self.coords.shape[0]} points.")
        return

        

    def compute_ao_and_grad(self):
        """
        Calculates the AO values, gradients, and Hessians at the grid points.

        Parameters
        -------
        ao_value : ndarray
            AO values (Ngrid, Nao)。
        ao_grad : ndarray
            AO gradients (3, Ngrid, Nao)。
        ao_hess : ndarray
            AO Hessians (6, Ngrid, Nao)。
        """
        if self.coords is None:
            raise ValueError("グリッドが生成されていません。まずbuild_grid()を呼び出してください。")
        
        ao = dft.numint.eval_ao(self.refmol, self.coords, deriv=2)  # (10, Ngrid, Nao)
        self.ao_value = ao[0]        # (Ngrid, Nao)
        self.ao_grad  = ao[1:4]      # (3, Ngrid, Nao)
        ao_hess = ao[4:10]           # (6, Ngrid, Nao)
        self.ao_hess = np.zeros((3, 3, self.ao_value.shape[0], self.ao_value.shape[1]))
        # AOヘッセ行列を3x3の形に変換(要チェック)
        # k = 0 → ∂xx
        # k = 1 → ∂yy
        # k = 2 → ∂zz
        # k = 3 → ∂xy
        # k = 4 → ∂xz
        # k = 5 → ∂yz
        self.ao_hess[0, 0] = ao_hess[0]
        self.ao_hess[0, 1] = ao_hess[3]
        self.ao_hess[0, 2] = ao_hess[4]
        self.ao_hess[1, 0] = ao_hess[3]
        self.ao_hess[1, 1] = ao_hess[1]
        self.ao_hess[1, 2] = ao_hess[5]
        self.ao_hess[2, 0] = ao_hess[4]
        self.ao_hess[2, 1] = ao_hess[5]
        self.ao_hess[2, 2] = ao_hess[2]
        if self.divergence_check:
            print("nan check@ao_hess:", np.isnan(self.ao_hess).any())
            print("nan check@ao_grad:", np.isnan(self.ao_grad).any())
            print("nan check@ao_value:", np.isnan(self.ao_value).any())
        return
    
    def compute_riemannian_metric(self):
        """
        Calculates the Riemannian metric at each grid point.

        Returns
        -------
        g_ij_values : ndarray
            The Riemannian metric at each grid point (Ngrid, 3, 3)。
        """
        if self.ao_value is None or self.ao_grad is None:
            raise ValueError("AO値と勾配が計算されていません。まずcompute_ao_and_grad()を呼び出してください。")
        
        Ngrid = self.ao_value.shape[0]
        self.g_ij_values = np.zeros((Ngrid, 3, 3))
        self.g_ij_up_values = np.zeros((Ngrid, 3, 3))
        self.g_ij_det_values = np.zeros(Ngrid)
        self.g_ij_grad_values = np.zeros((Ngrid, 3, 3, 3)) ## (ncoord, dx_i, 3, 3) の形状に変更
        self.g_ij_det_grad_values = np.zeros((Ngrid, 3))
        
        for i in range(Ngrid):
            x, y, z = self.coords[i]
            self.g_ij_values[i] = np.asarray(g_ij(x, y, z, a=self.constant))
            self.g_ij_up_values[i] = np.asarray(g_ij_up(x, y, z, a=self.constant))
            self.g_ij_det_values[i] = g_ij_det(x, y, z, a=self.constant)
            self.g_ij_grad_values[i] = np.asarray([dgup_dx(x, y, z, a=self.constant),
                                                   dgup_dy(x, y, z, a=self.constant),
                                                   dgup_dz(x, y, z, a=self.constant)])
            self.g_ij_det_grad_values[i] = np.asarray([dgdet_dx(x, y, z, a=self.constant),
                                                       dgdet_dy(x, y, z, a=self.constant),
                                                       dgdet_dz(x, y, z, a=self.constant)])
        self.g_ij_values = np.nan_to_num(self.g_ij_values, nan=0.0, posinf=0.0, neginf=0.0)
        self.g_ij_up_values = np.nan_to_num(self.g_ij_up_values, nan=0.0, posinf=0.0, neginf=0.0)
        self.g_ij_det_values = np.nan_to_num(self.g_ij_det_values, nan=0.0, posinf=0.0, neginf=0.0)
        self.g_ij_det_grad_values = np.nan_to_num(self.g_ij_det_grad_values, nan=0.0, posinf=0.0, neginf=0.0)
        self.g_ij_grad_values = np.nan_to_num(self.g_ij_grad_values, nan=0.0, posinf=0.0, neginf=0.0)

        if self.divergence_check:
            print("nan check@g_ij_values:", np.isnan(self.g_ij_values).any())
            print("nan check@g_ij_up_values:", np.isnan(self.g_ij_up_values).any())
            print("nan check@g_ij_det_values:", np.isnan(self.g_ij_det_values).any())
            print("nan check@g_ij_grad_values:", np.isnan(self.g_ij_grad_values).any())
            print("nan check@g_ij_det_grad_values:", np.isnan(self.g_ij_det_grad_values).any())
        return 
    
    def get_ham(self):
        """
        Calculates the Hamiltonian matrix using the Riemannian metric and AO values.

        Returns
        -------
        ham : ndarray
            The Hamiltonian matrix (Nao, Nao)。
        """
        if self.g_ij_values is None or self.g_ij_up_values is None:
            raise ValueError("リーマン計量が計算されていません。まずcompute_riemannian_metric()を呼び出してください。")
        
        Ngrid = self.ao_value.shape[0]
        nao = self.ao_value.shape[1]
        term1 = np.zeros((nao, nao))

        det_g_inv = 1.0/self.g_ij_det_values  # (Ngrid,)
        det_g_inv = np.nan_to_num(det_g_inv, copy=False, nan=0.0, posinf=0.0, neginf=0.0)  # 発散チェック
        
        ## term1
        for dx_i in range(3):
            g_ij_grad_val = self.g_ij_det_grad_values[:, dx_i]  
            for dx_j in range(3):
                g_ij_up_val = self.g_ij_up_values[:, dx_i, dx_j] 
                ao_grad_j = self.ao_grad[dx_j]  
                term1 += np.einsum('gi,gj,g,g,g,g->ij', self.ao_value, ao_grad_j, g_ij_up_val, self.weights, det_g_inv, g_ij_grad_val)  # (Nao, Nao)
                    
        ## term2
        term2 = np.zeros((nao, nao))
        for dx_i in range(3):
            g_ij_grad_val = self.g_ij_grad_values[:, dx_i]  # (Ngrid, 3)
            for dx_j in range(3):
                ao_grad_j = self.ao_grad[dx_j]  # (Nao,)
                term2 += np.einsum('gi,gj,g,g->ij', self.ao_value, ao_grad_j, self.weights, g_ij_grad_val[:, dx_i, dx_j])

        ## term3
        term3 = np.zeros((nao, nao))
        for dx_i in range(3):
            for dx_j in range(3):
                ao_hess_ij = self.ao_hess[dx_i, dx_j]  
                g_ij_up_val = self.g_ij_up_values[:, dx_i, dx_j]
                term3 += np.einsum('gi,gj,g,g->ij', self.ao_value, ao_hess_ij, self.weights, g_ij_up_val)


        ham = (term1 * 0.5 + term2 + term3) * -1
        return ham
    

    def run(self, level=0, use_dft_grid=False, nx=4, ny=4, nz=4, x_range=(0, 1), y_range=(0, 1), z_range=(0, 1)):
        """
        Runs the full pipeline: grid generation, AO computation, Riemannian metric computation, and Hamiltonian eigenvalue calculation.

        Parameters
        ----------
        level : int, optional
            Grid resolution level (0-9, higher means denser). Default is 0.
        use_dft_grid : bool, optional
            If True, use PySCF's DFT grid. If False, use manual grid. Default is False.
        nx, ny, nz : int, optional
            Number of grid points in x, y, z directions for manual grid. Default is 4.
        x_range, y_range, z_range : tuple, optional
            Ranges for x, y, z directions for manual grid. Default is (0, 1).
        """
        if use_dft_grid:
            self.build_grid(level=level)
        else:
            self.make_manual_grids(nx=nx, ny=ny, nz=nz, x_range=x_range, y_range=y_range, z_range=z_range)
        # self.build_grid(level=level)
        self.compute_ao_and_grad()
        self.compute_riemannian_metric()
        ham = self.get_ham()
        return np.linalg.eigvals(ham)  # 固有値を計算して返す

        
    

    
