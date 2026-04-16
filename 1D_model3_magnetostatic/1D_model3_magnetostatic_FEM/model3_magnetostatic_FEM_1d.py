# -*- coding: utf-8 -*-
"""
1D axisymmetric FEM for magnetostatic A(r) using FEniCSx
FEniCSx 一维轴对称静磁有限元模型

Governing equation / 控制方程:
    -(1/r) * d/dr [ r * (1/mu(r)) * dA/dr ] = J(r)

Weak form / 弱形式:
    ∫_0^R r * (1/mu(r)) * A'(r) * v'(r) dr = ∫_0^R r * J(r) * v(r) dr

Unknown / 未知量:
    A(r) = A_z(r), unit 单位: Wb/m
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import ufl

from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem


# =========================================================
# 1) Constants / 常数
# =========================================================
mu0 = 4.0 * np.pi * 1.0e-7          # 真空磁导率 Vacuum permeability, H/m
mur_fe = 1000.0                     # 铁磁相对磁导率 Relative permeability of ferro
mu_fe = mur_fe * mu0                # 铁磁绝对磁导率 Absolute permeability of ferro, H/m

# =========================================================
# 2) Geometry parameters / 几何参数
# =========================================================
r_coil_cm = 5.0                     # 线圈半径 / Coil radius, cm
r_fe_in_cm = 10.0                   # 铁磁内半径 / Ferro inner radius, cm
r_fe_out_cm = 10.5                  # 铁磁外半径 / Ferro outer radius, cm
r_out_cm = 30.0                     # 外边界半径 / Outer boundary radius, cm

cm_to_m = 1.0e-2
r_coil = r_coil_cm * cm_to_m        # m
r_fe_in = r_fe_in_cm * cm_to_m      # m
r_fe_out = r_fe_out_cm * cm_to_m    # m
R = r_out_cm * cm_to_m              # m

# =========================================================
# 3) Source current density / 电流密度
# =========================================================
J0_cm2 = 1.0                        # A/cm^2
J0 = J0_cm2 * 1.0e4                 # A/m^2

# =========================================================
# 4) Build 1D interval mesh / 建立 1D 区间网格
#    选 h = 0.5 mm = 0.0005 m，使各分界面都落在节点上
# =========================================================
comm = MPI.COMM_WORLD
rank = comm.rank

if comm.size != 1:
    raise RuntimeError("这份 1D 程序建议串行运行：python model_1d.py")

h = 5.0e-4                          # 网格步长 / Mesh size, m
num_cells = int(round(R / h))       # 0.30 / 0.0005 = 600
assert np.isclose(num_cells * h, R)

msh = mesh.create_interval(comm, num_cells, [0.0, R])

# =========================================================
# 5) Function space / 函数空间
# =========================================================
V = fem.functionspace(msh, ("Lagrange", 1))

# =========================================================
# 6) Axisymmetric coefficients / 轴对称材料与源项
# =========================================================
x = ufl.SpatialCoordinate(msh)
r = x[0]

# mu(r): coil + air = mu0, ferro ring = mu_fe
mu_expr = ufl.conditional(
    ufl.le(r, r_fe_in),
    mu0,
    ufl.conditional(
        ufl.le(r, r_fe_out),
        mu_fe,
        mu0
    )
)

# J(r): only inside coil
J_expr = ufl.conditional(
    ufl.le(r, r_coil),
    J0,
    ScalarType(0.0)
)

# =========================================================
# 7) Boundary condition / 边界条件
#    A(R) = 0
# =========================================================
dofs_R = fem.locate_dofs_geometrical(
    V, lambda x: np.isclose(x[0], R)
)
bc_R = fem.dirichletbc(ScalarType(0.0), dofs_R, V)

# =========================================================
# 8) Variational form / 弱形式
#    ∫ r * (1/mu) * A' * v' dr = ∫ r * J * v dr
# =========================================================
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a_form = (r / mu_expr) * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L_form = r * J_expr * v * ufl.dx

# =========================================================
# 9) Solve / 求解
# =========================================================
problem = LinearProblem(
    a_form,
    L_form,
    bcs=[bc_R],
    petsc_options_prefix="axisym_1d_",
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
    },
)

A = problem.solve()
A.name = "A"
A.x.scatter_forward()

# =========================================================
# 10) Extract nodal result / 提取节点结果
# =========================================================
r_dof = V.tabulate_dof_coordinates()[:, 0]   # 单位 m
A_vals = A.x.array.real

# 排序，防止显示次序混乱
idx = np.argsort(r_dof)
r_dof = r_dof[idx]
A_vals = A_vals[idx]

# =========================================================
# 11) Approximate B_theta = -dA/dr / 近似计算 B_theta
# =========================================================
# 用节点差分近似，仅用于后处理画图
# For post-processing only
B_vals = np.zeros_like(A_vals)

# 内部点中心差分
B_vals[1:-1] = -(A_vals[2:] - A_vals[:-2]) / (r_dof[2:] - r_dof[:-2])

# 两端单边差分
B_vals[0] = -(A_vals[1] - A_vals[0]) / (r_dof[1] - r_dof[0])
B_vals[-1] = -(A_vals[-1] - A_vals[-2]) / (r_dof[-1] - r_dof[-2])

# =========================================================
# 12) Console output / 终端输出
# =========================================================
print("=" * 60)
print("FEniCSx 1D axisymmetric magnetostatic solve finished")
print("FEniCSx 一维轴对称静磁求解完成")
print(f"Max(A) = {np.max(A_vals):.6e} Wb/m")
print(f"Min(A) = {np.min(A_vals):.6e} Wb/m")
print(f"Max(|B_theta|) = {np.max(np.abs(B_vals)):.6e} T")
print(f"Min(|B_theta|) = {np.min(np.abs(B_vals)):.6e} T")
print("=" * 60)

# =========================================================
# 13) Plot A(r) / 绘制 A(r)
# =========================================================
save_dir = os.path.dirname(os.path.abspath(__file__))

fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(r_dof * 100.0, A_vals, linewidth=2.0, label=r"$A(r)$")

ax1.axvline(r_coil_cm, linestyle="--", linewidth=1.2, label="coil radius")
ax1.axvline(r_fe_in_cm, linestyle="--", linewidth=1.2, label="ferro inner radius")
ax1.axvline(r_fe_out_cm, linestyle="--", linewidth=1.2, label="ferro outer radius")

ax1.set_xlabel("r / cm")
ax1.set_ylabel("A(r) / Wb/m")
ax1.set_title("1D Axisymmetric FEM Solution of Magnetic Vector Potential")
ax1.grid(True, alpha=0.3)
ax1.legend()
fig1.tight_layout()

save_path_A = os.path.join(save_dir, "A_1D_axisymmetric.png")
fig1.savefig(save_path_A, dpi=300, bbox_inches="tight")
plt.close(fig1)

# =========================================================
# 14) Plot B_theta(r) / 绘制 B_theta(r)
# =========================================================
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(r_dof * 100.0, B_vals, linewidth=2.0, label=r"$B_\theta(r)\approx -dA/dr$")

ax2.axvline(r_coil_cm, linestyle="--", linewidth=1.2, label="coil radius")
ax2.axvline(r_fe_in_cm, linestyle="--", linewidth=1.2, label="ferro inner radius")
ax2.axvline(r_fe_out_cm, linestyle="--", linewidth=1.2, label="ferro outer radius")

ax2.set_xlabel("r / cm")
ax2.set_ylabel(r"$B_\theta(r)$ / T")
ax2.set_title("1D Axisymmetric FEM Approximation of Magnetic Flux Density")
ax2.grid(True, alpha=0.3)
ax2.legend()
fig2.tight_layout()

save_path_B = os.path.join(save_dir, "Btheta_1D_axisymmetric.png")
fig2.savefig(save_path_B, dpi=300, bbox_inches="tight")
plt.close(fig2)

print(f"图像已保存为: {save_path_A}")
print(f"图像已保存为: {save_path_B}")