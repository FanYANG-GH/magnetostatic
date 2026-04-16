# -*- coding: utf-8 -*-
"""
Model 3 : 2D multi-domain magnetostatic PINN (PyTorch)
模型3：二维多区域静磁问题 PINN（PyTorch）

Problem / 问题:
    Unknown scalar magnetic vector potential A(x, y) = A_z(x, y)
    未知量为标量磁矢势 A(x, y) = A_z(x, y)

Governing PDE / 控制方程:
    -∇·((1/μ) ∇A) = Jz

Region-wise equations / 分区域方程:
    Coil region / 线圈区:       -ΔA = μ0 J0
    Air regions / 空气区:       -∇·((1/μ0)∇A) = 0
    Ferro region / 铁磁区:      -∇·((1/μfe)∇A) = 0

Boundary condition / 边界条件:
    A = 0 on outer circle r = R
    外边界圆 r = R 上 A = 0

Interface conditions / 界面条件:
    A^- = A^+
    q^-·n = q^+·n,  where q = (1/μ)∇A

Mixed first-order form / 混合一阶形式:
    qx - (1/μ) ∂A/∂x = 0
    qy - (1/μ) ∂A/∂y = 0
    ∂qx/∂x + ∂qy/∂y + Jz = 0

Dimensionless variables / 无量纲变量:
    X = x / R,  Y = y / R
    A = A0 * a,      A0 = μ0 * J0 * R^2
    q = q0 * qh,     q0 = J0 * R

Then / 则有:
    qhx - (1/μr) ∂a/∂X = 0
    qhy - (1/μr) ∂a/∂Y = 0
    ∂qhx/∂X + ∂qhy/∂Y + Jhat = 0

This script follows the design idea of your successful 1D PINN example:
本脚本沿用了你那个成功的一维 PINN 示例的核心思路：
    1) multi-domain subnetworks / 多子区域子网络
    2) mixed first-order residuals / 混合一阶残差
    3) interface continuity constraints / 界面连续约束
    4) Adam + LBFGS optimization / Adam + LBFGS 优化
    5) optional FEM point supervision / 可选 FEM 点监督

Notes / 说明:
    - This is a true 2D Cartesian PINN on the full circular cross-section.
      这是在完整圆形截面上的真正二维笛卡尔 PINN，不是把 1D 径向公式直接拿来训练。
    - Because the exact solution is radially symmetric, a radial exact solution is provided
      for verification and plotting.
      由于该算例解析上具有径向对称性，脚本也提供了径向解析解用于校验与绘图。
"""

import os
import math
import time
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# =========================================================
# 0) Reproducibility / 随机种子
# =========================================================
def set_seed(seed: int = 42):
    """Set random seed / 设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# 1) Physical parameters / 物理参数
# =========================================================
@dataclass
class PhysParams:
    # -----------------------------
    # Constants / 常数
    # -----------------------------
    mu0: float = 4.0 * math.pi * 1.0e-7      # 真空磁导率 / Vacuum permeability, H/m
    mur_fe: float = 1000.0                   # 铁磁相对磁导率 / Relative permeability

    # -----------------------------
    # Geometry / 几何参数，单位 cm
    # -----------------------------
    r_coil_cm: float = 5.0                   # 线圈半径 / Coil radius, cm
    r_fe_in_cm: float = 10.0                 # 铁磁环内半径 / Ferro inner radius, cm
    r_fe_out_cm: float = 10.5                # 铁磁环外半径 / Ferro outer radius, cm
    r_out_cm: float = 30.0                   # 外边界半径 / Outer boundary radius, cm

    # -----------------------------
    # Source / 源项，单位 A/cm^2
    # -----------------------------
    J0_cm2: float = 1.0                      # 电流密度 / Current density, A/cm^2

    def __post_init__(self):
        # -----------------------------
        # Unit conversion / 单位换算
        # -----------------------------
        cm_to_m = 1.0e-2

        self.mu_fe = self.mur_fe * self.mu0                  # 铁磁绝对磁导率 / Absolute permeability, H/m
        self.r_coil = self.r_coil_cm * cm_to_m               # 线圈半径 / m
        self.r_fe_in = self.r_fe_in_cm * cm_to_m             # 铁磁内半径 / m
        self.r_fe_out = self.r_fe_out_cm * cm_to_m           # 铁磁外半径 / m
        self.R = self.r_out_cm * cm_to_m                     # 外边界半径 / m
        self.J0 = self.J0_cm2 * 1.0e4                        # 电流密度 / A/m^2

        # -----------------------------
        # Dimensionless radii / 无量纲半径
        # -----------------------------
        self.rho_coil = self.r_coil / self.R
        self.rho_fe_in = self.r_fe_in / self.R
        self.rho_fe_out = self.r_fe_out / self.R

        # -----------------------------
        # Characteristic scales / 特征尺度
        # -----------------------------
        self.A0 = self.mu0 * self.J0 * self.R ** 2           # A 的特征尺度 / Wb/m
        self.q0 = self.J0 * self.R                           # q=(1/mu)gradA 的特征尺度 / A/m
        self.B0 = self.mu0 * self.J0 * self.R                # B 的特征尺度 / T

        # -----------------------------
        # Consistency check / 一致性检查
        # -----------------------------
        if not (0.0 < self.rho_coil < self.rho_fe_in < self.rho_fe_out < 1.0):
            raise ValueError("Geometry radii must satisfy 0 < r_coil < r_fe_in < r_fe_out < R")


# =========================================================
# 2) Training parameters / 训练参数
# =========================================================
@dataclass
class TrainParams:
    # -----------------------------
    # Network / 网络参数
    # -----------------------------
    hidden_layers: int = 4
    hidden_units: int = 64

    # -----------------------------
    # Collocation points / 配点数
    # -----------------------------
    N1: int = 2200            # Coil / 线圈区
    N2: int = 2200            # Inner air / 内空气区
    N3: int = 4600            # Ferro thin ring / 铁磁薄层
    N4: int = 4200            # Outer air / 外空气区

    # Interface points / 界面点数
    Nif12: int = 1024
    Nif23: int = 1024 #1024
    Nif34: int = 1024 #1024

    # Evaluation grid / 后处理网格
    n_plot: int = 361

    # -----------------------------
    # Optimization / 优化参数
    # -----------------------------
    adam_epochs: int = 1000
    lbfgs_steps: int = 0
    lr_adam: float = 1.0e-3
    print_every: int = 200
    resample_every: int = 1000

    # -----------------------------
    # Loss weights / 损失权重
    # -----------------------------
    w_pde: float = 1.0
    w_if: float = 80.0
    w_sup_A: float = 100.0
    w_sup_B: float = 0.001

    # -----------------------------
    # FEM supervision /  FEM 监督
    # 期望列：x_m, y_m, A_Wb_per_m
    # 可选列：Bx_T, By_T, weight
    # -----------------------------
    use_fem_supervision: bool = True # False
    fem_csv_relpath: str = "fem_2d_supervision_points.csv"

    # -----------------------------
    # Checkpoint resume / 断点续训
    # -----------------------------
    resume_from_checkpoint: bool = True # False
    checkpoint_relpath: str = "model3_pinn_2d_checkpoint.pth"
    strict_resume: bool = True




# =========================================================
# 3) Neural network blocks / 神经网络模块
# =========================================================
class SubNet2D(nn.Module):
    """
    One subnetwork for one subdomain.
    一个子区域对应一个子网络。

    Input / 输入:
        (X, Y) with X,Y in [-1, 1]

    Output / 输出:
        [a_raw, qhx_raw, qhy_raw]
    """
    def __init__(self, hidden_layers: int = 4, hidden_units: int = 64):
        super().__init__()
        layers = []
        in_dim = 2
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_units))
            layers.append(nn.Tanh())
            in_dim = hidden_units
        layers.append(nn.Linear(in_dim, 3))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.net(xy)


class MultiDomainPINN2D(nn.Module):
    """
    4 subnetworks for 4 subdomains / 四个子区域对应四个子网络

    Region 1 / 第1区: coil
    Region 2 / 第2区: inner air gap
    Region 3 / 第3区: ferro ring
    Region 4 / 第4区: outer air
    """
    def __init__(self, phys: PhysParams, train: TrainParams):
        super().__init__()
        self.phys = phys
        self.net1 = SubNet2D(train.hidden_layers, train.hidden_units)
        self.net2 = SubNet2D(train.hidden_layers, train.hidden_units)
        self.net3 = SubNet2D(train.hidden_layers, train.hidden_units)
        self.net4 = SubNet2D(train.hidden_layers, train.hidden_units)

    def fields(self, subnet: SubNet2D, xy: torch.Tensor, region_id: int):
        """
        Return dimensionless fields a(X,Y), qhx(X,Y), qhy(X,Y)
        返回无量纲场 a(X,Y), qhx(X,Y), qhy(X,Y)

        Hard constraint / 硬约束:
            Region 4 (outer air): a = 0 on outer circle rho=1
            第4区(外空气)在外边界 rho=1 处严格满足 a=0
        """
        raw = subnet(xy)
        a_raw = raw[:, 0:1]
        qx_raw = raw[:, 1:2]
        qy_raw = raw[:, 2:3]

        if region_id == 4:
            x = xy[:, 0:1]
            y = xy[:, 1:2]
            rho2 = x * x + y * y
            # Hard BC / 外边界硬约束
            a = (1.0 - rho2) * a_raw
        else:
            a = a_raw

        qx = qx_raw
        qy = qy_raw
        return a, qx, qy


# =========================================================
# 4) Differential utilities / 自动微分工具
# =========================================================
def grad_scalar(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Gradient of scalar field / 标量场梯度"""
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]


# =========================================================
# 5) Geometry helpers / 几何辅助函数
# =========================================================
def radius_from_xy(xy: torch.Tensor) -> torch.Tensor:
    """Return rho = sqrt(X^2+Y^2) / 返回无量纲半径 rho"""
    return torch.sqrt(torch.clamp(xy[:, 0:1] ** 2 + xy[:, 1:2] ** 2, min=1.0e-30))


def unit_normal_on_circle(xy: torch.Tensor) -> torch.Tensor:
    """Unit outward normal on a circle / 圆界面外法向"""
    rho = radius_from_xy(xy)
    return xy / rho


def sample_disk(n: int, rho_max: float, device, dtype) -> torch.Tensor:
    """
    Area-uniform random sampling in a disk.
    在圆盘内做面积均匀随机采样。
    """
    u = torch.rand(n, 1, device=device, dtype=dtype)
    theta = 2.0 * math.pi * torch.rand(n, 1, device=device, dtype=dtype)
    rho = rho_max * torch.sqrt(u)
    x = rho * torch.cos(theta)
    y = rho * torch.sin(theta)
    xy = torch.cat([x, y], dim=1)
    xy.requires_grad_(True)
    return xy


def sample_annulus(n: int, rho_in: float, rho_out: float, device, dtype) -> torch.Tensor:
    """
    Area-uniform random sampling in an annulus.
    在圆环内做面积均匀随机采样。
    """
    u = torch.rand(n, 1, device=device, dtype=dtype)
    theta = 2.0 * math.pi * torch.rand(n, 1, device=device, dtype=dtype)
    rho = torch.sqrt((rho_out ** 2 - rho_in ** 2) * u + rho_in ** 2)
    x = rho * torch.cos(theta)
    y = rho * torch.sin(theta)
    xy = torch.cat([x, y], dim=1)
    xy.requires_grad_(True)
    return xy


def sample_annulus_edge_dense(
    n: int,
    rho_in: float,
    rho_out: float,
    device,
    dtype,
    power: float = 0.35,
) -> torch.Tensor:
    """
    Edge-dense sampling for a thin annulus.
    对薄圆环在两侧界面附近加密采样。
    """
    n1 = n // 2
    n2 = n - n1

    theta1 = 2.0 * math.pi * torch.rand(n1, 1, device=device, dtype=dtype)
    theta2 = 2.0 * math.pi * torch.rand(n2, 1, device=device, dtype=dtype)

    u1 = torch.rand(n1, 1, device=device, dtype=dtype)
    u2 = torch.rand(n2, 1, device=device, dtype=dtype)

    # Dense near inner interface / 靠近内界面加密
    rho1 = rho_in + (rho_out - rho_in) * (u1 ** power)
    # Dense near outer interface / 靠近外界面加密
    rho2 = rho_out - (rho_out - rho_in) * (u2 ** power)

    x1 = rho1 * torch.cos(theta1)
    y1 = rho1 * torch.sin(theta1)
    x2 = rho2 * torch.cos(theta2)
    y2 = rho2 * torch.sin(theta2)

    xy = torch.cat([torch.cat([x1, y1], dim=1), torch.cat([x2, y2], dim=1)], dim=0)
    idx = torch.randperm(n, device=device)
    xy = xy[idx]
    xy.requires_grad_(True)
    return xy


def sample_circle(n: int, rho: float, device, dtype) -> torch.Tensor:
    """Uniform sampling on a circle / 在圆界面上均匀随机采样"""
    theta = 2.0 * math.pi * torch.rand(n, 1, device=device, dtype=dtype)
    x = rho * torch.cos(theta)
    y = rho * torch.sin(theta)
    xy = torch.cat([x, y], dim=1)
    xy.requires_grad_(True)
    return xy


# =========================================================
# 6) Region material/source maps / 区域材料与源项映射
# =========================================================
def mu_r_of_region(region_id: int, phys: PhysParams) -> float:
    """Relative permeability of each region / 各区域相对磁导率"""
    if region_id == 3:
        return phys.mur_fe
    return 1.0


def Jhat_of_region(region_id: int) -> float:
    """Dimensionless source term / 无量纲源项"""
    if region_id == 1:
        return 1.0
    return 0.0


def build_collocation(phys: PhysParams, train: TrainParams, device, dtype) -> Dict[str, torch.Tensor]:
    """Build collocation and interface points / 构造配点和界面点"""
    colloc = {
        # Domain collocation points / 域内配点
        "xy1": sample_disk(train.N1, phys.rho_coil, device, dtype),
        "xy2": sample_annulus(train.N2, phys.rho_coil, phys.rho_fe_in, device, dtype),
        "xy3": sample_annulus_edge_dense(train.N3, phys.rho_fe_in, phys.rho_fe_out, device, dtype, power=0.35),
        "xy4": sample_annulus(train.N4, phys.rho_fe_out, 1.0, device, dtype),
        # Interface points / 界面点
        "if12": sample_circle(train.Nif12, phys.rho_coil, device, dtype),
        "if23": sample_circle(train.Nif23, phys.rho_fe_in, device, dtype),
        "if34": sample_circle(train.Nif34, phys.rho_fe_out, device, dtype),
    }
    return colloc


# =========================================================
# 7) FEM supervision / FEM 监督
# =========================================================
def load_fem_supervision(csv_path: str, phys: PhysParams, device, dtype) -> Dict[str, torch.Tensor]:
    """
    Load FEM supervision points from CSV.
    从 CSV 读取 FEM 监督点。

    Required columns / 必需列:
        x_m, y_m, A_Wb_per_m

    Optional columns / 可选列:
        Bx_T, By_T, weight
    """
    data = np.genfromtxt(csv_path, delimiter=",", names=True, encoding="utf-8")
    data = np.atleast_1d(data)
    names = data.dtype.names

    required = ["x_m", "y_m", "A_Wb_per_m"]
    for key in required:
        if key not in names:
            raise ValueError(f"CSV missing required column: {key}")

    x_m = np.asarray(data["x_m"], dtype=np.float64).reshape(-1)
    y_m = np.asarray(data["y_m"], dtype=np.float64).reshape(-1)
    A = np.asarray(data["A_Wb_per_m"], dtype=np.float64).reshape(-1)

    if "Bx_T" in names:
        Bx = np.asarray(data["Bx_T"], dtype=np.float64).reshape(-1)
        has_Bx = True
    else:
        Bx = np.zeros_like(A)
        has_Bx = False

    if "By_T" in names:
        By = np.asarray(data["By_T"], dtype=np.float64).reshape(-1)
        has_By = True
    else:
        By = np.zeros_like(A)
        has_By = False

    if "weight" in names:
        weight = np.asarray(data["weight"], dtype=np.float64).reshape(-1)
    else:
        weight = np.ones_like(A)

    rho = np.sqrt(x_m ** 2 + y_m ** 2) / phys.R
    keep = rho <= 1.0 + 1.0e-12

    x_m = x_m[keep]
    y_m = y_m[keep]
    A = A[keep]
    Bx = Bx[keep]
    By = By[keep]
    weight = weight[keep]

    weight = weight / np.mean(weight)

    X = x_m / phys.R
    Y = y_m / phys.R
    a = A / phys.A0
    bx = Bx / phys.B0
    by = By / phys.B0

    sup = {
        "xy": torch.tensor(np.column_stack([X, Y]), device=device, dtype=dtype),
        "a": torch.tensor(a[:, None], device=device, dtype=dtype),
        "bx": torch.tensor(bx[:, None], device=device, dtype=dtype),
        "by": torch.tensor(by[:, None], device=device, dtype=dtype),
        "w": torch.tensor(weight[:, None], device=device, dtype=dtype),
        "has_Bx": has_Bx,
        "has_By": has_By,
    }
    return sup


# =========================================================
# 8) Piecewise model evaluation / 分区域预测
# =========================================================
def predict_region(model: MultiDomainPINN2D, xy: torch.Tensor, region_id: int):
    """Predict fields in one region / 在单一区域内预测场"""
    subnet = getattr(model, f"net{region_id}")
    return model.fields(subnet, xy, region_id)


def predict_piecewise(model: MultiDomainPINN2D, xy: torch.Tensor):
    """
    Predict piecewise fields for arbitrary interior points.
    对任意内部点按所属区域进行分段预测。
    """
    phys = model.phys
    rho = radius_from_xy(xy)[:, 0]

    a = torch.zeros((xy.shape[0], 1), device=xy.device, dtype=xy.dtype)
    qx = torch.zeros_like(a)
    qy = torch.zeros_like(a)

    m1 = rho <= phys.rho_coil
    m2 = (rho > phys.rho_coil) & (rho <= phys.rho_fe_in)
    m3 = (rho > phys.rho_fe_in) & (rho <= phys.rho_fe_out)
    m4 = (rho > phys.rho_fe_out) & (rho <= 1.0)

    if torch.any(m1):
        a1, qx1, qy1 = predict_region(model, xy[m1], 1)
        a[m1] = a1
        qx[m1] = qx1
        qy[m1] = qy1
    if torch.any(m2):
        a2, qx2, qy2 = predict_region(model, xy[m2], 2)
        a[m2] = a2
        qx[m2] = qx2
        qy[m2] = qy2
    if torch.any(m3):
        a3, qx3, qy3 = predict_region(model, xy[m3], 3)
        a[m3] = a3
        qx[m3] = qx3
        qy[m3] = qy3
    if torch.any(m4):
        a4, qx4, qy4 = predict_region(model, xy[m4], 4)
        a[m4] = a4
        qx[m4] = qx4
        qy[m4] = qy4

    return a, qx, qy


# =========================================================
# 9) PDE and interface losses / PDE 与界面损失
# =========================================================
def pde_residual_region(
    model: MultiDomainPINN2D,
    xy: torch.Tensor,
    region_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Mixed first-order residuals in one region.
    单一区域的一阶混合残差。

    Residuals / 残差:
        res_qx  = qhx - (1/mur) * da/dX
        res_qy  = qhy - (1/mur) * da/dY
        res_div = dqhx/dX + dqhy/dY + Jhat
    """
    a, qx, qy = predict_region(model, xy, region_id)
    grad_a = grad_scalar(a, xy)
    grad_qx = grad_scalar(qx, xy)
    grad_qy = grad_scalar(qy, xy)

    inv_mur = 1.0 / mu_r_of_region(region_id, model.phys)
    Jhat = Jhat_of_region(region_id)

    res_qx = qx - inv_mur * grad_a[:, 0:1]
    res_qy = qy - inv_mur * grad_a[:, 1:2]
    res_div = grad_qx[:, 0:1] + grad_qy[:, 1:2] + Jhat

    return res_qx, res_qy, res_div, a, qx, qy


def interface_loss(
    model: MultiDomainPINN2D,
    xy_if: torch.Tensor,
    left_id: int,
    right_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Interface loss between two subdomains.
    两个子区域之间的界面损失。

    Enforce / 约束:
        a^- = a^+
        q^-·n = q^+·n
    """
    aL, qxL, qyL = predict_region(model, xy_if, left_id)
    aR, qxR, qyR = predict_region(model, xy_if, right_id)

    nvec = unit_normal_on_circle(xy_if)
    qnL = qxL * nvec[:, 0:1] + qyL * nvec[:, 1:2]
    qnR = qxR * nvec[:, 0:1] + qyR * nvec[:, 1:2]

    loss_a = torch.mean((aL - aR) ** 2)
    loss_qn = torch.mean((qnL - qnR) ** 2)
    loss_if = loss_a + loss_qn
    return loss_if, loss_a.detach(), loss_qn.detach()


def compute_loss(
    model: MultiDomainPINN2D,
    colloc: Dict[str, torch.Tensor],
    train: TrainParams,
    sup_data: Optional[Dict[str, torch.Tensor]] = None,
):
    """Compute full loss / 计算总损失"""
    # -----------------------------------------------------
    # PDE losses / PDE 残差损失
    # -----------------------------------------------------
    r1_qx, r1_qy, r1_div, _, _, _ = pde_residual_region(model, colloc["xy1"], 1)
    r2_qx, r2_qy, r2_div, _, _, _ = pde_residual_region(model, colloc["xy2"], 2)
    r3_qx, r3_qy, r3_div, _, _, _ = pde_residual_region(model, colloc["xy3"], 3)
    r4_qx, r4_qy, r4_div, _, _, _ = pde_residual_region(model, colloc["xy4"], 4)

    loss_r1 = torch.mean(r1_qx ** 2) + torch.mean(r1_qy ** 2) + torch.mean(r1_div ** 2)
    loss_r2 = torch.mean(r2_qx ** 2) + torch.mean(r2_qy ** 2) + torch.mean(r2_div ** 2)
    loss_r3 = torch.mean(r3_qx ** 2) + torch.mean(r3_qy ** 2) + torch.mean(r3_div ** 2)
    loss_r4 = torch.mean(r4_qx ** 2) + torch.mean(r4_qy ** 2) + torch.mean(r4_div ** 2)

    # Slightly emphasize the thin ferro ring / 对薄铁磁层稍微加权
    loss_pde = 1.0 * loss_r1 + 1.0 * loss_r2 + 2.0 * loss_r3 + 1.0 * loss_r4

    # -----------------------------------------------------
    # Interface losses / 界面连续损失
    # -----------------------------------------------------
    loss_if12, loss_if12_a, loss_if12_qn = interface_loss(model, colloc["if12"], 1, 2)
    loss_if23, loss_if23_a, loss_if23_qn = interface_loss(model, colloc["if23"], 2, 3)
    loss_if34, loss_if34_a, loss_if34_qn = interface_loss(model, colloc["if34"], 3, 4)
    loss_if = loss_if12 + loss_if23 + loss_if34

    # -----------------------------------------------------
    # FEM supervision / FEM 点监督
    # -----------------------------------------------------
    loss_sup_A = torch.tensor(0.0, device=colloc["xy1"].device, dtype=colloc["xy1"].dtype)
    loss_sup_B = torch.tensor(0.0, device=colloc["xy1"].device, dtype=colloc["xy1"].dtype)

    if (sup_data is not None) and train.use_fem_supervision:
        xy_sup = sup_data["xy"]
        w_sup = sup_data["w"]
        a_ref = sup_data["a"]

        a_pred, qx_pred, qy_pred = predict_piecewise(model, xy_sup)
        loss_sup_A = torch.mean(w_sup * (a_pred - a_ref) ** 2)

        rho = radius_from_xy(xy_sup)[:, 0]
        mur = torch.ones((xy_sup.shape[0], 1), device=xy_sup.device, dtype=xy_sup.dtype)
        m3 = (rho > model.phys.rho_fe_in) & (rho <= model.phys.rho_fe_out)
        mur[m3, :] = model.phys.mur_fe

        # Bx = mu * qy,  By = -mu * qx
        bx_pred = mur * qy_pred
        by_pred = -mur * qx_pred

        if sup_data["has_Bx"]:
            loss_sup_B = loss_sup_B + torch.mean(w_sup * (bx_pred - sup_data["bx"]) ** 2)
        if sup_data["has_By"]:
            loss_sup_B = loss_sup_B + torch.mean(w_sup * (by_pred - sup_data["by"]) ** 2)

    # -----------------------------------------------------
    # Total loss / 总损失
    # -----------------------------------------------------
    loss = (
        train.w_pde * loss_pde
        + train.w_if * loss_if
        + train.w_sup_A * loss_sup_A
        + train.w_sup_B * loss_sup_B
    )

    info = {
        "loss": float(loss.detach().cpu()),
        "loss_pde": float(loss_pde.detach().cpu()),
        "loss_if": float(loss_if.detach().cpu()),
        "loss_sup_A": float(loss_sup_A.detach().cpu()),
        "loss_sup_B": float(loss_sup_B.detach().cpu()),
        "loss_r1": float(loss_r1.detach().cpu()),
        "loss_r2": float(loss_r2.detach().cpu()),
        "loss_r3": float(loss_r3.detach().cpu()),
        "loss_r4": float(loss_r4.detach().cpu()),
        "loss_if12_a": float(loss_if12_a.cpu()),
        "loss_if12_qn": float(loss_if12_qn.cpu()),
        "loss_if23_a": float(loss_if23_a.cpu()),
        "loss_if23_qn": float(loss_if23_qn.cpu()),
        "loss_if34_a": float(loss_if34_a.cpu()),
        "loss_if34_qn": float(loss_if34_qn.cpu()),
    }
    return loss, info


# =========================================================
# 10) Training / 训练
# =========================================================
def train_model(
    model: MultiDomainPINN2D,
    phys: PhysParams,
    train: TrainParams,
    device,
    dtype,
    sup_data: Optional[Dict[str, torch.Tensor]] = None,
):
    """Train PINN / 训练 PINN"""
    hist = {
        "total": [],
        "pde": [],
        "interface": [],
        "sup_A": [],
        "sup_B": [],
        "r1": [], "r2": [], "r3": [], "r4": [],
        "if12_a": [], "if12_qn": [],
        "if23_a": [], "if23_qn": [],
        "if34_a": [], "if34_qn": [],
    }

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=train.lr_adam)
    t0 = time.time()

    colloc = build_collocation(phys, train, device, dtype)

    for ep in range(1, train.adam_epochs + 1):
        if ep > 1 and (ep % train.resample_every == 1):
            colloc = build_collocation(phys, train, device, dtype)

        optimizer.zero_grad()
        loss, info = compute_loss(model, colloc, train, sup_data=sup_data)
        loss.backward()
        optimizer.step()

        hist["total"].append(info["loss"])
        hist["pde"].append(info["loss_pde"])
        hist["interface"].append(info["loss_if"])
        hist["sup_A"].append(info["loss_sup_A"])
        hist["sup_B"].append(info["loss_sup_B"])
        hist["r1"].append(info["loss_r1"])
        hist["r2"].append(info["loss_r2"])
        hist["r3"].append(info["loss_r3"])
        hist["r4"].append(info["loss_r4"])
        hist["if12_a"].append(info["loss_if12_a"])
        hist["if12_qn"].append(info["loss_if12_qn"])
        hist["if23_a"].append(info["loss_if23_a"])
        hist["if23_qn"].append(info["loss_if23_qn"])
        hist["if34_a"].append(info["loss_if34_a"])
        hist["if34_qn"].append(info["loss_if34_qn"])

        if ep == 1 or ep % train.print_every == 0:
            print(
                f"[Adam] {ep:5d}/{train.adam_epochs:5d} | "
                f"Total={info['loss']:.3e} | PDE={info['loss_pde']:.3e} | "
                f"IF={info['loss_if']:.3e} | SUP-A={info['loss_sup_A']:.3e} | SUP-B={info['loss_sup_B']:.3e}"
            )

    print(f"Adam finished in {time.time() - t0:.2f} s")

    if train.lbfgs_steps > 0:
        model.train()
        fixed = build_collocation(phys, train, device, dtype)
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=train.lbfgs_steps,
            max_eval=train.lbfgs_steps,
            tolerance_grad=1e-12,
            tolerance_change=1e-12,
            history_size=50,
            line_search_fn="strong_wolfe",
        )
        counter = {"k": 0}
        t1 = time.time()

        def closure():
            lbfgs.zero_grad()
            loss, info = compute_loss(model, fixed, train, sup_data=sup_data)
            loss.backward()
            counter["k"] += 1

            hist["total"].append(info["loss"])
            hist["pde"].append(info["loss_pde"])
            hist["interface"].append(info["loss_if"])
            hist["sup_A"].append(info["loss_sup_A"])
            hist["sup_B"].append(info["loss_sup_B"])
            hist["r1"].append(info["loss_r1"])
            hist["r2"].append(info["loss_r2"])
            hist["r3"].append(info["loss_r3"])
            hist["r4"].append(info["loss_r4"])
            hist["if12_a"].append(info["loss_if12_a"])
            hist["if12_qn"].append(info["loss_if12_qn"])
            hist["if23_a"].append(info["loss_if23_a"])
            hist["if23_qn"].append(info["loss_if23_qn"])
            hist["if34_a"].append(info["loss_if34_a"])
            hist["if34_qn"].append(info["loss_if34_qn"])

            if counter["k"] == 1 or counter["k"] % 50 == 0:
                print(
                    f"[LBFGS] {counter['k']:4d}/{train.lbfgs_steps:4d} | "
                    f"Total={info['loss']:.3e} | PDE={info['loss_pde']:.3e} | "
                    f"IF={info['loss_if']:.3e} | SUP-A={info['loss_sup_A']:.3e} | SUP-B={info['loss_sup_B']:.3e}"
                )
            return loss

        lbfgs.step(closure)
        print(f"LBFGS finished in {time.time() - t1:.2f} s")

    return hist


# =========================================================
# 11) Exact radial solution / 径向解析解
# =========================================================
def exact_radial_solution(
    r: np.ndarray,
    mu0: float,
    mu_fe: float,
    J0: float,
    r_coil: float,
    r_fe_in: float,
    r_fe_out: float,
    R: float,
):
    """
    Exact radial solution for verification.
    用于验证的径向解析解。

    Output / 输出:
        A(r), B_theta(r)
    """
    r = np.asarray(r, dtype=float)
    A = np.zeros_like(r)
    B = np.zeros_like(r)

    Q = -0.5 * J0 * r_coil ** 2

    def A4(rr):
        return mu0 * Q * np.log(rr / R)

    A4_rfo = A4(r_fe_out)

    def A3(rr):
        return A4_rfo + mu_fe * Q * np.log(rr / r_fe_out)

    A3_rfi = A3(r_fe_in)

    def A2(rr):
        return A3_rfi + mu0 * Q * np.log(rr / r_fe_in)

    A2_rc = A2(r_coil)
    C1 = A2_rc + mu0 * J0 * r_coil ** 2 / 4.0

    def A1(rr):
        return C1 - mu0 * J0 * rr ** 2 / 4.0

    m1 = r <= r_coil
    m2 = (r > r_coil) & (r <= r_fe_in)
    m3 = (r > r_fe_in) & (r <= r_fe_out)
    m4 = r > r_fe_out

    A[m1] = A1(r[m1])
    A[m2] = A2(r[m2])
    A[m3] = A3(r[m3])
    A[m4] = A4(r[m4])

    B[m1] = mu0 * J0 * r[m1] / 2.0
    B[m2] = mu0 * J0 * r_coil ** 2 / (2.0 * r[m2])
    B[m3] = mu_fe * J0 * r_coil ** 2 / (2.0 * r[m3])
    B[m4] = mu0 * J0 * r_coil ** 2 / (2.0 * r[m4])

    if np.any(r == 0.0):
        B[r == 0.0] = 0.0

    return A, B


def exact_solution_from_phys(r: np.ndarray, phys: PhysParams):
    """Wrapper using PhysParams / 使用 PhysParams 的封装"""
    return exact_radial_solution(
        r=r,
        mu0=phys.mu0,
        mu_fe=phys.mu_fe,
        J0=phys.J0,
        r_coil=phys.r_coil,
        r_fe_in=phys.r_fe_in,
        r_fe_out=phys.r_fe_out,
        R=phys.R,
    )


# =========================================================
# 12) Post-processing / 后处理
# =========================================================
def eval_on_grid(model: MultiDomainPINN2D, phys: PhysParams, n_plot: int, device, dtype):
    """
    Evaluate PINN on a Cartesian grid and mask outside the disk.
    在笛卡尔网格上评估 PINN, 并对圆外区域做掩膜。
    """
    x = np.linspace(-1.0, 1.0, n_plot)
    y = np.linspace(-1.0, 1.0, n_plot)
    XX, YY = np.meshgrid(x, y, indexing="xy")
    RR = np.sqrt(XX ** 2 + YY ** 2)
    inside = RR <= 1.0

    xy_np = np.column_stack([XX[inside], YY[inside]])
    xy = torch.tensor(xy_np, device=device, dtype=dtype, requires_grad=False)

    model.eval()
    with torch.no_grad():
        a, qx, qy = predict_piecewise(model, xy)

        rho = np.sqrt(xy_np[:, 0] ** 2 + xy_np[:, 1] ** 2)
        mur = np.ones((xy_np.shape[0], 1), dtype=np.float64)
        m3 = (rho > phys.rho_fe_in) & (rho <= phys.rho_fe_out)
        mur[m3, :] = phys.mur_fe
        mur_t = torch.tensor(mur, device=device, dtype=dtype)

        bx = mur_t * qy
        by = -mur_t * qx
        bmag = torch.sqrt(bx ** 2 + by ** 2)

    A_grid = np.full_like(XX, np.nan, dtype=np.float64)
    Bmag_grid = np.full_like(XX, np.nan, dtype=np.float64)
    A_grid[inside] = (phys.A0 * a.detach().cpu().numpy().reshape(-1))
    Bmag_grid[inside] = (phys.B0 * bmag.detach().cpu().numpy().reshape(-1))

    Xcm = XX * phys.R * 100.0
    Ycm = YY * phys.R * 100.0
    return Xcm, Ycm, A_grid, Bmag_grid


def eval_radial_line(model: MultiDomainPINN2D, phys: PhysParams, device, dtype, n_line: int = 1200):
    """
    Evaluate along y=0, x>=0 for comparison with exact radial solution.
    沿 y=0, x>=0 方向评估，用于和解析径向解比较。
    """
    rho = np.linspace(0.0, 1.0, n_line)
    X = rho.copy()
    Y = np.zeros_like(X)
    xy = torch.tensor(np.column_stack([X, Y]), device=device, dtype=dtype, requires_grad=False)

    # To avoid exact interface ambiguity, shift a tiny amount.
    # 避免恰好落在界面上导致分区歧义，做极小扰动。
    eps = 1.0e-12
    X[0] = 0.0
    Y[:] = eps
    xy = torch.tensor(np.column_stack([X, Y]), device=device, dtype=dtype, requires_grad=False)

    model.eval()
    with torch.no_grad():
        a, qx, qy = predict_piecewise(model, xy)
        # On y≈0 and x>=0, for radial solution B magnitude can be computed directly
        # 这里仍然统一用分量方式计算 |B|
        rho_eval = np.sqrt(X ** 2 + Y ** 2)
        mur = np.ones((len(rho_eval), 1), dtype=np.float64)
        m3 = (rho_eval > phys.rho_fe_in) & (rho_eval <= phys.rho_fe_out)
        mur[m3, :] = phys.mur_fe
        mur_t = torch.tensor(mur, device=device, dtype=dtype)
        bx = mur_t * qy
        by = -mur_t * qx
        bmag = torch.sqrt(bx ** 2 + by ** 2)

    r_m = rho * phys.R
    A_pinn = phys.A0 * a.detach().cpu().numpy().reshape(-1)
    B_pinn = phys.B0 * bmag.detach().cpu().numpy().reshape(-1)
    A_exact, B_exact = exact_solution_from_phys(r_m, phys)
    return r_m, A_pinn, B_pinn, A_exact, B_exact


def save_plots(
    Xcm: np.ndarray,
    Ycm: np.ndarray,
    A_grid: np.ndarray,
    Bmag_grid: np.ndarray,
    hist: Dict[str, list],
    phys: PhysParams,
    model: MultiDomainPINN2D,
    save_dir: str,
    device,
    dtype,
):
    """Save post-processing figures / 保存后处理图像"""
    os.makedirs(save_dir, exist_ok=True)

    # -----------------------------------------------------
    # A contour / A 云图
    # -----------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    cf1 = ax1.contourf(Xcm, Ycm, A_grid, levels=40)
    ax1.contour(Xcm, Ycm, A_grid, levels=10, colors="k", linewidths=0.2)
    ax1.set_aspect("equal")
    ax1.set_xlabel("x / cm")
    ax1.set_ylabel("y / cm")
    ax1.set_title(r"PINN: Magnetic Vector Potential $A(x,y)$")
    fig1.colorbar(cf1, ax=ax1, label="A (Wb/m)")
    fig1.tight_layout()
    p1 = os.path.join(save_dir, "pinn_A_contour_2D.png")
    fig1.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close(fig1)

    # -----------------------------------------------------
    # |B| magnitude / |B| 云图
    # -----------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    cf2 = ax2.contourf(Xcm, Ycm, Bmag_grid, levels=40)
    ax2.set_aspect("equal")
    ax2.set_xlabel("x / cm")
    ax2.set_ylabel("y / cm")
    ax2.set_title(r"PINN: Magnetic Flux Density Magnitude $|\mathbf{B}|$")
    fig2.colorbar(cf2, ax=ax2, label=r"$|\mathbf{B}|$ (T)")
    fig2.tight_layout()
    p2 = os.path.join(save_dir, "pinn_Bmag_2D.png")
    fig2.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close(fig2)

    # -----------------------------------------------------
    # Radial comparison / 径向对比图
    # -----------------------------------------------------
    r_m, A_pinn, B_pinn, A_exact, B_exact = eval_radial_line(model, phys, device, dtype)
    r_cm = r_m * 100.0

    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax3a.plot(r_cm, A_pinn, linewidth=2.0, label="PINN")
    ax3a.plot(r_cm, A_exact, "--", linewidth=1.5, label="Exact")
    ax3a.axvline(phys.r_coil_cm, linestyle="--", linewidth=1.0, color="gray")
    ax3a.axvline(phys.r_fe_in_cm, linestyle="--", linewidth=1.0, color="gray")
    ax3a.axvline(phys.r_fe_out_cm, linestyle="--", linewidth=1.0, color="gray")
    ax3a.set_xlabel("r / cm")
    ax3a.set_ylabel("A (Wb/m)")
    ax3a.set_title("Radial comparison of A")
    ax3a.grid(True, alpha=0.3)
    ax3a.legend()

    ax3b.plot(r_cm, B_pinn, linewidth=2.0, label="PINN")
    ax3b.plot(r_cm, B_exact, "--", linewidth=1.5, label="Exact")
    ax3b.axvline(phys.r_coil_cm, linestyle="--", linewidth=1.0, color="gray")
    ax3b.axvline(phys.r_fe_in_cm, linestyle="--", linewidth=1.0, color="gray")
    ax3b.axvline(phys.r_fe_out_cm, linestyle="--", linewidth=1.0, color="gray")
    ax3b.set_xlabel("r / cm")
    ax3b.set_ylabel(r"$|\mathbf{B}|$ (T)")
    ax3b.set_title("Radial comparison of |B|")
    ax3b.grid(True, alpha=0.3)
    ax3b.legend()

    fig3.tight_layout()
    p3 = os.path.join(save_dir, "pinn_radial_comparison.png")
    fig3.savefig(p3, dpi=300, bbox_inches="tight")
    plt.close(fig3)

    # -----------------------------------------------------
    # Loss history / 损失历史
    # -----------------------------------------------------
    fig4, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.semilogy(hist["total"], label="total")
    ax.semilogy(hist["pde"], label="pde")
    ax.semilogy(hist["interface"], label="interface")
    if max(hist["sup_A"]) > 0.0:
        ax.semilogy(hist["sup_A"], label="sup_A")
    if max(hist["sup_B"]) > 0.0:
        ax.semilogy(hist["sup_B"], label="sup_B")
    ax.set_title("Total loss history")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.semilogy(hist["r1"], label="coil")
    ax.semilogy(hist["r2"], label="air gap")
    ax.semilogy(hist["r3"], label="ferro")
    ax.semilogy(hist["r4"], label="outer air")
    ax.set_title("PDE region losses")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.semilogy(hist["if12_a"], label="if12 A")
    ax.semilogy(hist["if12_qn"], label="if12 q·n")
    ax.semilogy(hist["if23_a"], label="if23 A")
    ax.semilogy(hist["if23_qn"], label="if23 q·n")
    ax.set_title("Interface losses (1)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)

    ax = axes[1, 1]
    ax.semilogy(hist["if34_a"], label="if34 A")
    ax.semilogy(hist["if34_qn"], label="if34 q·n")
    ax.set_title("Interface losses (2)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig4.tight_layout()
    p4 = os.path.join(save_dir, "pinn_loss_history_2D.png")
    fig4.savefig(p4, dpi=300, bbox_inches="tight")
    plt.close(fig4)

    return p1, p2, p3, p4

# =========================================================
# 12.5) Checkpoint utilities / 断点续训辅助函数
# =========================================================
def try_resume_model(model: nn.Module, ckpt_path: str, device, strict: bool = True):
    """
    Load checkpoint if it exists / 若存在 checkpoint, 则加载模型参数

    Parameters
    ----------
    model : nn.Module
        PINN model / PINN 模型
    ckpt_path : str
        Path to checkpoint / checkpoint 路径
    device :
        torch device / 设备
    strict : bool
        Strict state_dict loading or not / 是否严格加载

    Returns
    -------
    resumed : bool
        Whether checkpoint was successfully loaded / 是否成功续训
    ckpt : dict or None
        Loaded checkpoint dictionary / 加载出的 checkpoint 字典
    """
    if not os.path.exists(ckpt_path):
        print(f"[Resume] Checkpoint not found: {ckpt_path}")
        print("[Resume] Training will start from scratch.")
        return False, None

    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" not in ckpt:
        raise KeyError(f"Checkpoint missing key: 'model_state_dict' -> {ckpt_path}")

    model.load_state_dict(ckpt["model_state_dict"], strict=strict)

    print("=" * 78)
    print(f"[Resume] Loaded checkpoint: {ckpt_path}")
    if "train_params" in ckpt:
        print("[Resume] Checkpoint contains saved train_params.")
    if "phys_params" in ckpt:
        print("[Resume] Checkpoint contains saved phys_params.")
    print("=" * 78)

    return True, ckpt

# =========================================================
# 13) Main / 主程序
# =========================================================
def main():
    set_seed(42)
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    phys = PhysParams()
    train = TrainParams()

    print("=" * 78)
    print("PyTorch 2D multi-domain PINN for Model 3 magnetostatics")
    print(f"Device = {device}")
    print(f"A0 = {phys.A0:.6e} Wb/m, q0 = {phys.q0:.6e} A/m, B0 = {phys.B0:.6e} T")
    print("=" * 78)

    # FEM supervision /  FEM 监督
    base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    sup_data = None
    if train.use_fem_supervision:
        fem_csv = os.path.join(base_dir, train.fem_csv_relpath)
        if os.path.exists(fem_csv):
            sup_data = load_fem_supervision(fem_csv, phys, device, dtype)
            print(f"Loaded FEM supervision points: {sup_data['xy'].shape[0]}")
            print(f"FEM CSV: {fem_csv}")
        else:
            print(f"[Warning] FEM supervision file not found: {fem_csv}")
            print("Training will continue without FEM supervision.")
            train.use_fem_supervision = False

    model = MultiDomainPINN2D(phys, train).to(device=device, dtype=dtype)

    # -----------------------------------------------------
    # Resume from checkpoint / 从 checkpoint 继续训练
    # -----------------------------------------------------
    ckpt_path = os.path.join(base_dir, train.checkpoint_relpath)
    if train.resume_from_checkpoint:
        try_resume_model(
            model=model,
            ckpt_path=ckpt_path,
            device=device,
            strict=train.strict_resume,
        )

    hist = train_model(model, phys, train, device, dtype, sup_data=sup_data)

    # Save checkpoint / 保存模型参数
    ckpt_path = os.path.join(base_dir, "model3_pinn_2d_checkpoint.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "phys_params": vars(phys),
        "train_params": vars(train),
        "dtype": str(dtype),
        "device": str(device),
    }, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    # Post-processing / 后处理
    Xcm, Ycm, A_grid, Bmag_grid = eval_on_grid(model, phys, train.n_plot, device, dtype)
    p1, p2, p3, p4 = save_plots(Xcm, Ycm, A_grid, Bmag_grid, hist, phys, model, base_dir, device, dtype)

    # Quick numerical comparison / 快速数值比较
    r_m, A_pinn, B_pinn, A_exact, B_exact = eval_radial_line(model, phys, device, dtype)
    rel_A = np.linalg.norm(A_pinn - A_exact) / (np.linalg.norm(A_exact) + 1.0e-30)
    rel_B = np.linalg.norm(B_pinn - B_exact) / (np.linalg.norm(B_exact) + 1.0e-30)

    print("\n" + "=" * 78)
    print("2D PINN solve finished")
    print(f"Relative L2 error of A along radial line   = {rel_A:.6e}")
    print(f"Relative L2 error of |B| along radial line = {rel_B:.6e}")
    print(f"Saved: {p1}")
    print(f"Saved: {p2}")
    print(f"Saved: {p3}")
    print(f"Saved: {p4}")
    print("=" * 78)


if __name__ == "__main__":
    main()
