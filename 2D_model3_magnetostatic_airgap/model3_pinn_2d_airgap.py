# -*- coding: utf-8 -*-
"""
Model 3 with a narrow air gap in the ferromagnetic ring (PyTorch PINN)
模型3: 在铁磁环上引入窄空气间隙的二维 PINN

Geometry assumption / 几何假设:
    A narrow rectangular air gap is cut through the ferromagnetic ring near the +x direction.
    在 +x 方向附近，用一个窄矩形缝隙近似铁磁环空气间隙。

    Gap width / 间隙宽度: 0.1 cm  (tangential / 沿 y 方向)
    Gap spans the full ring thickness / 间隙贯穿整个铁磁环厚度:
        x in [r_fe_in, r_fe_out], y in [-gap/2, +gap/2]

Key idea / 核心思路:
    - Keep the same 4-subnetwork architecture so that the model3_pinn_2d_checkpoint.pth can be loaded directly.
      保持与旧模型一致的 4 子网络结构，这样 model3_pinn_2d_checkpoint.pth 可直接加载。
    - Region 2 becomes: inner air + air gap
      第2区改成: 内空气区 + 空气间隙
    - Region 3 becomes: ferromagnetic ring excluding the air gap
      第3区改成: 除去空气间隙后的铁磁环
    - Region 4 remains outer air with hard BC A=0 on the outer circle.
      第4区仍是外空气区, 并保留外边界 A=0 的硬约束。

Outputs / 输出:
    1) A(x,y) contour
    2) |B|(x,y) contour
    3) x-axis line cut through the gap
    4) loss history
    5) checkpoint after gap training
"""

import os
import math
import time
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# =========================================================
# 0) Reproducibility / 随机种子
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# 1) Physical parameters / 物理参数
# =========================================================
@dataclass
class PhysParams:
    mu0: float = 4.0 * math.pi * 1.0e-7
    mur_fe: float = 1000.0

    r_coil_cm: float = 5.0
    r_fe_in_cm: float = 10.0
    r_fe_out_cm: float = 10.5
    r_out_cm: float = 30.0

    J0_cm2: float = 1.0

    # New air-gap parameter / 新空气间隙参数
    gap_width_cm: float = 0.1

    def __post_init__(self):
        cm_to_m = 1.0e-2
        self.mu_fe = self.mur_fe * self.mu0
        self.r_coil = self.r_coil_cm * cm_to_m
        self.r_fe_in = self.r_fe_in_cm * cm_to_m
        self.r_fe_out = self.r_fe_out_cm * cm_to_m
        self.R = self.r_out_cm * cm_to_m
        self.J0 = self.J0_cm2 * 1.0e4
        self.gap_width = self.gap_width_cm * cm_to_m
        self.gap_half = 0.5 * self.gap_width

        self.rho_coil = self.r_coil / self.R
        self.rho_fe_in = self.r_fe_in / self.R
        self.rho_fe_out = self.r_fe_out / self.R
        self.gap_half_rho = self.gap_half / self.R

        self.A0 = self.mu0 * self.J0 * self.R ** 2
        self.q0 = self.J0 * self.R
        self.B0 = self.mu0 * self.J0 * self.R

        if not (0.0 < self.rho_coil < self.rho_fe_in < self.rho_fe_out < 1.0):
            raise ValueError("Geometry radii must satisfy 0 < r_coil < r_fe_in < r_fe_out < R")


# =========================================================
# 2) Training parameters / 训练参数
# =========================================================
@dataclass
class TrainParams:
    hidden_layers: int = 4
    hidden_units: int = 64

    # Domain collocation points / 域内配点
    N1: int = 2200            # coil
    N2_inner: int = 2200      # inner air annulus
    N2_gap: int = 2400        # air gap rectangle
    N3: int = 5200            # ferro body
    N4: int = 4200            # outer air

    # Interface points / 界面点
    Nif12: int = 1024
    Nif23: int = 1536         # inner circular air-ferro interface, excluding the gap opening
    Nif34: int = 1536         # outer circular ferro-air interface, excluding the gap opening
    Nif_gap_tb: int = 1024    # top / bottom faces of the gap
    Nif24: int = 768          # artificial air-air interface at x=r_fe_out inside the gap

    # Plot grid / 绘图网格
    n_plot: int = 361
    n_line: int = 1400

    # Optimization / 优化
    adam_epochs: int = 5000
    lbfgs_steps: int = 0
    lr_adam: float = 2e-4
    print_every: int = 200
    resample_every: int = 500

    # Loss weights / 损失权重
    w_pde: float = 1.0
    w_if: float = 100.0

    # Resume from the no-gap checkpoint / 从无间隙模型继续训练
    resume_from_checkpoint: bool = True #False
    checkpoint_relpath: str = "model3_pinn_2d_checkpoint.pth"
    strict_resume: bool = True

# =========================================================
# 3) Neural network blocks / 神经网络模块
# =========================================================
class SubNet2D(nn.Module):
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


class MultiDomainPINN2DGap(nn.Module):
    """
    4 subnetworks are kept exactly as in the old no-gap model.
    保持与旧无间隙模型完全相同的 4 子网络结构。

    Region 1: coil
    Region 2: inner air + air gap
    Region 3: ferro body excluding gap
    Region 4: outer air
    """
    def __init__(self, phys: PhysParams, train: TrainParams):
        super().__init__()
        self.phys = phys
        self.net1 = SubNet2D(train.hidden_layers, train.hidden_units)
        self.net2 = SubNet2D(train.hidden_layers, train.hidden_units)
        self.net3 = SubNet2D(train.hidden_layers, train.hidden_units)
        self.net4 = SubNet2D(train.hidden_layers, train.hidden_units)

    def fields(self, subnet: SubNet2D, xy: torch.Tensor, region_id: int):
        raw = subnet(xy)
        a_raw = raw[:, 0:1]
        qx_raw = raw[:, 1:2]
        qy_raw = raw[:, 2:3]

        # Keep the original hard BC only for outer-air network.
        # 仅对外空气网络保留原来的外边界硬约束。
        if region_id == 4:
            x = xy[:, 0:1]
            y = xy[:, 1:2]
            rho2 = x * x + y * y
            a = (1.0 - rho2) * a_raw
        else:
            a = a_raw

        return a, qx_raw, qy_raw


# =========================================================
# 4) Differential utilities / 自动微分工具
# =========================================================
def grad_scalar(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True, #False,
        retain_graph=True,
        only_inputs=True,
    )[0]


# =========================================================
# 5) Geometry helpers / 几何辅助函数
# =========================================================
def radius_from_xy(xy: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.clamp(xy[:, 0:1] ** 2 + xy[:, 1:2] ** 2, min=1.0e-30))


def unit_normal_on_circle(xy: torch.Tensor) -> torch.Tensor:
    rho = radius_from_xy(xy)
    return xy / rho


def is_in_gap(xy: torch.Tensor, phys: PhysParams) -> torch.Tensor:
    """
    Narrow rectangular slit through the ring / 穿过铁磁环的窄矩形缝隙
    In dimensionless coordinates:
        X in [rho_fe_in, rho_fe_out], |Y| <= gap_half_rho
    """
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    return (
        (x >= phys.rho_fe_in)
        & (x <= phys.rho_fe_out)
        & (torch.abs(y) <= phys.gap_half_rho)
    )[:, 0]


def sample_disk(n: int, rho_max: float, device, dtype) -> torch.Tensor:
    u = torch.rand(n, 1, device=device, dtype=dtype)
    theta = 2.0 * math.pi * torch.rand(n, 1, device=device, dtype=dtype)
    rho = rho_max * torch.sqrt(u)
    x = rho * torch.cos(theta)
    y = rho * torch.sin(theta)
    xy = torch.cat([x, y], dim=1)
    xy.requires_grad_(True)
    return xy


def sample_annulus(n: int, rho_in: float, rho_out: float, device, dtype) -> torch.Tensor:
    u = torch.rand(n, 1, device=device, dtype=dtype)
    theta = 2.0 * math.pi * torch.rand(n, 1, device=device, dtype=dtype)
    rho = torch.sqrt((rho_out ** 2 - rho_in ** 2) * u + rho_in ** 2)
    x = rho * torch.cos(theta)
    y = rho * torch.sin(theta)
    xy = torch.cat([x, y], dim=1)
    xy.requires_grad_(True)
    return xy


def sample_annulus_edge_dense(n: int, rho_in: float, rho_out: float, device, dtype, power: float = 0.35) -> torch.Tensor:
    n1 = n // 2
    n2 = n - n1

    theta1 = 2.0 * math.pi * torch.rand(n1, 1, device=device, dtype=dtype)
    theta2 = 2.0 * math.pi * torch.rand(n2, 1, device=device, dtype=dtype)
    u1 = torch.rand(n1, 1, device=device, dtype=dtype)
    u2 = torch.rand(n2, 1, device=device, dtype=dtype)

    rho1 = rho_in + (rho_out - rho_in) * (u1 ** power)
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


def sample_rectangle(n: int, x_min: float, x_max: float, y_min: float, y_max: float, device, dtype) -> torch.Tensor:
    x = x_min + (x_max - x_min) * torch.rand(n, 1, device=device, dtype=dtype)
    y = y_min + (y_max - y_min) * torch.rand(n, 1, device=device, dtype=dtype)
    xy = torch.cat([x, y], dim=1)
    xy.requires_grad_(True)
    return xy


def sample_ferro_body(n: int, phys: PhysParams, device, dtype) -> torch.Tensor:
    """Sample points in the ferro ring excluding the air gap / 在除去间隙后的铁磁环中采样"""
    pts = []
    target = n
    while sum(p.shape[0] for p in pts) < target:
        cand = sample_annulus_edge_dense(max(2 * (target - sum(p.shape[0] for p in pts)), 512),
                                         phys.rho_fe_in, phys.rho_fe_out, device, dtype, power=0.35)
        keep = ~is_in_gap(cand, phys)
        kept = cand[keep]
        if kept.shape[0] > 0:
            pts.append(kept)
    xy = torch.cat(pts, dim=0)[:target]
    xy.requires_grad_(True)
    return xy


def sample_circle(n: int, rho: float, device, dtype) -> torch.Tensor:
    theta = 2.0 * math.pi * torch.rand(n, 1, device=device, dtype=dtype)
    x = rho * torch.cos(theta)
    y = rho * torch.sin(theta)
    xy = torch.cat([x, y], dim=1)
    xy.requires_grad_(True)
    return xy


def sample_circle_excluding_gap_arc(n: int, rho: float, gap_half_rho: float, device, dtype) -> torch.Tensor:
    """Sample points on a circle excluding the narrow arc around +x / 在圆上采样并排除 +x 方向间隙张角"""
    alpha = float(math.asin(min(0.999999, gap_half_rho / rho)))
    # allowed arc length = 2*pi - 2*alpha
    u = torch.rand(n, 1, device=device, dtype=dtype)
    theta = alpha + (2.0 * math.pi - 2.0 * alpha) * u
    x = rho * torch.cos(theta)
    y = rho * torch.sin(theta)
    xy = torch.cat([x, y], dim=1)
    xy.requires_grad_(True)
    return xy


def sample_horizontal_segment(n: int, x0: float, x1: float, y_const: float, device, dtype) -> torch.Tensor:
    x = x0 + (x1 - x0) * torch.rand(n, 1, device=device, dtype=dtype)
    y = torch.full_like(x, y_const)
    xy = torch.cat([x, y], dim=1)
    xy.requires_grad_(True)
    return xy


def sample_vertical_segment(n: int, x_const: float, y0: float, y1: float, device, dtype) -> torch.Tensor:
    y = y0 + (y1 - y0) * torch.rand(n, 1, device=device, dtype=dtype)
    x = torch.full_like(y, x_const)
    xy = torch.cat([x, y], dim=1)
    xy.requires_grad_(True)
    return xy


# =========================================================
# 6) Material and source maps / 材料与源项映射
# =========================================================
def mu_r_of_region(region_id: int, phys: PhysParams) -> float:
    return phys.mur_fe if region_id == 3 else 1.0


def Jhat_of_region(region_id: int) -> float:
    return 1.0 if region_id == 1 else 0.0


def build_collocation(phys: PhysParams, train: TrainParams, device, dtype) -> Dict[str, torch.Tensor]:
    colloc = {
        "xy1": sample_disk(train.N1, phys.rho_coil, device, dtype),
        "xy2_inner": sample_annulus(train.N2_inner, phys.rho_coil, phys.rho_fe_in, device, dtype),
        "xy2_gap": sample_rectangle(train.N2_gap, phys.rho_fe_in, phys.rho_fe_out,
                                     -phys.gap_half_rho, phys.gap_half_rho, device, dtype),
        "xy3": sample_ferro_body(train.N3, phys, device, dtype),
        "xy4": sample_annulus(train.N4, phys.rho_fe_out, 1.0, device, dtype),
        # Interfaces / 界面
        "if12": sample_circle(train.Nif12, phys.rho_coil, device, dtype),
        "if23": sample_circle_excluding_gap_arc(train.Nif23, phys.rho_fe_in, phys.gap_half_rho, device, dtype),
        "if34": sample_circle_excluding_gap_arc(train.Nif34, phys.rho_fe_out, phys.gap_half_rho, device, dtype),
        "if_gap_top": sample_horizontal_segment(train.Nif_gap_tb, phys.rho_fe_in, phys.rho_fe_out,
                                                 phys.gap_half_rho, device, dtype),
        "if_gap_bot": sample_horizontal_segment(train.Nif_gap_tb, phys.rho_fe_in, phys.rho_fe_out,
                                                 -phys.gap_half_rho, device, dtype),
        "if24": sample_vertical_segment(train.Nif24, phys.rho_fe_out, -phys.gap_half_rho, phys.gap_half_rho, device, dtype),
    }
    return colloc


# =========================================================
# 7) Piecewise prediction / 分区域预测
# =========================================================
def predict_region(model: MultiDomainPINN2DGap, xy: torch.Tensor, region_id: int):
    subnet = getattr(model, f"net{region_id}")
    return model.fields(subnet, xy, region_id)


def predict_piecewise(model: MultiDomainPINN2DGap, xy: torch.Tensor):
    phys = model.phys
    rho = radius_from_xy(xy)[:, 0]
    gap = is_in_gap(xy, phys)

    a = torch.zeros((xy.shape[0], 1), device=xy.device, dtype=xy.dtype)
    qx = torch.zeros_like(a)
    qy = torch.zeros_like(a)

    m1 = rho <= phys.rho_coil
    m2 = ((rho > phys.rho_coil) & (rho <= phys.rho_fe_in)) | gap
    m3 = (rho > phys.rho_fe_in) & (rho <= phys.rho_fe_out) & (~gap)
    m4 = (rho > phys.rho_fe_out) & (rho <= 1.0)

    if torch.any(m1):
        a1, qx1, qy1 = predict_region(model, xy[m1], 1)
        a[m1], qx[m1], qy[m1] = a1, qx1, qy1
    if torch.any(m2):
        a2, qx2, qy2 = predict_region(model, xy[m2], 2)
        a[m2], qx[m2], qy[m2] = a2, qx2, qy2
    if torch.any(m3):
        a3, qx3, qy3 = predict_region(model, xy[m3], 3)
        a[m3], qx[m3], qy[m3] = a3, qx3, qy3
    if torch.any(m4):
        a4, qx4, qy4 = predict_region(model, xy[m4], 4)
        a[m4], qx[m4], qy[m4] = a4, qx4, qy4

    return a, qx, qy


# =========================================================
# 8) PDE and interface losses / PDE 与界面损失
# =========================================================
def pde_residual_region(model: MultiDomainPINN2DGap, xy: torch.Tensor, region_id: int):
    a, qx, qy = predict_region(model, xy, region_id)
    grad_a = grad_scalar(a, xy)
    grad_qx = grad_scalar(qx, xy)
    grad_qy = grad_scalar(qy, xy)

    inv_mur = 1.0 / mu_r_of_region(region_id, model.phys)
    Jhat = Jhat_of_region(region_id)

    res_qx = qx - inv_mur * grad_a[:, 0:1]
    res_qy = qy - inv_mur * grad_a[:, 1:2]
    res_div = grad_qx[:, 0:1] + grad_qy[:, 1:2] + Jhat
    return res_qx, res_qy, res_div


def interface_loss(model: MultiDomainPINN2DGap, xy_if: torch.Tensor, left_id: int, right_id: int):
    aL, qxL, qyL = predict_region(model, xy_if, left_id)
    aR, qxR, qyR = predict_region(model, xy_if, right_id)

    # Normal direction / 法向方向
    if torch.allclose(xy_if[:, 1], torch.full_like(xy_if[:, 1], xy_if[0, 1].item())):
        # horizontal segment -> normal in y / 水平线段 -> y 法向
        if float(xy_if[0, 1].detach().cpu()) >= 0.0:
            nvec = torch.cat([torch.zeros_like(xy_if[:, 0:1]), torch.ones_like(xy_if[:, 1:2])], dim=1)
        else:
            nvec = torch.cat([torch.zeros_like(xy_if[:, 0:1]), -torch.ones_like(xy_if[:, 1:2])], dim=1)
    elif torch.allclose(xy_if[:, 0], torch.full_like(xy_if[:, 0], xy_if[0, 0].item())):
        # vertical segment -> normal in x / 垂直线段 -> x 法向
        nvec = torch.cat([torch.ones_like(xy_if[:, 0:1]), torch.zeros_like(xy_if[:, 1:2])], dim=1)
    else:
        # circular interface / 圆界面
        nvec = unit_normal_on_circle(xy_if)

    qnL = qxL * nvec[:, 0:1] + qyL * nvec[:, 1:2]
    qnR = qxR * nvec[:, 0:1] + qyR * nvec[:, 1:2]

    loss_a = torch.mean((aL - aR) ** 2)
    loss_qn = torch.mean((qnL - qnR) ** 2)
    return loss_a + loss_qn, loss_a.detach(), loss_qn.detach()


def compute_loss(model: MultiDomainPINN2DGap, colloc: Dict[str, torch.Tensor], train: TrainParams):
    # PDE residuals / PDE 残差
    r1_qx, r1_qy, r1_div = pde_residual_region(model, colloc["xy1"], 1)
    r2a_qx, r2a_qy, r2a_div = pde_residual_region(model, colloc["xy2_inner"], 2)
    r2g_qx, r2g_qy, r2g_div = pde_residual_region(model, colloc["xy2_gap"], 2)
    r3_qx, r3_qy, r3_div = pde_residual_region(model, colloc["xy3"], 3)
    r4_qx, r4_qy, r4_div = pde_residual_region(model, colloc["xy4"], 4)

    loss_r1 = torch.mean(r1_qx ** 2) + torch.mean(r1_qy ** 2) + torch.mean(r1_div ** 2)
    loss_r2_inner = torch.mean(r2a_qx ** 2) + torch.mean(r2a_qy ** 2) + torch.mean(r2a_div ** 2)
    loss_r2_gap = torch.mean(r2g_qx ** 2) + torch.mean(r2g_qy ** 2) + torch.mean(r2g_div ** 2)
    loss_r2 = loss_r2_inner + 2.0 * loss_r2_gap
    loss_r3 = torch.mean(r3_qx ** 2) + torch.mean(r3_qy ** 2) + torch.mean(r3_div ** 2)
    loss_r4 = torch.mean(r4_qx ** 2) + torch.mean(r4_qy ** 2) + torch.mean(r4_div ** 2)

    loss_pde = 1.0 * loss_r1 + 1.0 * loss_r2 + 2.0 * loss_r3 + 1.0 * loss_r4

    # Interfaces / 界面
    loss_if12, loss_if12_a, loss_if12_qn = interface_loss(model, colloc["if12"], 1, 2)
    loss_if23, loss_if23_a, loss_if23_qn = interface_loss(model, colloc["if23"], 2, 3)
    loss_if34, loss_if34_a, loss_if34_qn = interface_loss(model, colloc["if34"], 3, 4)
    loss_ifgt, loss_ifgt_a, loss_ifgt_qn = interface_loss(model, colloc["if_gap_top"], 2, 3)
    loss_ifgb, loss_ifgb_a, loss_ifgb_qn = interface_loss(model, colloc["if_gap_bot"], 2, 3)
    loss_if24, loss_if24_a, loss_if24_qn = interface_loss(model, colloc["if24"], 2, 4)

    loss_if = loss_if12 + loss_if23 + loss_if34 + loss_ifgt + loss_ifgb + 0.5 * loss_if24

    loss = train.w_pde * loss_pde + train.w_if * loss_if

    info = {
        "loss": float(loss.detach().cpu()),
        "loss_pde": float(loss_pde.detach().cpu()),
        "loss_if": float(loss_if.detach().cpu()),
        "loss_sup_A": 0.0,
        "loss_sup_B": 0.0,
        "loss_r1": float(loss_r1.detach().cpu()),
        "loss_r2": float(loss_r2.detach().cpu()),
        "loss_r3": float(loss_r3.detach().cpu()),
        "loss_r4": float(loss_r4.detach().cpu()),
        "loss_if12_a": float(loss_if12_a.cpu()),
        "loss_if12_qn": float(loss_if12_qn.cpu()),
        "loss_if23_a": float(loss_if23_a.cpu() + loss_ifgt_a.cpu() + loss_ifgb_a.cpu()),
        "loss_if23_qn": float(loss_if23_qn.cpu() + loss_ifgt_qn.cpu() + loss_ifgb_qn.cpu()),
        "loss_if34_a": float(loss_if34_a.cpu() + loss_if24_a.cpu()),
        "loss_if34_qn": float(loss_if34_qn.cpu() + loss_if24_qn.cpu()),
    }
    return loss, info


# =========================================================
# 9) Training / 训练
# =========================================================
def train_model(model: MultiDomainPINN2DGap, phys: PhysParams, train: TrainParams, device, dtype):
    hist = {
        "total": [], "pde": [], "interface": [], "sup_A": [], "sup_B": [],
        "r1": [], "r2": [], "r3": [], "r4": [],
        "if12_a": [], "if12_qn": [], "if23_a": [], "if23_qn": [], "if34_a": [], "if34_qn": [],
    }

    optimizer = torch.optim.Adam(model.parameters(), lr=train.lr_adam)
    model.train()
    t0 = time.time()

    colloc = build_collocation(phys, train, device, dtype)
    for ep in range(1, train.adam_epochs + 1):
        if ep > 1 and (ep % train.resample_every == 1):
            colloc = build_collocation(phys, train, device, dtype)

        optimizer.zero_grad()
        loss, info = compute_loss(model, colloc, train)
        loss.backward(retain_graph=True)
        optimizer.step()

        for k_hist, k_info in [
            ("total", "loss"), ("pde", "loss_pde"), ("interface", "loss_if"),
            ("sup_A", "loss_sup_A"), ("sup_B", "loss_sup_B"),
            ("r1", "loss_r1"), ("r2", "loss_r2"), ("r3", "loss_r3"), ("r4", "loss_r4"),
            ("if12_a", "loss_if12_a"), ("if12_qn", "loss_if12_qn"),
            ("if23_a", "loss_if23_a"), ("if23_qn", "loss_if23_qn"),
            ("if34_a", "loss_if34_a"), ("if34_qn", "loss_if34_qn"),
        ]:
            hist[k_hist].append(info[k_info])

        if ep == 1 or ep % train.print_every == 0:
            print(
                f"[Adam] {ep:5d}/{train.adam_epochs:5d} | "
                f"Total={info['loss']:.3e} | PDE={info['loss_pde']:.3e} | IF={info['loss_if']:.3e}"
            )

    print(f"Adam finished in {time.time() - t0:.2f} s")

    if train.lbfgs_steps > 0:
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
            loss, info = compute_loss(model, fixed, train)
            loss.backward()
            counter["k"] += 1

            for k_hist, k_info in [
                ("total", "loss"), ("pde", "loss_pde"), ("interface", "loss_if"),
                ("sup_A", "loss_sup_A"), ("sup_B", "loss_sup_B"),
                ("r1", "loss_r1"), ("r2", "loss_r2"), ("r3", "loss_r3"), ("r4", "loss_r4"),
                ("if12_a", "loss_if12_a"), ("if12_qn", "loss_if12_qn"),
                ("if23_a", "loss_if23_a"), ("if23_qn", "loss_if23_qn"),
                ("if34_a", "loss_if34_a"), ("if34_qn", "loss_if34_qn"),
            ]:
                hist[k_hist].append(info[k_info])

            if counter["k"] == 1 or counter["k"] % 50 == 0:
                print(
                    f"[LBFGS] {counter['k']:4d}/{train.lbfgs_steps:4d} | "
                    f"Total={info['loss']:.3e} | PDE={info['loss_pde']:.3e} | IF={info['loss_if']:.3e}"
                )
            return loss

        lbfgs.step(closure)
        print(f"LBFGS finished in {time.time() - t1:.2f} s")

    return hist


# =========================================================
# 10) Post-processing / 后处理
# =========================================================
def eval_on_grid(model: MultiDomainPINN2DGap, phys: PhysParams, n_plot: int, device, dtype):
    x = np.linspace(-1.0, 1.0, n_plot)
    y = np.linspace(-1.0, 1.0, n_plot)
    XX, YY = np.meshgrid(x, y, indexing="xy")
    RR = np.sqrt(XX ** 2 + YY ** 2)
    inside = RR <= 1.0

    xy_np = np.column_stack([XX[inside], YY[inside]])
    xy = torch.tensor(xy_np, device=device, dtype=dtype)

    model.eval()
    with torch.no_grad():
        a, qx, qy = predict_piecewise(model, xy)
        rho = np.sqrt(xy_np[:, 0] ** 2 + xy_np[:, 1] ** 2)
        mur = np.ones((xy_np.shape[0], 1), dtype=np.float64)
        gap_mask = (
            (xy_np[:, 0] >= phys.rho_fe_in)
            & (xy_np[:, 0] <= phys.rho_fe_out)
            & (np.abs(xy_np[:, 1]) <= phys.gap_half_rho)
        )
        m3 = (rho > phys.rho_fe_in) & (rho <= phys.rho_fe_out) & (~gap_mask)
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


def eval_xaxis_line(model: MultiDomainPINN2DGap, phys: PhysParams, device, dtype, n_line: int = 1400):
    x = np.linspace(-1.0, 1.0, n_line)
    y = np.full_like(x, 1.0e-12)
    xy = torch.tensor(np.column_stack([x, y]), device=device, dtype=dtype)

    model.eval()
    with torch.no_grad():
        a, qx, qy = predict_piecewise(model, xy)
        gap_mask = (
            (x >= phys.rho_fe_in)
            & (x <= phys.rho_fe_out)
            & (np.abs(y) <= phys.gap_half_rho)
        )
        mur = np.ones((len(x), 1), dtype=np.float64)
        m3 = (x > phys.rho_fe_in) & (x <= phys.rho_fe_out) & (~gap_mask)
        mur[m3, :] = phys.mur_fe
        mur_t = torch.tensor(mur, device=device, dtype=dtype)
        bx = mur_t * qy
        by = -mur_t * qx
        bmag = torch.sqrt(bx ** 2 + by ** 2)

    return x * phys.R * 100.0, phys.A0 * a.cpu().numpy().reshape(-1), phys.B0 * bmag.cpu().numpy().reshape(-1)


def save_plots(Xcm, Ycm, A_grid, Bmag_grid, hist, phys: PhysParams, model: MultiDomainPINN2DGap, save_dir: str, device, dtype):
    os.makedirs(save_dir, exist_ok=True)

    # A contour
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    cf1 = ax1.contourf(Xcm, Ycm, A_grid, levels=40)
    ax1.contour(Xcm, Ycm, A_grid, levels=12, colors="k", linewidths=0.2)
    ax1.set_aspect("equal")
    ax1.set_xlabel("x / cm")
    ax1.set_ylabel("y / cm")
    ax1.set_title(r"PINN: Magnetic Vector Potential $A(x,y)$ with air gap")
    # draw the gap box / 标出间隙位置
    rect = plt.Rectangle((phys.r_fe_in_cm, -0.5 * phys.gap_width_cm),
                         phys.r_fe_out_cm - phys.r_fe_in_cm,
                         phys.gap_width_cm,
                         fill=False, color="white", linewidth=1.0, linestyle="--")
    ax1.add_patch(rect)
    fig1.colorbar(cf1, ax=ax1, label="A (Wb/m)")
    fig1.tight_layout()
    p1 = os.path.join(save_dir, "airgap_A_contour_2D.png")
    fig1.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close(fig1)

    # |B| contour
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    cf2 = ax2.contourf(Xcm, Ycm, Bmag_grid, levels=40)
    ax2.set_aspect("equal")
    ax2.set_xlabel("x / cm")
    ax2.set_ylabel("y / cm")
    ax2.set_title(r"PINN: Magnetic Flux Density Magnitude $|\mathbf{B}|$ with air gap")
    rect2 = plt.Rectangle((phys.r_fe_in_cm, -0.5 * phys.gap_width_cm),
                          phys.r_fe_out_cm - phys.r_fe_in_cm,
                          phys.gap_width_cm,
                          fill=False, color="white", linewidth=1.0, linestyle="--")
    ax2.add_patch(rect2)
    fig2.colorbar(cf2, ax=ax2, label=r"$|\mathbf{B}|$ (T)")
    fig2.tight_layout()
    p2 = os.path.join(save_dir, "airgap_Bmag_2D.png")
    fig2.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close(fig2)

    # x-axis line cut
    x_cm, A_line, B_line = eval_xaxis_line(model, phys, device, dtype, n_line=1400)
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax3a.plot(x_cm, A_line, linewidth=2.0)
    ax3a.axvspan(phys.r_fe_in_cm, phys.r_fe_out_cm, color="gray", alpha=0.15, label="ferromagnetic region")
    ax3a.set_xlabel("x / cm (y≈0)")
    ax3a.set_ylabel("A (Wb/m)")
    ax3a.set_title("Line cut along x-axis through the air gap (-30cm to 30cm)")
    ax3a.grid(True, alpha=0.3)

    ax3b.plot(x_cm, B_line, linewidth=2.0)
    ax3b.axvspan(phys.r_fe_in_cm, phys.r_fe_out_cm, color="gray", alpha=0.15, label="ferromagnetic region")
    ax3b.set_xlabel("x / cm (y≈0)")
    ax3b.set_ylabel(r"$|\mathbf{B}|$ (T)")
    ax3b.set_title("|B| along x-axis through the air gap (-30cm to 30cm)")
    ax3b.grid(True, alpha=0.3)

    fig3.tight_layout()
    p3 = os.path.join(save_dir, "airgap_xaxis_linecut.png")
    fig3.savefig(p3, dpi=300, bbox_inches="tight")
    plt.close(fig3)

    # Loss history
    fig4, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ax.semilogy(hist["total"], label="total")
    ax.semilogy(hist["pde"], label="pde")
    ax.semilogy(hist["interface"], label="interface")
    ax.set_title("Total loss history")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.semilogy(hist["r1"], label="coil")
    ax.semilogy(hist["r2"], label="air + gap")
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
    ax.semilogy(hist["if23_a"], label="air-ferro A")
    ax.semilogy(hist["if23_qn"], label="air-ferro q·n")
    ax.set_title("Interface losses (1)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)

    ax = axes[1, 1]
    ax.semilogy(hist["if34_a"], label="outer-side A")
    ax.semilogy(hist["if34_qn"], label="outer-side q·n")
    ax.set_title("Interface losses (2)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig4.tight_layout()
    p4 = os.path.join(save_dir, "airgap_loss_history_2D.png")
    fig4.savefig(p4, dpi=300, bbox_inches="tight")
    plt.close(fig4)

    return p1, p2, p3, p4


# =========================================================
# 11) Checkpoint utility / 断点续训工具
# =========================================================
def try_resume_model(model: nn.Module, ckpt_path: str, device, strict: bool = True):
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
    print("=" * 78)
    return True, ckpt


# =========================================================
# 12) Main / 主程序
# =========================================================
def main():
    set_seed(42)
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    phys = PhysParams()
    train = TrainParams()

    print("=" * 78)
    print("PyTorch 2D multi-domain PINN for Model 3 with a narrow air gap")
    print(f"Device = {device}")
    print(f"Gap width = {phys.gap_width_cm:.3f} cm at +x side")
    print(f"A0 = {phys.A0:.6e} Wb/m, q0 = {phys.q0:.6e} A/m, B0 = {phys.B0:.6e} T")
    print("=" * 78)

    base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

    model = MultiDomainPINN2DGap(phys, train).to(device=device, dtype=dtype)

    if train.resume_from_checkpoint:
        ckpt_path = os.path.join(base_dir, train.checkpoint_relpath)
        try_resume_model(model, ckpt_path, device=device, strict=train.strict_resume)

    hist = train_model(model, phys, train, device, dtype)

    save_ckpt = os.path.join(base_dir, "model3_pinn_2d_airgap_checkpoint.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "phys_params": vars(phys),
        "train_params": vars(train),
        "dtype": str(dtype),
        "device": str(device),
    }, save_ckpt)
    print(f"Saved checkpoint: {save_ckpt}")

    Xcm, Ycm, A_grid, Bmag_grid = eval_on_grid(model, phys, train.n_plot, device, dtype)
    p1, p2, p3, p4 = save_plots(Xcm, Ycm, A_grid, Bmag_grid, hist, phys, model, base_dir, device, dtype)

    x_cm, A_line, B_line = eval_xaxis_line(model, phys, device, dtype, n_line=train.n_line)
    print("\n" + "=" * 78)
    print("2D PINN with air gap finished")
    print(f"Saved: {p1}")
    print(f"Saved: {p2}")
    print(f"Saved: {p3}")
    print(f"Saved: {p4}")
    print("=" * 78)


if __name__ == "__main__":
    main()
