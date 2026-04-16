# -*- coding: utf-8 -*-
"""
Model 3 : 1D axisymmetric magnetostatic PINN (mixed first-order form, PyTorch)

Original PDE / 原始方程:
    -(1/r) d/dr [ r * (1/mu(r)) * dA/dr ] = J(r)

Introduce auxiliary variable / 引入辅助变量:
    q(r) = r/mu(r) * dA/dr

Then the PDE becomes the first-order system / 则方程变为一阶系统:
    dA/dr - mu(r)/r * q = 0
    dq/dr + r J(r) = 0

Dimensionless variables / 无量纲变量:
    x = r / R
    A = A0 * a(x),   A0 = mu0 * J0 * R^2
    q = q0 * qh(x),  q0 = J0 * R^2

Dimensionless first-order system / 无量纲一阶系统:
    da/dx - mu_r(x)/x * qh = 0
    dqh/dx + x * Jhat(x) = 0

Boundary and interface conditions / 边界与界面条件:
    qh(eps) = 0
    a(1) = 0
    a^- = a^+
    qh^- = qh^+

Post-processing / 后处理:
    B_theta = -dA/dr = -(A0/R) * da/dx = -mu0*J0*R * da/dx
"""

import os
import math
import time
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    """Set random seed / 设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class PhysParams:
    # Constants / 常数
    mu0: float = 4.0 * math.pi * 1.0e-7
    mur_fe: float = 1000.0

    # Geometry / 几何参数，单位 cm
    r_coil_cm: float = 5.0
    r_fe_in_cm: float = 10.0
    r_fe_out_cm: float = 10.5
    r_out_cm: float = 30.0

    # Source / 源项，单位 A/cm^2
    J0_cm2: float = 1.0

    # Small offset from axis / 轴线小偏移
    eps_r: float = 1.0e-6

    def __post_init__(self):
        cm_to_m = 1.0e-2
        self.mu_fe = self.mur_fe * self.mu0
        self.r_coil = self.r_coil_cm * cm_to_m
        self.r_fe_in = self.r_fe_in_cm * cm_to_m
        self.r_fe_out = self.r_fe_out_cm * cm_to_m
        self.R = self.r_out_cm * cm_to_m
        self.J0 = self.J0_cm2 * 1.0e4

        # Dimensionless interface locations / 无量纲分界位置
        self.x_eps = self.eps_r / self.R
        self.x_coil = self.r_coil / self.R
        self.x_fe_in = self.r_fe_in / self.R
        self.x_fe_out = self.r_fe_out / self.R

        # Characteristic scales / 特征尺度
        self.q0 = self.J0 * self.r_coil  ** 2
        self.A0 = self.mu0 * self.q0
        self.B0 = self.A0 / self.R


@dataclass
class TrainParams:
    # Network / 网络参数
    hidden_layers: int = 3
    hidden_units: int = 48

    # Collocation / 配点数
    N1: int = 256  # 区域1：线圈区 / coil
    N2: int = 256  # 区域2：内空气区 / inner air
    N3: int = 2000  # 区域3：铁磁薄层 / ferro thin layer  
    N4: int = 1000  # 区域4：外空气区 / outer air

    # Optimization / 优化参数
    adam_epochs: int = 12000
    lbfgs_steps: int = 50
    lr_adam: float = 1.0e-4
    print_every: int = 500

    # Loss weights / 损失权重
    w_pde: float = 1.0
    w_bc: float = 20.0
    w_if: float = 100.0
    w_axis: float = 20.0

    # FEM supervision / FEM监督参数
    use_fem_supervision: bool = True
    fem_csv_relpath: str = "fem_supervised_points_dense_5to10cm.csv"
    w_sup_A: float = 180.0   # A supervision weight / A监督权重
    w_sup_B: float = 60.0    # B supervision weight / B监督权重


class SubNet(nn.Module):
    """One subnetwork outputs two fields: a(x), qh(x). / 一个子网络输出两个场: a(x), qh(x)。"""
    def __init__(self, x_min, x_max, hidden_layers=3, hidden_units=48):
        super().__init__()
        self.x_min = float(x_min)
        self.x_max = float(x_max)

        layers = []
        in_dim = 1
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_units))
            layers.append(nn.Tanh())
            in_dim = hidden_units
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def normalize(self, x):
        return 2.0 * (x - self.x_min) / (self.x_max - self.x_min) - 1.0

    def forward_raw(self, x):
        return self.net(self.normalize(x))


class MixedPINN(nn.Module):
    """Four subnetworks for four subdomains / 四个子区域对应四个子网络"""
    def __init__(self, phys: PhysParams, train: TrainParams):
        super().__init__()
        self.phys = phys

        self.net1 = SubNet(phys.x_eps,   phys.x_coil,   train.hidden_layers, train.hidden_units)
        self.net2 = SubNet(phys.x_coil,  phys.x_fe_in,  train.hidden_layers, train.hidden_units)
        self.net3 = SubNet(phys.x_fe_in, phys.x_fe_out, train.hidden_layers, train.hidden_units)
        self.net4 = SubNet(phys.x_fe_out, 1.0,          train.hidden_layers, train.hidden_units)

        # =====================================================
        # Learnable constants for source-free regions
        # 无源区的可学习常量
        #
        # 物理意义：
        # 在 region 2/3/4 中，理论上 dqh/dx = 0，
        # 所以 qh 应接近常量，而不是任意自由函数。
        #
        # 初值取 -0.5：
        # 你的程序里 q_anchor 已经用了 target_q_coil = -0.5，
        # 用同样的量级作为 region 2/3/4 的初始常量最自然。
        # =====================================================
        self.c_out = nn.Parameter(torch.tensor([[-0.5]], dtype=torch.float64))

    def fields(self, subnet: SubNet, x, region_id: int):
        """
        Return dimensionless fields a(x), qh(x).
        返回无量纲场 a(x), qh(x)。

        Hard constraints / 硬约束:
        - region 1: qh(x_eps)=0 通过乘以 (x-x_eps)
        - region 4: a(1)=0       通过乘以 (1-x)
        - region 2/3/4: qh 采用“常量 + 小修正”形式
        """
        raw = subnet.forward_raw(x)
        a_raw = raw[:, 0:1]
        q_raw = raw[:, 1:2]

        if region_id == 1:
            a = a_raw
            qh = (x - self.phys.x_eps) * q_raw

        elif region_id == 2:
            # region 2: x_coil <= x <= x_fe_in
            a = a_raw
            xL = self.phys.x_coil
            xR = self.phys.x_fe_in

            # 常量 + 小修正
            # 在区间两端修正项为 0，中间允许小幅调整
            alpha2 = 0.05
            qh = self.c_out + alpha2 * (x - xL) * (xR - x) * q_raw

        elif region_id == 3:
            # region 3: x_fe_in <= x <= x_fe_out
            a = a_raw
            xL = self.phys.x_fe_in
            xR = self.phys.x_fe_out

            # 小修正系数：先从很小开始
            alpha3 = 0.02

            # 常量 + 小修正
            # 修正项在两端界面都为 0，
            # 因此不会破坏界面处 qh 的基本连续结构
            
            qh = self.c_out + alpha3 * (x - xL) * (xR - x) * q_raw

            # qh = self.c_out+ 0.0 * x

        elif region_id == 4:
            xL = self.phys.x_fe_out
            beta4 = 0.03
            a = self.c_out * torch.log(x) + beta4 * (1.0 - x) * (x - xL) * a_raw
            qh = self.c_out + 0.0 * x
            # a = self.c_out * torch.log(x)
            # qh = self.c_out + 0.0 * x
            

        else:
            raise ValueError(f"Unknown region_id = {region_id}")

        return a, qh


def grad(y, x):
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]


def sample_uniform(n, a, b, device, dtype):
    x = a + (b - a) * torch.rand(n, 1, device=device, dtype=dtype)
    x.requires_grad_(True)
    return x

def sample_edge_dense(n, a, b, device, dtype, power=0.35):
    """
    Sample points densely near both ends of [a, b].
    在区间 [a, b] 两端加密采样。

    Parameters
    ----------
    n : int
        number of points / 采样点数
    a, b : float
        interval bounds / 区间端点
    power : float
        smaller => denser near edges / 越小，两端越密

    Returns
    -------
    x : torch.Tensor, shape (n,1)
        collocation points / 配点
    """
    u = torch.rand(n, 1, device=device, dtype=dtype)

    # half points near left edge, half near right edge
    # 一半点靠近左端，一半点靠近右端
    n_left = n // 2
    n_right = n - n_left

    u_left = u[:n_left]
    u_right = u[n_right:]

    # near left edge / 左端加密
    x_left = a + (b - a) * (u_left ** power)

    # near right edge / 右端加密
    x_right = b - (b - a) * (u_right ** power)

    x = torch.cat([x_left, x_right], dim=0)

    # shuffle / 打乱顺序
    idx = torch.randperm(n, device=device)
    x = x[idx]

    x.requires_grad_(True)
    return x

def build_collocation(phys: PhysParams, train: TrainParams, device, dtype):
    return {
        "x1": sample_uniform(train.N1, phys.x_eps, phys.x_coil, device, dtype),
        "x2": sample_uniform(train.N2, phys.x_coil, phys.x_fe_in, device, dtype),
        # "x3": sample_uniform(train.N3, phys.x_fe_in, phys.x_fe_out, device, dtype),
        # ferro region: dense near both interfaces
        # 铁磁薄层：在两端界面附近加密
        "x3": sample_edge_dense(train.N3, phys.x_fe_in, phys.x_fe_out, device, dtype, power=0.35),
        "x4": sample_uniform(train.N4, phys.x_fe_out, 1.0, device, dtype),
    }

def load_fem_supervision(csv_path: str, phys: PhysParams, device, dtype):
    """
    Load FEM supervised data from CSV / 从 CSV 读取 FEM 监督点数据

    Expected columns / 期望列名:
        r_m, A_Wb_per_m, weight
    Optional columns / 可选列名:
        r_cm, Btheta_T, region_id
    """
    data = np.genfromtxt(csv_path, delimiter=",", names=True, encoding="utf-8")
    data = np.atleast_1d(data)

    names = data.dtype.names

    if "r_m" not in names:
        raise ValueError("CSV 缺少列: r_m")
    if "A_Wb_per_m" not in names:
        raise ValueError("CSV 缺少列: A_Wb_per_m")
    
    # -----------------------------
    # Read FEM raw data / 读取FEM原始数据
    # -----------------------------
    r_m = np.asarray(data["r_m"], dtype=np.float64).reshape(-1)
    A_fem = np.asarray(data["A_Wb_per_m"], dtype=np.float64).reshape(-1)

    if "Btheta_T" in names:
        B_fem = np.asarray(data["Btheta_T"], dtype=np.float64).reshape(-1)       # 单位 T / T
        has_B = True
    else:
        B_fem = np.zeros_like(r_m)
        has_B = False

    if "weight" in names:
        w = np.asarray(data["weight"], dtype=np.float64).reshape(-1)
    else:
        w = np.ones_like(r_m)

    # Normalize weights / 权重归一化，避免数值过大
    w = w / np.mean(w)

    # Convert to dimensionless variables / 转成无量纲变量
    x = r_m / phys.R
    a_fem = A_fem / phys.A0
    b_fem = B_fem / phys.B0


    # Important / 重要：
    # 当前网络第1区起点是 x_eps，不是严格 0
    # 因此把 x 夹到 [x_eps, 1] 更稳妥
    x = np.clip(x, phys.x_eps, 1.0)

    # -----------------------------
    # B-loss mask / B监督掩码
    #    轴线点不要直接参与 B 监督，
    #    因为后面要用 b = -mu_r * qh / x，x 太小时数值不稳
    # -----------------------------
    if has_B:
        w_b = w.copy()
        w_b[r_m <= 5.0 * phys.eps_r] = 0.0
    else:
        w_b = np.zeros_like(w)

    sup_data = {
        "x": torch.tensor(x[:, None], device=device, dtype=dtype),
        "a": torch.tensor(a_fem[:, None], device=device, dtype=dtype),
        "b": torch.tensor(b_fem[:, None], device=device, dtype=dtype),
        "w": torch.tensor(w[:, None], device=device, dtype=dtype),
        "w_b": torch.tensor(w_b[:, None],  device=device, dtype=dtype),
        "has_B": has_B,
    }
    return sup_data

def predict_piecewise_a_q(model: MixedPINN, x):
    """
    Predict piecewise fields a(x), qh(x) for arbitrary x in [x_eps, 1]
    对任意监督点 x，按所在区域调用对应子网络预测 a(x), qh(x)
    """
    phys = model.phys

    # 保证输入是列向量
    x_col = x.reshape(-1, 1)

    # 输出也保持列向量
    a_out = torch.zeros_like(x_col)
    q_out = torch.zeros_like(x_col)

    # 关键：mask 用一维
    x_flat = x_col[:, 0]

    m1 = (x_flat >= phys.x_eps) & (x_flat <= phys.x_coil)
    m2 = (x_flat > phys.x_coil) & (x_flat <= phys.x_fe_in)
    m3 = (x_flat > phys.x_fe_in) & (x_flat <= phys.x_fe_out)
    m4 = (x_flat > phys.x_fe_out) & (x_flat <= 1.0)

    if torch.any(m1):
        a1, q1 = model.fields(model.net1, x_col[m1], 1)
        a_out[m1, :] = a1
        q_out[m1, :] = q1

    if torch.any(m2):
        a2, q2 = model.fields(model.net2, x_col[m2], 2)
        a_out[m2, :] = a2
        q_out[m2, :] = q2

    if torch.any(m3):
        a3, q3 = model.fields(model.net3, x_col[m3], 3)
        a_out[m3, :] = a3
        q_out[m3, :] = q3

    if torch.any(m4):
        a4, q4 = model.fields(model.net4, x_col[m4], 4)
        a_out[m4, :] = a4
        q_out[m4, :] = q4

    return a_out, q_out

def piecewise_mu_r(x, phys: PhysParams):
    """
    Return piecewise relative permeability mu_r(x)
    返回分段相对磁导率 mu_r(x)
    """
    x_col = x.reshape(-1, 1)
    x_flat = x_col[:, 0]

    mu_r = torch.ones_like(x_col)

    # Region 3: ferro layer / 第3区：铁磁薄层
    m3 = (x_flat > phys.x_fe_in) & (x_flat <= phys.x_fe_out)
    mu_r[m3, :] = phys.mur_fe

    return mu_r

def pde_residual(model: MixedPINN, subnet: SubNet, x, mu_r, Jsrc_hat, region_id: int):
    """
    Dimensionless mixed residuals:
        da/dx - mu_r/x * qh = 0
        dqh/dx + x * Jsrc_hat = 0

    Here Jsrc_hat means the source coefficient after nondimensionalization.
    这里 Jsrc_hat 是无量纲化之后的源项系数，不再直接等于 J/J0。
    """
    a, qh = model.fields(subnet, x, region_id)
    da_dx = grad(a, x)
    dq_dx = grad(qh, x)

    res_a = da_dx - mu_r * qh / x
    res_q = dq_dx + x * Jsrc_hat
    return res_a, res_q, a, qh, da_dx


def interface_loss(model: MixedPINN, netL: SubNet, netR: SubNet, x_if, left_id, right_id, device, dtype):
    x = torch.tensor([[x_if]], device=device, dtype=dtype, requires_grad=True)
    aL, qL = model.fields(netL, x, left_id)
    aR, qR = model.fields(netR, x, right_id)
    loss_a = torch.mean((aL - aR) ** 2)
    loss_q = torch.mean((qL - qR) ** 2)
    return loss_a + loss_q, loss_a.detach(), loss_q.detach()


def compute_loss(model: MixedPINN, colloc, phys: PhysParams, train: TrainParams, device, dtype, sup_data=None):
    # PDE residuals / PDE 残差
    Jsrc_hat_coil = 1.0 / (phys.x_coil ** 2)
    res1_a, res1_q, _, _, _ = pde_residual(model, model.net1, colloc["x1"], 1.0, Jsrc_hat_coil, 1)
    res2_a, res2_q, _, _, _ = pde_residual(model, model.net2, colloc["x2"], 1.0, 0.0, 2)
    res3_a, res3_q, _, _, _ = pde_residual(model, model.net3, colloc["x3"], phys.mur_fe, 0.0, 3)
    res4_a, res4_q, _, _, _ = pde_residual(model, model.net4, colloc["x4"], 1.0, 0.0, 4)

    loss_res1_a = torch.mean(res1_a ** 2)
    loss_res1_q = torch.mean(res1_q ** 2)
    loss_res2_a = torch.mean(res2_a ** 2)
    loss_res2_q = torch.mean(res2_q ** 2)
    loss_res3_a = torch.mean(res3_a ** 2)
    loss_res3_q = torch.mean(res3_q ** 2)
    loss_res4_a = torch.mean(res4_a ** 2)
    loss_res4_q = torch.mean(res4_q ** 2)

    loss_pde = (
        0.5* loss_res1_a + 50.0* loss_res1_q
     + 1 * loss_res2_a + 10.0 * loss_res2_q
     + 5.0* loss_res3_a + 20.0 * loss_res3_q
     + 10.0 * loss_res4_a + 10.0 * loss_res4_q
    )

    # Outer boundary a(1)=0 / 外边界 a(1)=0
    xR = torch.tensor([[1.0]], device=device, dtype=dtype, requires_grad=True)
    aR, _ = model.fields(model.net4, xR, 4)
    loss_bc = torch.mean(aR ** 2)

    # Axis regularity qh(x_eps)=0 / 轴线正则 qh(x_eps)=0
    x0 = torch.tensor([[phys.x_eps]], device=device, dtype=dtype, requires_grad=True)
    _, q0 = model.fields(model.net1, x0, 1)
    loss_axis = torch.mean(q0 ** 2)

    # Interface continuity / 界面连续
    loss_if12, loss_if12_a, loss_if12_q = interface_loss(model, model.net1, model.net2, phys.x_coil, 1, 2, device, dtype)
    loss_if23, loss_if23_a, loss_if23_q = interface_loss(model, model.net2, model.net3, phys.x_fe_in, 2, 3, device, dtype)
    loss_if34, loss_if34_a, loss_if34_q = interface_loss(model, model.net3, model.net4, phys.x_fe_out, 3, 4, device, dtype)

    loss_if = loss_if12 + loss_if23 +  loss_if34


    # FEM supervised loss / FEM监督损失
    loss_sup_A = torch.tensor(0.0, device=device, dtype=dtype)
    loss_sup_B = torch.tensor(0.0, device=device, dtype=dtype)


    if (sup_data is not None) and train.use_fem_supervision:
        x_sup = sup_data["x"]          # 无量纲半径 / dimensionless radius
        a_sup = sup_data["a"]          # 无量纲A / dimensionless A
        b_sup = sup_data["b"]          # 无量纲B / dimensionless B
        w_sup = sup_data["w"]          # A监督权重 / A weights
        w_sup_B = sup_data["w_b"]      # B监督权重 / B weights

        # Predict a(x), qh(x) / 预测 a(x), qh(x)
        a_pred_sup, q_pred_sup = predict_piecewise_a_q(model, x_sup)

        # Weighted MSE / 加权均方误差
        # A supervision / A监督
        loss_sup_A = torch.mean(w_sup * (a_pred_sup - a_sup) ** 2)

        # B supervision / B监督
        # b = B/B0 = - mu_r * qh / x
        if sup_data["has_B"]:
            x_safe = torch.clamp(x_sup, min=phys.x_eps)
            mu_r_sup = piecewise_mu_r(x_sup, phys)
            b_pred_sup = -mu_r_sup * q_pred_sup / x_safe

            loss_sup_B = torch.mean(w_sup_B * (b_pred_sup - b_sup) ** 2)

    loss =(
        train.w_pde * loss_pde 
        + train.w_bc * loss_bc
        + train.w_if * loss_if
        + train.w_axis * loss_axis
        + train.w_sup_A * loss_sup_A
        + train.w_sup_B * loss_sup_B
    ) 
    

    info = {
        "loss": loss.detach().item(),
        "loss_pde": loss_pde.detach().item(),
        "loss_bc": loss_bc.detach().item(),
        "loss_if": loss_if.detach().item(),
        "loss_axis": loss_axis.detach().item(),
        "loss_sup_A": loss_sup_A.detach().item(),
        "loss_sup_B": loss_sup_B.detach().item(),

        # interface losses / 界面损失
        "loss_if12_a": loss_if12_a.detach().item(),
        "loss_if12_q": loss_if12_q.detach().item(),
        "loss_if23_a": loss_if23_a.detach().item(),
        "loss_if23_q": loss_if23_q.detach().item(),
        "loss_if34_a": loss_if34_a.detach().item(),
        "loss_if34_q": loss_if34_q.detach().item(),

        # PDE component losses / PDE 分项损失
        "loss_res1_a": loss_res1_a.detach().item(),
        "loss_res1_q": loss_res1_q.detach().item(),
        "loss_res2_a": loss_res2_a.detach().item(),
        "loss_res2_q": loss_res2_q.detach().item(),
        "loss_res3_a": loss_res3_a.detach().item(),
        "loss_res3_q": loss_res3_q.detach().item(),
        "loss_res4_a": loss_res4_a.detach().item(),
        "loss_res4_q": loss_res4_q.detach().item(),
    }
    
    return loss, info


def train_model(model, phys: PhysParams, train: TrainParams, device, dtype, sup_data=None):
    model.train()
    hist = {
        # total losses / 总损失
        "total": [], "pde": [], "bc": [], "interface": [], "axis": [], 
        "sup_A": [],"sup_B": [],
        # interface component losses / 界面分项损失
        "if12_a": [], "if12_q": [], "if23_a": [], "if23_q": [], "if34_a": [], "if34_q": [],
        # PDE component losses / PDE 分项损失
        "res1_a": [], "res1_q": [], "res2_a": [], "res2_q": [], "res3_a": [], "res3_q": [], "res4_a": [], "res4_q": [],        
    }

    opt = torch.optim.Adam(model.parameters(), lr=train.lr_adam)
    t0 = time.time()

    # 每 2000 次 Adam 重新采样一次/Adam resamples every 2000 times.
    refresh_every = 2000#Adam resamples
    # 先固定采样，为了减少损失里的周期性尖峰
    colloc_adam = build_collocation(phys, train, device, dtype)
    
    for ep in range(1, train.adam_epochs + 1):
        if ep % refresh_every == 1 and ep > 1:#Adam resamples
            colloc_adam = build_collocation(phys, train, device, dtype)#Adam resamples

        colloc = colloc_adam

        opt.zero_grad()
        loss, info = compute_loss(model, colloc, phys, train, device, dtype, sup_data=sup_data)
        loss.backward()
        opt.step()

        hist["total"].append(info["loss"])
        hist["pde"].append(info["loss_pde"])
        hist["bc"].append(info["loss_bc"])
        hist["interface"].append(info["loss_if"])
        hist["axis"].append(info["loss_axis"])
        hist["sup_A"].append(info["loss_sup_A"])
        hist["sup_B"].append(info["loss_sup_B"])        

        # interface component losses / 界面分项损失
        hist["if12_a"].append(info["loss_if12_a"])
        hist["if12_q"].append(info["loss_if12_q"])
        hist["if23_a"].append(info["loss_if23_a"])
        hist["if23_q"].append(info["loss_if23_q"])
        hist["if34_a"].append(info["loss_if34_a"])
        hist["if34_q"].append(info["loss_if34_q"])

        # PDE component losses / PDE 分项损失
        hist["res1_a"].append(info["loss_res1_a"])
        hist["res1_q"].append(info["loss_res1_q"])
        hist["res2_a"].append(info["loss_res2_a"])
        hist["res2_q"].append(info["loss_res2_q"])
        hist["res3_a"].append(info["loss_res3_a"])
        hist["res3_q"].append(info["loss_res3_q"])
        hist["res4_a"].append(info["loss_res4_a"])
        hist["res4_q"].append(info["loss_res4_q"])        
            
        if ep == 1 or ep % train.print_every == 0:
            print(
                f"[Adam] {ep:5d}/{train.adam_epochs:5d} | "
                f"Total={info['loss']:.3e} | PDE={info['loss_pde']:.3e} | "
                f"SUP={info['loss_sup_A']:.3e} | SUP-B={info['loss_sup_B']:.3e} | "
                f"BC={info['loss_bc']:.3e} | IF={info['loss_if']:.3e} | AX={info['loss_axis']:.3e}"
            )
    print(f"Adam finished in {time.time()-t0:.2f} s")

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
        it = {"k": 0}
        t1 = time.time()

        def closure():
            lbfgs.zero_grad()
            loss, info = compute_loss(model, fixed, phys, train, device, dtype, sup_data=sup_data)
            loss.backward()
            it["k"] += 1

            # total losses / 总损失
            hist["total"].append(info["loss"])
            hist["pde"].append(info["loss_pde"])
            hist["bc"].append(info["loss_bc"])
            hist["interface"].append(info["loss_if"])
            hist["axis"].append(info["loss_axis"])
            hist["sup_A"].append(info["loss_sup_A"])
            hist["sup_B"].append(info["loss_sup_B"])

            # interface component losses / 界面分项损失
            hist["if12_a"].append(info["loss_if12_a"])
            hist["if12_q"].append(info["loss_if12_q"])
            hist["if23_a"].append(info["loss_if23_a"])
            hist["if23_q"].append(info["loss_if23_q"])
            hist["if34_a"].append(info["loss_if34_a"])
            hist["if34_q"].append(info["loss_if34_q"])

            # PDE component losses / PDE 分项损失
            hist["res1_a"].append(info["loss_res1_a"])
            hist["res1_q"].append(info["loss_res1_q"])
            hist["res2_a"].append(info["loss_res2_a"])
            hist["res2_q"].append(info["loss_res2_q"])
            hist["res3_a"].append(info["loss_res3_a"])
            hist["res3_q"].append(info["loss_res3_q"])
            hist["res4_a"].append(info["loss_res4_a"])
            hist["res4_q"].append(info["loss_res4_q"])

            if it["k"] == 1 or it["k"] % 100 == 0:
                print(
                    f"[LBFGS] {it['k']:5d}/{train.lbfgs_steps:5d} | "
                    f"Total={info['loss']:.3e} | PDE={info['loss_pde']:.3e} | "
                    f"SUP={info['loss_sup_A']:.3e} | SUP-B={info['loss_sup_B']:.3e} |"
                    f"BC={info['loss_bc']:.3e} | IF={info['loss_if']:.3e} | AX={info['loss_axis']:.3e}"
                )
            return loss

        lbfgs.step(closure)
        print(f"LBFGS finished in {time.time()-t1:.2f} s")
    return hist


def eval_subdomain(model: MixedPINN, subnet: SubNet, x_np, region_id: int, device, dtype):
    x = torch.tensor(x_np.reshape(-1, 1), device=device, dtype=dtype, requires_grad=True)
    a, qh = model.fields(subnet, x, region_id)
    da_dx = grad(a, x)
    return (
        x.detach().cpu().numpy().squeeze(),
        a.detach().cpu().numpy().squeeze(),
        qh.detach().cpu().numpy().squeeze(),
        da_dx.detach().cpu().numpy().squeeze(),
    )


def stitch_prediction(model, phys: PhysParams, device, dtype):
    n1, n2, n3, n4 = 300, 300, 180, 400
    x1 = np.linspace(phys.x_eps, phys.x_coil, n1)
    x2 = np.linspace(phys.x_coil, phys.x_fe_in, n2)
    x3 = np.linspace(phys.x_fe_in, phys.x_fe_out, n3)
    x4 = np.linspace(phys.x_fe_out, 1.0, n4)

    xx1, a1, q1, da1 = eval_subdomain(model, model.net1, x1, 1, device, dtype)
    xx2, a2, q2, da2 = eval_subdomain(model, model.net2, x2, 2, device, dtype)
    xx3, a3, q3, da3 = eval_subdomain(model, model.net3, x3, 3, device, dtype)
    xx4, a4, q4, da4 = eval_subdomain(model, model.net4, x4, 4, device, dtype)

    x_all = np.concatenate(([0.0], xx1[1:], xx2[1:], xx3[1:], xx4[1:]))
    a_all = np.concatenate(([a1[0]], a1[1:], a2[1:], a3[1:], a4[1:]))
    da_all = np.concatenate(([0.0], da1[1:], da2[1:], da3[1:], da4[1:]))
    q_all = np.concatenate(([0.0], q1[1:], q2[1:], q3[1:], q4[1:]))

    r_all = phys.R * x_all
    A_all = phys.A0 * a_all
    B_all = -phys.B0 * da_all
    q_phys = phys.q0 * q_all
    return r_all, A_all, B_all, q_phys

def exact_solution(r, mu0, mu_fe, J0, r_coil, r_fe_in, r_fe_out, R):
    """
    Piecewise analytical solution / 分段解析解

    Input:
        r       : radius array, unit m / 半径数组，单位 m
        mu0     : vacuum permeability / 真空磁导率
        mu_fe   : ferro permeability / 铁磁磁导率
        J0      : source current density, unit A/m^2 / 电流密度
        r_coil  : coil radius / 线圈半径
        r_fe_in : ferro inner radius / 铁磁内半径
        r_fe_out: ferro outer radius / 铁磁外半径
        R       : outer radius / 外边界半径

    Output:
        A_exact : analytical magnetic vector potential / 解析磁矢势
        B_exact : analytical magnetic flux density / 解析磁感应强度
    """
    r = np.asarray(r, dtype=float)
    A = np.zeros_like(r)
    B = np.zeros_like(r)

    # Constant from flux continuity / 由通量连续得到的常数
    Q = -0.5 * J0 * r_coil**2

    # Region 4 / 第4区: r_fe_out < r <= R
    def A4(rr):
        return mu0 * Q * np.log(rr / R)

    # Region 3 / 第3区: r_fe_in < r <= r_fe_out
    A4_rfo = A4(r_fe_out)

    def A3(rr):
        return A4_rfo + mu_fe * Q * np.log(rr / r_fe_out)

    # Region 2 / 第2区: r_coil < r <= r_fe_in
    A3_rfi = A3(r_fe_in)

    def A2(rr):
        return A3_rfi + mu0 * Q * np.log(rr / r_fe_in)

    # Region 1 / 第1区: 0 <= r <= r_coil
    A2_rc = A2(r_coil)
    C1 = A2_rc + mu0 * J0 * r_coil**2 / 4.0

    def A1(rr):
        return C1 - mu0 * J0 * rr**2 / 4.0

    # Piecewise assemble / 分段组装
    m1 = (r <= r_coil)
    m2 = (r > r_coil) & (r <= r_fe_in)
    m3 = (r > r_fe_in) & (r <= r_fe_out)
    m4 = (r > r_fe_out)

    A[m1] = A1(r[m1])
    A[m2] = A2(r[m2])
    A[m3] = A3(r[m3])
    A[m4] = A4(r[m4])

    # B_theta = -dA/dr / 磁感应强度
    B[m1] = mu0 * J0 * r[m1] / 2.0
    B[m2] = mu0 * J0 * r_coil**2 / (2.0 * r[m2])
    B[m3] = mu_fe * J0 * r_coil**2 / (2.0 * r[m3])
    B[m4] = mu0 * J0 * r_coil**2 / (2.0 * r[m4])

    # At r = 0, set B = 0 explicitly / 在 r=0 处显式赋值 B=0
    if np.any(r == 0.0):
        B[r == 0.0] = 0.0

    return A, B


def exact_solution_from_phys(r, phys: PhysParams):
    """
    Wrapper using PhysParams / 使用 PhysParams 的封装函数
    """
    return exact_solution(
        r=r,
        mu0=phys.mu0,
        mu_fe=phys.mu_fe,
        J0=phys.J0,
        r_coil=phys.r_coil,
        r_fe_in=phys.r_fe_in,
        r_fe_out=phys.r_fe_out,
        R=phys.R,
    )

def save_plots(r_all, A_all, B_all, hist, phys: PhysParams, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    # Analytical reference / 解析解参考曲线
    r_ref = np.linspace(0.0, phys.R, 2000)
    A_ref, B_ref = exact_solution_from_phys(r_ref, phys)
  
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(r_all * 100.0, A_all, linewidth=2.0, label="PINN")
    ax1.plot(r_ref * 100.0, A_ref, "--", linewidth=1.6, label="Analytical")
    ax1.axvline(phys.r_coil_cm, linestyle="--", linewidth=1.2, color="gray")
    ax1.axvline(phys.r_fe_in_cm, linestyle="--", linewidth=1.2, color="gray")
    ax1.axvline(phys.r_fe_out_cm, linestyle="--", linewidth=1.2, color="gray")
    ax1.set_xlabel("r / cm")
    ax1.set_ylabel("A(r) / Wb/m")
    ax1.set_title("1D axisymmetric mixed PINN solution of magnetic vector potential")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig1.tight_layout()
    p1 = os.path.join(save_dir, "A_1D_axisymmetric_mixed_PINN.png")
    fig1.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.plot(r_all * 100.0, B_all, linewidth=2.0, label="PINN")
    ax2.plot(r_ref * 100.0, B_ref, "--", linewidth=1.6, label="Analytical")
    ax2.axvline(phys.r_coil_cm, linestyle="--", linewidth=1.2, color="gray")
    ax2.axvline(phys.r_fe_in_cm, linestyle="--", linewidth=1.2, color="gray")
    ax2.axvline(phys.r_fe_out_cm, linestyle="--", linewidth=1.2, color="gray")
    ax2.set_xlabel("r / cm")
    ax2.set_ylabel(r"$B_\theta(r)$ / T")
    ax2.set_title("1D axisymmetric mixed PINN solution of magnetic flux density")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    p2 = os.path.join(save_dir, "Btheta_1D_axisymmetric_mixed_PINN.png")
    fig2.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(9, 5))
    ax3.semilogy(hist["total"], label="Total", linewidth=1.5)
    ax3.semilogy(hist["pde"], label="PDE", linewidth=1.2)
    ax3.semilogy(hist["bc"], label="BC", linewidth=1.2)
    ax3.semilogy(hist["interface"], label="Interface", linewidth=1.2)
    ax3.semilogy(hist["axis"], label="Axis", linewidth=1.2)    
    ax3.semilogy(hist["sup_A"], label="SUP-A", linewidth=1.2)
    ax3.semilogy(hist["sup_B"], label="SUP-B", linewidth=1.2)
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Loss")
    ax3.set_title("Training history of mixed multi-domain PINN")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    fig3.tight_layout()
    p3 = os.path.join(save_dir, "loss_history_mixed_PINN.png")
    fig3.savefig(p3, dpi=300, bbox_inches="tight")
    plt.close(fig3)

    # PDE component loss history / PDE 分项损失历史
    fig4, ax4 = plt.subplots(figsize=(10, 6))

    ax4.semilogy(hist["res1_a"], label="res1_a", linewidth=1.2)
    ax4.semilogy(hist["res1_q"], label="res1_q", linewidth=1.2)
    ax4.semilogy(hist["res2_a"], label="res2_a", linewidth=1.2)
    ax4.semilogy(hist["res2_q"], label="res2_q", linewidth=1.2)
    ax4.semilogy(hist["res3_a"], label="res3_a", linewidth=1.2)
    ax4.semilogy(hist["res3_q"], label="res3_q", linewidth=1.2)
    ax4.semilogy(hist["res4_a"], label="res4_a", linewidth=1.2)
    ax4.semilogy(hist["res4_q"], label="res4_q", linewidth=1.2)

    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Loss")
    ax4.set_title("PDE residual component history")
    ax4.grid(True, alpha=0.3)
    ax4.legend(ncol=2)

    fig4.tight_layout()
    p4 = os.path.join(save_dir, "loss_history_pde_components.png")
    fig4.savefig(p4, dpi=300, bbox_inches="tight")
    plt.close(fig4)

    return p1, p2, p3, p4


def main():
    set_seed(42)
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    phys = PhysParams()
    train = TrainParams()

    print("=" * 72)
    print("PyTorch mixed PINN for 1D axisymmetric magnetostatics")
    print(f"Device: {device}")
    print(f"A0 = {phys.A0:.6e} Wb/m, q0 = {phys.q0:.6e} A, B0 = {phys.B0:.6e} T")
    print("=" * 72)

    # Load FEM supervision / 读取 FEM 监督点
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fem_csv_path = os.path.join(base_dir, train.fem_csv_relpath)

    sup_data = None
    if train.use_fem_supervision:
        sup_data = load_fem_supervision(fem_csv_path, phys, device, dtype)
        print(f"Loaded FEM supervised points: {sup_data['x'].shape[0]}")
        print(f"FEM CSV path: {fem_csv_path}")

    model = MixedPINN(phys, train).to(device=device, dtype=dtype)
    hist = train_model(model, phys, train, device, dtype, sup_data=sup_data)

    # Save trained model / 保存训练好的模型
    ckpt_path = os.path.join(base_dir, "mixed_pinn_final_checkpoint.pth")

    torch.save({
        "model_state_dict": model.state_dict(),
        "phys_params": vars(phys),
        "train_params": vars(train),
        "dtype": str(dtype),
        "device": str(device),
    }, ckpt_path)

    print(f"Saved model checkpoint: {ckpt_path}")

    r_all, A_all, B_all, _ = stitch_prediction(model, phys, device, dtype)
    A_exact_all, B_exact_all = exact_solution_from_phys(r_all, phys)

    save_dir = os.path.dirname(os.path.abspath(__file__))
    p1, p2, p3, p4 = save_plots(r_all, A_all, B_all, hist, phys, save_dir)
   
    print("\n" + "=" * 72)
    print("Mixed PINN solve finished")    
    print(f"Max(A) PINN         = {np.max(A_all):.6e} Wb/m")
    print(f"Max(A) Analytical   = {np.max(A_exact_all):.6e} Wb/m")
    print(f"Max(|B|) PINN       = {np.max(np.abs(B_all)):.6e} T")
    print(f"Max(|B|) Analytical = {np.max(np.abs(B_exact_all)):.6e} T")

    print(f"Saved: {p1}")
    print(f"Saved: {p2}")
    print(f"Saved: {p3}")
    print(f"Saved: {p4}")
    print("=" * 72)


if __name__ == "__main__":
    main()
