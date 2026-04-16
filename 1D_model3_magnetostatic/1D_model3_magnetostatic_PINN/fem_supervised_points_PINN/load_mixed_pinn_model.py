import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from dataclasses import fields

from model3_pinn_fem_supervised import (
    PhysParams,
    TrainParams,
    MixedPINN,
    stitch_prediction,
    exact_solution_from_phys,
)

def load_trained_model(ckpt_path: str = "mixed_pinn_final_checkpoint.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    ckpt = torch.load(ckpt_path, map_location=device)

    phys_field_names = {f.name for f in fields(PhysParams)}
    train_field_names = {f.name for f in fields(TrainParams)}

    phys_kwargs = {k: v for k, v in ckpt["phys_params"].items() if k in phys_field_names}
    train_kwargs = {k: v for k, v in ckpt["train_params"].items() if k in train_field_names}

    phys = PhysParams(**phys_kwargs)
    train = TrainParams(**train_kwargs)

    model = MixedPINN(phys, train).to(device=device, dtype=dtype)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, phys, train, device, dtype

def save_loaded_prediction_plots(r_all, A_all, B_all, phys, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    r_ref = np.linspace(0.0, phys.R, 2000)
    A_ref, B_ref = exact_solution_from_phys(r_ref, phys)

    # A 图
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(r_all * 100.0, A_all, linewidth=2.0, label="PINN-loaded")
    ax1.plot(r_ref * 100.0, A_ref, "--", linewidth=1.6, label="Analytical")
    ax1.axvline(phys.r_coil_cm, linestyle="--", linewidth=1.2, color="gray")
    ax1.axvline(phys.r_fe_in_cm, linestyle="--", linewidth=1.2, color="gray")
    ax1.axvline(phys.r_fe_out_cm, linestyle="--", linewidth=1.2, color="gray")
    ax1.set_xlabel("r / cm")
    ax1.set_ylabel("A(r) / Wb/m")
    ax1.set_title("Loaded mixed PINN solution of magnetic vector potential")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(os.path.join(save_dir, "A_1D_axisymmetric_mixed_PINN_loaded.png"),
                 dpi=300, bbox_inches="tight")
    plt.close(fig1)

    # B 图
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.plot(r_all * 100.0, B_all, linewidth=2.0, label="PINN-loaded")
    ax2.plot(r_ref * 100.0, B_ref, "--", linewidth=1.6, label="Analytical")
    ax2.axvline(phys.r_coil_cm, linestyle="--", linewidth=1.2, color="gray")
    ax2.axvline(phys.r_fe_in_cm, linestyle="--", linewidth=1.2, color="gray")
    ax2.axvline(phys.r_fe_out_cm, linestyle="--", linewidth=1.2, color="gray")
    ax2.set_xlabel("r / cm")
    ax2.set_ylabel(r"$B_\theta(r)$ / T")
    ax2.set_title("Loaded mixed PINN solution of magnetic flux density")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(save_dir, "Btheta_1D_axisymmetric_mixed_PINN_loaded.png"),
                 dpi=300, bbox_inches="tight")
    plt.close(fig2)

if __name__ == "__main__":

    t_start = time.time()
    
    model, phys, train, device, dtype = load_trained_model("mixed_pinn_final_checkpoint.pth")

    r_all, A_all, B_all, q_all = stitch_prediction(model, phys, device, dtype)

    save_dir = os.path.dirname(os.path.abspath(__file__))
    save_loaded_prediction_plots(r_all, A_all, B_all, phys, save_dir)

    print("Model loaded successfully.")
    print("Max(A) =", A_all.max())
    print("Max(|B|) =", abs(B_all).max())

    t_end = time.time()     
    print(f"Total runtime = {t_end - t_start:.4f} s")