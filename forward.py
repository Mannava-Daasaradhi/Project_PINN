import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ✅ Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# 🔥 Constants (from the reference paper)
R = 8.3145
A = 1.4e8
v = 1.6
Ea = 1.5e5
W = 0.02897
lambda_ = 0.026
cp = 1000.0
qF = 5.0e7
T_in = 298.0
dTdx_in = 1e5
L = 1.5e-3
phi = 0.4  # Single training phi (not looping through multiple)
epsilon = 1e-6

# 📏 Scaling factors
scale_x = L
scale_T = 1800

# 🎯 Define the PINN Model
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x) * scale_T  # Scale output for temperature prediction

# 🧮 Define PDE Residual
def pde_residual(model, x):
    T_pred = model(x)  
    dTdx = torch.autograd.grad(T_pred, x, torch.ones_like(T_pred), create_graph=True)[0]
    d2Tdx2 = torch.autograd.grad(dTdx, x, torch.ones_like(dTdx), create_graph=True)[0]

    # 📌 Derived Quantities
    S_L = 0.4  # Given fixed input S_L
    YF_in = phi / (4 + phi)
    c = S_L + R * T_in / (W * S_L)
    u = (c - torch.sqrt(torch.clamp(c**2 - 4 * R * T_pred / W, min=epsilon))) / 2
    rho = torch.clamp(S_L / (u + epsilon), min=epsilon)
    YF = torch.clamp(YF_in + cp * (T_in - T_pred) / qF, min=0.0)

    log_omega = torch.log(torch.tensor(A, dtype=torch.float32, device=T_pred.device)) - Ea / (R * T_pred) + v * torch.log(rho * YF + epsilon)
    omega = torch.exp(torch.clamp(log_omega, max=20))

    # 🔥 Enforce Energy Equation
    residual = rho * S_L * cp * dTdx - lambda_ * d2Tdx2 - omega * qF
    return residual, T_pred, u, rho, omega, YF

# 🎓 Training Function
def train_forward_pinn():
    model = PINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 🏋️ Warmup Training (Z-curve Initialization)
    warmup_x = torch.linspace(0, L, steps=100, device=device).unsqueeze(1)
    warmup_T = torch.full_like(warmup_x, T_in)  # Initial condition

    for epoch in range(2000):
        optimizer.zero_grad()
        loss_warmup = torch.mean((model(warmup_x) - warmup_T) ** 2)
        loss_warmup.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f"Warmup Epoch {epoch}, Loss: {loss_warmup.item()}")

    # 🎓 Train PINN
    x_train = torch.linspace(0, L, steps=1000, device=device, requires_grad=True).unsqueeze(1)
    
    for epoch in range(40000):
        optimizer.zero_grad()
        residual, T_pred, u, rho, omega, YF = pde_residual(model, x_train)

        loss_pde = torch.mean(residual ** 2)
        loss_total = loss_pde

        loss_total.backward()
        optimizer.step()

        if epoch % 5000 == 0:
            print(f"Epoch {epoch}, PDE Loss: {loss_pde.item()}")

    return model

# 🔄 Train Model
model = train_forward_pinn()

# 🔥 Generate Predictions
x_test = torch.linspace(0, L, steps=200, device=device).unsqueeze(1)
_, T_pred, u_pred, rho_pred, omega_pred, YF_pred = pde_residual(model, x_test)

# 🖼️ Plot Results
fig, axs = plt.subplots(2, 3, figsize=(12, 8))
axs[0, 0].plot(x_test.cpu(), T_pred.cpu(), label="Temperature (T)")
axs[0, 1].plot(x_test.cpu(), YF_pred.cpu(), label="Fuel Mass Fraction (Y_F)")
axs[0, 2].plot(x_test.cpu(), u_pred.cpu(), label="Velocity (u)")
axs[1, 0].plot(x_test.cpu(), rho_pred.cpu(), label="Density (ρ)")
axs[1, 1].plot(x_test.cpu(), omega_pred.cpu(), label="Reaction Rate (ω)")
axs[1, 2].plot(x_test.cpu(), -rho_pred.cpu() * u_pred.cpu(), label="Pressure (p - p_in)")

for ax in axs.flatten():
    ax.legend()
    ax.set_xlabel("x (mm)")
plt.suptitle("Flame Properties for Forward PINN")
plt.show()
