import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ✅ Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# 🔥 Constants
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
epsilon = 1e-6

# 🔹 Scaling Factors
scale_x = L
scale_T = 1800
scale_SL = 1.0  # 🔥 Adjusted scaling for stability

# 📌 Define Inverse PINN Model
class InversePINN(nn.Module):
    def __init__(self):
        super(InversePINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.S_L = nn.Parameter(torch.tensor(0.4, device=device))  # 🔥 Trainable parameter

    def forward(self, x):
        return self.net(x)

# 🎯 PDE Residual Function
def pde_residual(model, x, phi):
    x.requires_grad = True  # ✅ Ensure gradients can be computed
    T_pred = model(x) * scale_T
    T_pred = torch.clamp(T_pred, min=T_in + epsilon, max=2500)

    # 🔹 Compute Derivatives
    dTdx = torch.autograd.grad(T_pred, x, torch.ones_like(T_pred), create_graph=True)[0]
    d2Tdx2 = torch.autograd.grad(dTdx, x, torch.ones_like(dTdx), create_graph=True)[0]

    # 🔹 Derived Variables
    S_L = torch.sigmoid(model.S_L) * scale_SL  # 🔥 Normalized S_L
    c = S_L + R * T_in / (W * S_L)
    u = (c - torch.sqrt(torch.clamp(c**2 - 4 * R * T_pred / W, min=epsilon))) / 2
    rho = torch.clamp(S_L / (u + epsilon), min=epsilon)
    YF = torch.clamp(0.1 + cp * (T_in - T_pred) / qF, min=0.0)

    log_omega = torch.log(torch.tensor(A, dtype=torch.float32, device=T_pred.device)) - Ea / (R * T_pred) + v * torch.log(rho * YF + epsilon)
    omega = torch.exp(torch.clamp(log_omega, max=20))

    # 🔥 PDE Constraint
    residual = rho * S_L * cp * dTdx - lambda_ * d2Tdx2 - omega * qF
    return residual, S_L

# 🎓 Training Function
def train_inverse_pinn(phi):
    model = InversePINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10000, gamma=0.5)

    x_train = torch.linspace(0, L, 1001, device=device, requires_grad=True).unsqueeze(1)

    for epoch in range(40000):
        optimizer.zero_grad()
        residual, S_L = pde_residual(model, x_train, phi)
        loss_pde = torch.mean(residual ** 2)

        loss_pde.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 5000 == 0:
            print(f"Epoch {epoch}, PDE Loss: {loss_pde.item()}, S_L: {S_L.item()}")

    return S_L.item()

# 🔄 Training for Different Phi Values
phi_values = torch.tensor([0.4, 0.42, 0.44, 0.46, 0.48, 0.5], device=device)
sl_pred = []

for phi in phi_values:
    sl_value = train_inverse_pinn(phi)
    sl_pred.append(sl_value)
    print(f"Phi: {phi.item()}, Predicted S_L: {sl_value}")

# ✅ Convert Results to NumPy for Plotting
phi_values_np = phi_values.cpu().numpy()
sl_pred_np = np.array(sl_pred)

# 📊 Plot Results
plt.plot(phi_values_np, sl_pred_np, '-o', label="Predicted S_L (PINN)")
plt.xlabel("Phi (φ)")
plt.ylabel("Laminar Flame Speed (S_L) [m/s]")
plt.title("Laminar Flame Speed vs Equivalence Ratio (Phi)")
plt.legend()
plt.grid(True)
plt.savefig("inverse_pinn_sl_vs_phi.png")
plt.show()
