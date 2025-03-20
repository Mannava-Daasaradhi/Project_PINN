import torch
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

# ✅ Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ✅ Constants & Parameters
R = 8.3145
A = 1.4e8
v = 1.6
Ea = 121417.2
W = 0.02897
lambda_ = 0.026
cp = 1000.0
qF = 5.0e7
T_in = 298.0
dTdx_in = 1e5
L = 1.5e-3
epsilon = 1e-6

# ✅ Neural Network Architecture (DeepXDE)
layer_sizes = [1, 64, 64, 64, 1]
activation = "tanh"
initializer = "Glorot normal"

# ✅ Define Domain & Geometry
geom = dde.geometry.Interval(0, L)

# ✅ Governing Equation (PDE for Temperature)
def flame_pde(x, T):
    dTdx = dde.grad.jacobian(T, x)
    d2Tdx2 = dde.grad.hessian(T, x)

    S_L = 0.4  # Fixed assumption for forward problem
    u = (S_L + R * T_in / (W * S_L) - torch.sqrt((S_L + R * T_in / (W * S_L))**2 - 4 * R * T / W)) / 2
    rho = S_L / (u + epsilon)
    YF = 0.4 / (4 + 0.4) + cp * (T_in - T) / qF
    omega = A * torch.exp(-Ea / (R * T)) * (rho * YF) ** v

    return rho * S_L * cp * dTdx - lambda_ * d2Tdx2 - omega * qF

# ✅ Boundary Conditions
def boundary_lhs(x, T):
    return T - T_in

def boundary_rhs(x, T):
    return dde.grad.jacobian(T, x) - dTdx_in

# ✅ Define PINN Model
data = dde.data.PDE(
    geom, flame_pde, 
    [dde.DirichletBC(geom, boundary_lhs, lambda x: np.isclose(x, 0)),
     dde.OperatorBC(geom, boundary_rhs, lambda x: np.isclose(x, 0))],
    num_domain=1001, num_boundary=2
)

net = dde.maps.FNN(layer_sizes, activation, initializer)
model = dde.Model(data, net)

# ✅ Warm-up Training
x_warmup = np.linspace(0, L, 100)[:, None]
T_warmup = np.linspace(T_in, 1800, 100)[:, None]
data_warm = dde.data.PDE(geom, lambda x, y: 0, [], num_domain=100)
model_warm = dde.Model(data_warm, net)

model_warm.compile("adam", lr=1e-3)
model_warm.train(epochs=1000)

# ✅ Full Training
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(epochs=30000)

# ✅ Plot Results
x_test = np.linspace(0, L, 1000)[:, None]
T_pred = model.predict(x_test)
plt.plot(x_test, T_pred, label="PINN Prediction")
plt.xlabel("x (m)")
plt.ylabel("Temperature (K)")
plt.legend()
plt.savefig("T_vs_x.png")
plt.show()
