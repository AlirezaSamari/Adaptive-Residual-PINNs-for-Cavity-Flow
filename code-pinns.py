"""
Author: Alireza Samari
"""
import numpy as np
import torch
import pandas as pd
from torch import nn, optim
from torchsummary import summary
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.io
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Your device is [{device}]')
from google.colab import drive
drive.mount('/content/drive')
path = '/content/drive/MyDrive/physics informed/Cavity/cavity_finite_difference_data.csv'

class PINN():
    class AdaptiveActivation(nn.Module):
        def __init__(self):
          super(PINN.AdaptiveActivation, self).__init__()
          self.tanh = nn.Tanh()
          self.alpha = nn.Parameter(torch.tensor(0.9))

        def forward(self, x):
          return self.tanh(self.alpha * x)

    class ResidualBlock(nn.Module):
        def __init__(self, input_size, output_size):
          super().__init__()
          self.activation = PINN.AdaptiveActivation()
          self.fc1 = nn.Linear(input_size, output_size)
          nn.init.xavier_uniform_(self.fc1.weight)
          self.fc2 = nn.Linear(output_size, output_size)
          nn.init.xavier_uniform_(self.fc2.weight)
          self.fc_residual = nn.Linear(input_size, output_size)
          nn.init.xavier_uniform_(self.fc_residual.weight)

          self.beta = nn.Parameter(torch.tensor(0.5))

        def forward(self, x):
          beta = torch.clamp(self.beta, min=0.01, max=1.0)
          residual = x
          x = self.activation(self.fc1(x))
          x = self.fc2(x)
          residual = self.fc_residual(residual)
          x = self.beta * x + (1 - self.beta) * residual
          x = self.activation(x)
          return x

    class AdaptiveResNet(nn.Module):
        def __init__(self, input_size, output_size, residual_blocks_neurons):
          super().__init__()
          layers = []
          prev_layer_size = input_size
          for neurons in residual_blocks_neurons:
              residual_block = PINN.ResidualBlock(prev_layer_size, neurons)
              layers.append(residual_block)
              prev_layer_size = neurons

          self.output_layer = nn.Linear(prev_layer_size, output_size)
          nn.init.xavier_uniform_(self.output_layer.weight)

          self.model = nn.Sequential(*layers)

        def forward(self, x):
          output = self.model(x)
          output = self.output_layer(output)
          return output

    def __init__(self, X, Y, X_physics, Y_physics, u, v, p, Xb, Yb, ub, vb, pb, pretrained):
        self.pretrained = pretrained
        self.x_physics = torch.tensor(X_physics, dtype=torch.float32, requires_grad=True).to(device)
        self.y_physics = torch.tensor(Y_physics, dtype=torch.float32, requires_grad=True).to(device)
        self.x = torch.tensor(X, dtype=torch.float32, requires_grad=True).to(device)
        self.y = torch.tensor(Y, dtype=torch.float32, requires_grad=True).to(device)

        self.u = torch.tensor(u, dtype=torch.float32).to(device)
        self.v = torch.tensor(v, dtype=torch.float32).to(device)
        self.p = torch.tensor(p, dtype=torch.float32).to(device)

        self.xb = torch.tensor(Xb, dtype=torch.float32, requires_grad=True).to(device)
        self.yb = torch.tensor(Yb, dtype=torch.float32, requires_grad=True).to(device)

        self.ub = torch.tensor(ub, dtype=torch.float32).to(device)
        self.vb = torch.tensor(vb, dtype=torch.float32).to(device)
        self.pb = torch.tensor(pb, dtype=torch.float32).to(device)

        self.nu = 0.01
        self.rho = 0.1

        self.pinn()
        self.model.to(device)
        self.optimizer = optim.LBFGS(self.model.parameters(), lr=1., max_iter=200000, max_eval=50000,
                                     history_size=50, tolerance_grad=1e-05, tolerance_change=0.5 * np.finfo(float).eps,
                                     line_search_fn="strong_wolfe")
        self.mse = nn.MSELoss()
        self.total_loss = 0
        self.iter = 0

    def pinn(self):
        input_size = 2
        output_size = 2
        residual_blocks_neurons = 5*[20]
        self.model = PINN.AdaptiveResNet(input_size, output_size, residual_blocks_neurons).to(device)
        if self.pretrained==True: self.model.load_state_dict(torch.load('/content/drive/MyDrive/physics informed/Cavity/ResBNet.pt', weights_only=True, map_location=torch.device(device)))
        else: pass

    def calc_grad(self, y, x):
        grad = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=torch.ones_like(y),
            create_graph=True
        )[0]
        return grad

    def NavierStokes(self, x, y):
        out = self.model(torch.hstack((x, y)))
        psi, p = out[:, 0:1], out[:, 1:2]

        u = self.calc_grad(psi, y)
        v = -1. * self.calc_grad(psi, x)
        u_x = self.calc_grad(u, x)
        u_xx = self.calc_grad(u_x, x)
        u_y = self.calc_grad(u, y)
        u_yy = self.calc_grad(u_y, y)
        v_x = self.calc_grad(v, x)
        v_xx = self.calc_grad(v_x, x)
        v_y = self.calc_grad(v, y)
        v_yy = self.calc_grad(v_y, y)

        p_x = self.calc_grad(p, x)
        p_y = self.calc_grad(p, y)

        f_u = u * u_x + v * u_y + (p_x/self.rho) - self.nu * (u_xx + u_yy)
        f_v = u * v_x + v * v_y + (p_y/self.rho) - self.nu * (v_xx + v_yy)

        return u, v, p, f_u, f_v

    def closure(self):
        self.optimizer.zero_grad()
        u_physics, v_physics, p_physics, f_u_physics, f_v_physics = self.NavierStokes(self.x_physics, self.y_physics)
        f_u_physics_loss = self.mse(f_u_physics, torch.zeros_like(f_u_physics))
        f_v_physics_loss = self.mse(f_v_physics, torch.zeros_like(f_v_physics))

        u_hat, v_hat, p_hat, f_u_hat, f_v_hat = self.NavierStokes(self.x, self.y)
        u_loss = self.mse(u_hat, self.u)
        v_loss = self.mse(v_hat, self.v)
        f_u_loss = self.mse(f_u_hat, torch.zeros_like(f_u_hat))
        f_v_loss = self.mse(f_v_hat, torch.zeros_like(f_v_hat))


        ub_hat, vb_hat, pb_hat, _, _ = self.NavierStokes(self.xb, self.yb)
        ub_loss = self.mse(ub_hat, self.ub)
        vb_loss = self.mse(vb_hat, self.vb)


        self.interior_loss = f_u_loss + f_v_loss + u_loss + v_loss
        self.boundary_loss = ub_loss + vb_loss
        self.physics_loss = f_u_physics_loss + f_v_physics_loss
        self.total_loss = self.interior_loss + self.boundary_loss + self.physics_loss
        self.total_loss.backward()

        self.iter += 1
        if self.iter % 100 == 0:
            print(f"Iteration: {self.iter}")
            print(f"Interior Loss: {self.interior_loss.item()}, U: {u_loss.item()}, V: {v_loss.item()}, fu: {f_u_loss.item()}, fv: {f_v_loss.item()}")
            print(f"Boundary Loss: {self.boundary_loss.item()}, Ub: {ub_loss.item()}, Vb: {vb_loss.item()}")
            print(f"Physics Loss: fu_physics: {f_u_physics_loss.item()}, fv_physics: {f_v_physics_loss.item()}")
            print(f"Total Loss: {self.total_loss}")
            print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
            torch.save(self.model.state_dict(), '/content/drive/MyDrive/physics informed/Cavity/ResBNet.pt')
        return self.total_loss

    def train(self):
        self.model.train()
        self.optimizer.step(self.closure)


physics_points = 5000
boundary_points = 396
interior_points = 100
df = pd.read_csv(path)
x = df.iloc[:, 0:1].values
y = df.iloc[:, 1:2].values
u = df.iloc[:, 2:3].values
v = df.iloc[:, 3:4].values
p = df.iloc[:, 4:5].values
epsilon = 0.01
x_physics_range = np.linspace(0, 1, int(np.sqrt(physics_points)))
y_physics_range = np.linspace(0, 1, int(np.sqrt(physics_points)))
X_grid, Y_grid = np.meshgrid(x_physics_range, y_physics_range, indexing='ij')
X_physics = X_grid.reshape(-1, 1)
Y_physics = Y_grid.reshape(-1, 1)

def is_boundary(x, y):
    return (x == 0) | (x == 1) | (y == 0) | (y == 1)

boundary_indices = is_boundary(x, y)
interior_indices = ~boundary_indices

x_boundary = x[boundary_indices]
y_boundary = y[boundary_indices]
u_boundary = u[boundary_indices]
v_boundary = v[boundary_indices]
p_boundary = p[boundary_indices]

x_interior = x[interior_indices]
y_interior = y[interior_indices]
u_interior = u[interior_indices]
v_interior = v[interior_indices]
p_interior = p[interior_indices]

N_boundary = len(x_boundary)
N_interior = len(x_interior)


N_train_boundary = min(N_boundary, boundary_points)
N_train_interior = min(N_interior, 100)

idx_boundary = np.random.choice(N_boundary, N_train_boundary, replace=False)
idx_interior = np.random.choice(N_interior, N_train_interior, replace=False)

xb_train = x_boundary[idx_boundary].reshape(-1, 1)
yb_train = y_boundary[idx_boundary].reshape(-1, 1)
ub_train = u_boundary[idx_boundary].reshape(-1, 1)
vb_train = v_boundary[idx_boundary].reshape(-1, 1)
pb_train = p_boundary[idx_boundary].reshape(-1, 1)

x_train = x_interior[idx_interior].reshape(-1, 1)
y_train = y_interior[idx_interior].reshape(-1, 1)
u_train = u_interior[idx_interior].reshape(-1, 1)
v_train = v_interior[idx_interior].reshape(-1, 1)
p_train = p_interior[idx_interior].reshape(-1, 1)


fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(X_physics, Y_physics, color='#66c2a5', marker='o', label='Physics Points' ,s=4, alpha=1)
ax.scatter(x_train, y_train, color='#e45549', marker='o', s=4, label='Labeled Points', alpha=0.7)
ax.scatter(xb_train, yb_train, color='#e45549', marker='o', s=4, alpha=0.7)

ax.set_title('Data Points')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/physics informed/Cavity/Data_Points.png', format='png', dpi=700)
plt.show()

pinn = PINN(x_train, y_train, X_physics, Y_physics, u_train, v_train, p_train, xb_train, yb_train, ub_train, vb_train, pb_train, pretrained=True)
pinn.train()

pinn.model.eval()
x_test = torch.tensor(x, dtype=torch.float32).clone().detach().requires_grad_(True).to(device)
y_test = torch.tensor(y, dtype=torch.float32).clone().detach().requires_grad_(True).to(device)

u_out, v_out, p_out, f_u_out, f_v_out = pinn.NavierStokes(x_test, y_test)

grid_shape = (100, 100)

u_plot = u_out.data.cpu().numpy().reshape(grid_shape)
v_plot = v_out.data.cpu().numpy().reshape(grid_shape)
p_plot = p_out.data.cpu().numpy().reshape(grid_shape)
u_magnitude = torch.sqrt(u_out**2 + v_out**2).data.cpu().numpy().reshape(grid_shape)

plt.figure(figsize=(6, 5))
contour_u = plt.contourf(u_plot, levels=50, cmap='Spectral')
plt.title(r'Horizontal Velocity Component ($u$)')
plt.colorbar(contour_u)
plt.savefig('/content/drive/MyDrive/physics informed/Cavity/horizontal_velocity_component_u.png', format='png', dpi=1000)
plt.show()

plt.figure(figsize=(6, 5))
contour_v = plt.contourf(v_plot, levels=50, cmap='Spectral')
plt.title(r'Vertical Velocity Component ($v$)')
plt.colorbar(contour_v)
plt.savefig('/content/drive/MyDrive/physics informed/Cavity/vertical_velocity_component_v.png', format='png', dpi=1000)
plt.show()

plt.figure(figsize=(6, 5))
contour_p = plt.contourf(p_plot, levels=100, cmap='Spectral')
plt.title(r'Pressure ($p$)')
plt.colorbar(contour_p)
plt.savefig('/content/drive/MyDrive/physics informed/Cavity/pressure_p.png', format='png', dpi=1000)
plt.show()

plt.figure(figsize=(6, 5))
contour_umag = plt.contourf(u_magnitude, levels=50, cmap='Spectral')
plt.title(r'Velocity Magnitude $|u|$')
plt.colorbar(contour_umag)
plt.savefig('/content/drive/MyDrive/physics informed/Cavity/velocity_magnitude_u.png', format='png', dpi=1000)
plt.show()
