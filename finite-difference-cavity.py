import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

nx = 100
ny = 100
Lx = 1.0
Ly = 1.0
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
rho = 1.0
nu = 0.01
dt = 0.001
nt = 10000
u0 = 1.
Re = (u0 * Lx) / nu
print(f'Re={Re}')

# Initialization
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))

x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)


def apply_bc(u, v):
    u[-1, :] = u0
    v[-1, :] = 0.0
    u[0, :] = 0.0
    u[1:, 0] = 0.0
    u[1:, -1] = 0.0
    v[0, :] = 0.0
    v[:, 0] = 0.0
    v[:, -1] = 0.0

def solve_pressure_poisson(p, b):
    pn = np.empty_like(p)
    for _ in range(50):
        pn = p.copy()
        p[1:-1, 1:-1] = ((pn[1:-1, 2:] + pn[1:-1, :-2]) * dy ** 2 +
                         (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx ** 2) / (2 * (dx ** 2 + dy ** 2)) - \
                        dx ** 2 * dy ** 2 / (2 * (dx ** 2 + dy ** 2)) * b[1:-1, 1:-1]
        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]
        p[0, :] = p[1, :]
        p[-1, :] = 0

# Main loop
for n in range(nt):
    un = u.copy()
    vn = v.copy()

    b[1:-1, 1:-1] = (rho * (1 / dt * ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx) +
                                       (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)) -
                             ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)) ** 2 -
                             2 * ((u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy) *
                                  (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)) -
                             ((v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)) ** 2))
    solve_pressure_poisson(p, b)

    u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, :-2]) -
                     vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[:-2, 1:-1]) -
                     dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, :-2]) +
                     nu * (dt / dx ** 2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) +
                           dt / dy ** 2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1])))

    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, :-2]) -
                     vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[:-2, 1:-1]) -
                     dt / (2 * rho * dy) * (p[2:, 1:-1] - p[:-2, 1:-1]) +
                     nu * (dt / dx ** 2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2]) +
                           dt / dy ** 2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1])))

    apply_bc(u, v)

df = pd.DataFrame({
    'x': np.tile(x, ny),
    'y': np.repeat(y, nx),
    'u': u.flatten(),
    'v': v.flatten(),
    'p': p.flatten()
})

df.to_csv('cavity_finite_difference_data.csv', index=False)

p[-1,:]

X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
plt.figure(figsize=(8, 6))
plt.imshow(np.sqrt(u**2+v**2), cmap='jet', extent=[0, Lx, 0, Ly])
plt.colorbar(label='Velocity Magnitude')
plt.streamplot(y, x, u, v, color='white')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Lid-driven Cavity Flow')
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
im = axs[0, 0].imshow(u, cmap='jet', extent=[0, Lx, 0, Ly])
axs[0, 0].set_title('Velocity u')
axs[0, 0].set_xlabel('X')
axs[0, 0].set_ylabel('Y')
axs[0, 0].set_aspect('equal')
im = axs[0, 1].imshow(v, cmap='jet', extent=[0, Lx, 0, Ly])
axs[0, 1].set_title('Velocity v')
axs[0, 1].set_xlabel('X')
axs[0, 1].set_ylabel('Y')
axs[0, 1].set_aspect('equal')
im = axs[1, 0].imshow(p, cmap='jet', extent=[0, Lx, 0, Ly])
axs[1, 0].set_title('Pressure p')
axs[1, 0].set_xlabel('X')
axs[1, 0].set_ylabel('Y')
axs[1, 0].set_aspect('equal')
im = axs[1, 1].imshow(np.sqrt(u**2 + v**2), cmap='jet', extent=[0, Lx, 0, Ly])
axs[1, 1].set_title('Velocity Magnitude')
axs[1, 1].set_xlabel('X')
axs[1, 1].set_ylabel('Y')
axs[1, 1].set_aspect('equal')
for ax in axs.flat:
    ax.label_outer()
    fig.colorbar(im, ax=ax, orientation='vertical')

plt.tight_layout()
plt.show()
