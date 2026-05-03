# Lecture Slide 63
# Simulate default path under Merton's model
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
V0 = 100
mu_V = 0.05 # drift (expected return)
sigma_V = 0.2
T = 1.0 # step size
B = 90 # default threshold
N = 1000 # number of steps
dt = T/N # time step

np.random.seed(42)
#for reproducibility

# standard Brownian motion
# delta_W_t = W_(t+delta_t)-W_t
# delta_W_t ~ N(0, delta_t)
delta_W = np.random.normal(0, np.sqrt(dt), N)
# N independent random samples from a normal distribution
W = delta_W.cumsum()

time_grid = np.linspace(0, T, N)
# create an array of N evenly spaced time points from 0 to T
# to serve as the time axis for simulating and plotting the asset value path


# without default
VT = V0 * np.exp((mu_V - 0.5 * sigma_V**2) * time_grid + sigma_V * W)
# simulate the path of V_t under Merton's model

default_indices = np.where(VT <= B)[0]
# to find the indices of all time points where the asset value VT is less than the default threshold B
# [0]:
# default_indices = np.where(VT <= B) returns to a tuple like (array([12, 13, 14, 20]),)
# [0] extract the first element of the tuple returned by np.where(), which contains the indices where the condition is true.

if len(default_indices) > 0:
    default_index = default_indices[0]
    default_time = time_grid[default_index]
    default_value = VT[default_index]

    print(f"Default occurs at time t = {default_time:.4f}")
    print(f"Asset value at default = {default_value:.4f}")
else:
    print("No default occurs during the simulation.")

plt.figure(figsize=(10,5))

plt.plot(time_grid, VT, label="Asset value $V_t$")
plt.axhline(y=B, linestyle="--", label="Default threshold B=90")

if len(default_indices) > 0:
    plt.scatter(default_time, default_value, label="Default point")

plt.xlabel("Time")
plt.ylabel("Asset Value")
plt.title("Simulated Default Path uner Merton's Model")
plt.legend()
plt.grid(True)
plt.show()

# with default probability
d = (np.log(B) - np.log(V0) - (mu_V - 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
default_prob = norm.cdf(d)

# simulate a path under Brownian motion
def simulate_path(V0, mu_V, sigma_V, T, N):
    dt = T / N
    time_grid = np.linspace(0, T, N+1)

    # Brownian motion with W_0 = 0
    dW = np.random.normal(loc=0, scale=np.sqrt(dt), size=N)
    W = np.concatenate(([0], dW.cumsum()))

    # Merton model path
    V = V0 * np.exp((mu_V - 0.5 * sigma_V ** 2) * time_grid + sigma_V * W)

    return time_grid, V

# simulate a path with default
attempt = 0
while True:
    attempt += 1
    time_grid, V_path = simulate_path(V0, mu_V, sigma_V, T, N)

    default_indices = np.where(V_path <= B)[0]
    if len(default_indices) > 0:
        default_index = default_indices[0]
        default_time = time_grid[default_index]
        default_value = V_path[default_index]
        break

print(f"Found a default path after {attempt} simulation(s).")
print(f"Default occurs at time t = {default_time:.4f}")
print(f"Asset value at default = {default_value:.2f}")

plt.figure(figsize=(10, 5))
plt.plot(time_grid, V_path, label="Asset value path $V_t$")
plt.axhline(y=B, linestyle="--", label=f"Default threshold B = {B}")

# Mark default point
plt.scatter(default_time, default_value, zorder=5, label="Default point")

plt.xlabel("Time")
plt.ylabel("Asset value")
plt.title(f"Default Path under Merton's Model\nAnalytical default probability = {default_prob:.4f}")
plt.legend()
plt.grid(True)
plt.show()