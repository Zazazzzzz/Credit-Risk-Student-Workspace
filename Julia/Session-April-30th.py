#Merton Model
import numpy as np
import matplotlib.pyplot as plt

V0=100  #asset value at beginning
mu_V= 0.05 #exp asset return
sigma_V= 0.2 #asset volatility
T=1.0 #1 year
B=90   #Debt face value (default barrier at maturity)
N=1000  #number of simulation steps
dt=(T/N)  #"time increment"

np.random.seed(42)

W=np.random.normal(0,np.sqrt(dt),N).cumsum()
#this creates the "brownian increments", (independent gaussian)
#the sum makes it a standard broenian motion

time_grid= np.linspace(0,T,N) #even space within the year

VT = V0*np.exp((mu_V-0.5*sigma_V**2)*time_grid + sigma_V*W)
# - 0.5 sigma is the "Ito correction term", exponentials
# of RVs are not linear
# Looks if at T: V > B: can repay debt, no default
#V<B firm defaults

default = VT[-1] < B #takes final value

print(f"Value at maturiy: {VT[-1]}, default: {default}")

plt.plot(time_grid,VT)
plt.axhline(B, color="red", linestyle="--",label="debt barrier")
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Merton model simulation")
plt.legend()
plt.show()






