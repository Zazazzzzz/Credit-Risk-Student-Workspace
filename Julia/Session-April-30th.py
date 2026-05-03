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

#Slide 50 First portfolio simulation
n_obligors= 100

M= 1000 #sim numbers

PD= 0.02 #prob of default
EAD = 100  #exposure at default "amount of
#money the lender is exposed to at the moment
#the borrower defaults"
LGD = 0.6#loss given default -> how much of the exposure
#is actually lost when a borrower defaults
#Loss= PD*EAD*LGD

portfolio_losses = []
for i in range(M):

    defaults = np.random.binomial(1, PD, n_obligors) #makes a simulation
    losses = defaults * EAD * LGD #individula losses
    total_loss = losses.sum()

    portfolio_losses.append(total_loss)

portfolio_losses = np.array(portfolio_losses)
EL = portfolio_losses.mean()
print("Expected Loss:", EL)
UL = portfolio_losses.std()
print("Unexpected Loss:", UL)

VaR_95 = np.percentile(portfolio_losses, 95)
print("95% VaR:in 95% of simulated scenario, losses are under ", VaR_95)

plt.hist(portfolio_losses, bins=30)
plt.xlabel("Portfolio Loss")
plt.ylabel("Frequency")
plt.title("Portfolio Loss Distr.")
plt.show()

#Monte Carlo portfolio setup, default or not matrix
np.random.seed(50)

M_MC= 1000 #n° simulations (rows as zaza said)
N_MC= 25 #n° firms

PD_MC = 0.02
EAD_MC= 100
LGD_MC = 0.6

defaults_MC= np.random.binomial(1,PD_MC,size=(M_MC,N_MC))
#setsup the default matrix

loss_matrix = defaults_MC * EAD_MC * LGD_MC #loss matrix

portfolio_losses_MC= loss_matrix.sum(axis=1)
# aggregates across firms

EL = portfolio_losses_MC.mean()
UL = portfolio_losses_MC.std() #volatility
print("Expected Loss:", EL)
print("Unexpected Loss:", UL)

VaR_95 = np.percentile(portfolio_losses_MC, 95)
print("95% VaR:", VaR_95)

plt.hist(portfolio_losses, bins=50)
plt.title("Loss Distribution")
plt.xlabel("Loss")
plt.ylabel("Frequency")
plt.show()
