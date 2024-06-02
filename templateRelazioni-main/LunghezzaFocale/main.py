import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2
# Dati
p = np.array([11.1, 10.2, 12.8, 14.8, 9.7, 16.4, 19.5, 11.2])
q = np.array([16.2, 18.3, 17.3, 15.8, 17.7, 15.3, 14.7, 16.6])
p = p * 10 ** (-2)
q = q * 10 ** (-2)
sigma = 0.01
# Calcolo delle variabili P e Q 
P = np.power(p, -1)
Q = np.power(q, -1)

# Propagazione degli errori
sigma_Q = 1 * ((0.5 * 10**(-2))/ q**2)
sigma_P =  (0.5 * 10**(-2)/ p**2)

# Definizione del modello
def model(x, m, F):
    return F + m * x

# Ottimizzazione del fit
popt, pcov = curve_fit(model, P, Q, sigma=sigma_Q)
res = Q - model(P, *popt)

acc = 0
# Errori efficaci e fit iterativo
for i in range(3):
    sigma_eff = np.sqrt(sigma_Q**2 + (popt[0] * sigma_P)**2)
    popt, pcov = curve_fit(model, P, Q, sigma=sigma_eff)
    # chisq = (((Q - model(P, *popt)) / sigma_eff)**2).sum()
    for el in zip(Q, P, sigma_eff):
        acc = acc + np.power((el[0]-model(el[1], *popt))/(el[2]) , 2)
    print(f'Step {i}...')
    print(f'Parametri ottimizzati: {popt}, Errori: {np.sqrt(pcov.diagonal())}')
    print(f'Chisquare = {acc:.2f}')
    acc = 0
# Creazione del grafico
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))
xgrid = np.linspace(0.0, np.max(P), 100)

# Grafico dati e modello
ax1.errorbar(P, Q, yerr=sigma_Q, xerr=sigma_P, fmt='o', label='Dati sperimentali')
ax1.plot(xgrid, model(xgrid, *popt), label='Modello di best fit')
ax1.set_ylabel('Q [cm**-1]')
ax1.grid(color='lightgray', ls='dashed')
ax1.legend()

# Calcolo e visualizzazione dei residui
# chisq = (((Q - model(P, *popt)) / sigma_eff)**2).sum()
chisq = np.sum(np.power((Q - model(P, *popt))/(sigma_eff), 2))
print(chisq)
print(f"p_value: {chi2.cdf(chisq, 8)}")
ax2.errorbar(P, res, sigma_Q, fmt='o')
ax2.plot(xgrid, np.full(xgrid.shape, 0.0), color='gray')
ax2.set_xlabel('1/p')
ax2.set_ylabel('Residui')
ax2.grid(color='lightgray', ls='dashed')

fig.align_ylabels((ax1, ax2))
plt.show()