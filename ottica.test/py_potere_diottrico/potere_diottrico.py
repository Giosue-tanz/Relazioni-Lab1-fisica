import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import scipy.stats
import math

# liste dei dati
p1 = [4.6, 6.4, 7.9, 8.5, 9.6, 10.5, 11.6, 12.7, 13.5, 15.2]
q1 = [6.1, 9.8, 14.9, 14.6, 19.5, 22.8, 28.3, 44.1, 49.4, 66.8]
sigma_p1 = 0.5
sigma_q1 = 0.5 * np.ones_like(q1)  # Incertezza comune di 0.5 cm per tutte le misure di q


p_hat = np.mean(p1)
for i in range(len(p1)):
    one_over_p_hat = 1 / p_hat
    sigma_q_over_q_squared = sigma_q1[i] / (q1[i] ** 2)
    print(f"Punto {i+1}: 1/p_hat = {one_over_p_hat:.4f}, sigma_q/q^2 = {sigma_q_over_q_squared:.4f}")

# array dei dati
p1 = np.array(p1)
q1 = np.array(q1)
sigma_p1 = np.array(sigma_p1)
sigma_q1 = np.array(sigma_q1)

p = -1 / (p1)
q = 1 / (q1)
sigma_p = np.abs(-sigma_p1 / p1 ** 2)
sigma_q = np.abs(-sigma_q1 / q1 ** 2)

def diottria(x, m, f):  # x = p, f = 1/f e ciò che esce è q --> 1/q = 1/f-1/p
    return m * x + f

# grafico  absolute_sigma=True
fig = plt.figure('Grafico focale lenti', figsize=(10.0, 8.0), frameon=True)
ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[8, 3], top=0.95, bottom=0.11, left=0.11, right=0.95, hspace=0.2, wspace=0.2))
popt_f, pcov_f = curve_fit(diottria, p, q, sigma=sigma_q)
sigma_eff = np.sqrt(sigma_q ** 2 + (popt_f[0] * sigma_p) ** 2)
popt_f, pcov_f = curve_fit(diottria, p, q, sigma=sigma_eff)
chisq = (((q - diottria(p, *popt_f)) / sigma_eff) ** 2).sum()
print(popt_f, np.sqrt(pcov_f.diagonal()), f'chi quadro =', chisq)


p_inviluppo = np.linspace(-0.06, -0.225, 10)
ax1.errorbar(p, q, sigma_eff, fmt='og', label='Dati')
ax1.plot(p_inviluppo, diottria(p_inviluppo, *popt_f), color='lawngreen', label=f'Modello di best-fit(χ² ridotto: {chisq/(len(p)-len(popt_f)):.2f})')
ax1.set_ylabel('1/q [cm$^{-1}$]')
ax1.set_title(f"Fit per la stima del potere diottrico con χ²: {chisq:.2f}, Gradi di Libertà: {len(p)-len(popt_f)}")
ax1.grid(which='both', ls='dashed', color='gray')
ax1.legend()
residui = q - diottria(p, *popt_f)
ax2.errorbar(p, residui, sigma_eff, fmt='og')
ax2.plot(p_inviluppo, np.full(p_inviluppo.shape, 0.0), color='lawngreen')
ax2.set_xlabel('1/p [cm$^{-1}$]')
ax2.set_ylabel('Residui [cm$^{-1}$]')
ax2.grid(which='both', ls='dashed', color='gray')
plt.savefig('fit e residui per il potere diottrico')
plt.show()

# valori restituiti dal fit
print("I valori restituiti dal fit sono i seguenti:",
      "m=", popt_f[0], np.sqrt(pcov_f[0, 0]),
      "1/f", 1 / popt_f[1], (np.sqrt(pcov_f[1, 1]) / popt_f[1] ** 2))

# errori relativi
er_m = np.sqrt(pcov_f[0, 0]) / popt_f[0]
er_f = np.sqrt(pcov_f[1, 1]) / popt_f[1]
print("Gli errori relativi sui parametri sono:",
      "er_m=", er_m * 100,
      "er_f=", er_f * 100)

# test del chi2
expected_chi = len(p) - len(popt_f)
real_chi = ((residui / sigma_eff) ** 2).sum()
variance_chi = np.sqrt(2 * expected_chi)
print("Il chi quadro ottenuto è", real_chi, "/", expected_chi, "dof, con una varianza di", variance_chi)


# p-value
def p_value(chi2, ndof):
    v = scipy.stats.chi2.cdf(chi2, ndof)
    if v > 0.5:
        v = 1 - v
    return v

print("p-value =", p_value(real_chi, expected_chi))