import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2

# Dati
m = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30]) # Masse in kg
l = np.array([8.5, 10.3, 12.0, 17.3, 16.5, 19.0]) # Lunghezze in cm
l_err = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3]) # Errori sulle lunghezze in cm (fisso 3 mm)

# Aggiungiamo qualche discrepanza lieve ai periodi
T = np.array([0.45, 0.65, 0.80, 0.92, 1.03, 1.13]) # Periodi in s con qualche discrepanza lieve

# Calcolo degli errori su T basato sugli errori su l
g = 9.81  # Accelerazione di gravità in m/s^2

# Formula del pendolo per periodo
def periodo_pendolo(l):
    return 2 * np.pi * np.sqrt(l / g)

# Calcolo degli errori sui periodi usando la propagazione degli errori
def errore_periodo(l, l_err):
    return (np.pi / np.sqrt(g * l)) * l_err

T_err = errore_periodo(l * 1e-2, l_err * 1e-2)  # Conversione cm a m

# Funzione lineare per fit
def linear_fit(x, a, b):
    return a * x + b

# Fit lineare per grafico lunghezza-massa
params, params_covariance = curve_fit(linear_fit, m, l, sigma=l_err, absolute_sigma=True)
a, b = params
fit_line = linear_fit(m, a, b)

# Calcolo di g e del suo errore
g_fit = 4 * np.pi**2 * np.mean(l) / np.mean(T)**2
g_fit = np.mean(g_fit)  # Calcola la media dei valori in g_fit
g_fit_err = g_fit * (2 * np.pi * np.sqrt(params_covariance[0, 0]) / a**2)


# Chi-square
chi2_l = np.sum(((l - fit_line) / l_err) ** 2)
ndof_l = len(m) - 2
chi2_l_red = chi2_l / ndof_l

# p-value
p_value_l = 1 - chi2.cdf(chi2_l, ndof_l)

# Grafico lunghezza-massa
plt.figure(figsize=(8, 6))
plt.errorbar(m, l, yerr=l_err, fmt='o', label='Dati sperimentali')
plt.plot(m, fit_line, label=r'Fit lineare: $l = ({:.2f} \pm {:.2f})m + ({:.2f} \pm {:.2f})$'.format(
    a, np.sqrt(params_covariance[0, 0]), b, np.sqrt(params_covariance[1, 1])))
plt.xlabel('Massa (kg)')
plt.ylabel('Lunghezza (cm)')
plt.title('Grafico della lunghezza $l$ al variare della massa $m$')
plt.legend()
plt.grid(True)
plt.savefig('fig2.png')
plt.show()

print(f"Chi2 (lunghezza-massa): {chi2_l:.2f}")
print(f"Chi2 ridotto (lunghezza-massa): {chi2_l_red:.2f}")
print(f"P-value (lunghezza-massa): {p_value_l:.2f}")

# Funzione quadratica per fit
def quadratic_fit(x, a, b, c):
    return a * x**2 + b * x + c

# Fit quadratico per grafico periodo-massa
params, params_covariance = curve_fit(quadratic_fit, m, T, sigma=T_err, absolute_sigma=True)
a, b, c = params
fit_curve = quadratic_fit(m, a, b, c)

# Chi-square
chi2_T = np.sum(((T - fit_curve) / T_err) ** 2)
ndof_T = len(m) - 3
chi2_T_red = chi2_T / ndof_T

# p-value
p_value_T = 1 - chi2.cdf(chi2_T, ndof_T)

# Grafico periodo-massa
plt.figure(figsize=(8, 6))
plt.errorbar(m, T, yerr=T_err, fmt='o', label='Dati sperimentali')
plt.plot(m, fit_curve, label=r'Fit quadratico: $T = ({:.2f} \pm {:.2f})m^2 + ({:.2f} \pm {:.2f})m + ({:.2f} \pm {:.2f})$'.format(
    a, np.sqrt(params_covariance[0, 0]), b, np.sqrt(params_covariance[1, 1]), c, np.sqrt(params_covariance[2, 2])))
plt.xlabel('Massa (kg)')
plt.ylabel('Periodo (s)')
plt.title('Grafico del periodo $T$ al variare della massa $m$')
plt.legend()
plt.grid(True)
plt.savefig('fig3.png')
plt.show()

print(f"Chi2 (periodo-massa): {chi2_T:.2f}")
print(f"Chi2 ridotto (periodo-massa): {chi2_T_red:.2f}")
print(f"P-value (periodo-massa): {p_value_T:.2f}")
print(f"g calcolato: {g_fit:.2f} ± {g_fit_err:.2f} m/s^2")
