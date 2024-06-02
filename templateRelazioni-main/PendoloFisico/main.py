import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define constants
l_cm = 54.55
g = 9.80665
dist_raw = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0], dtype=float)
dist = 105.6 - dist_raw
d = abs(l_cm - dist) * 1e-2
err_dist = np.full(len(dist), np.sqrt(0.1**2 + 0.1**2)) * 1e-2

# Function to calculate mean and error
def calculate_mean_and_error(data):
    mean = np.mean(data)
    sigma = np.sqrt(np.sum((data - mean) ** 2) / (len(data) * (len(data) - 1)))
    return mean, sigma

# Process data
buco = {
    "buco_1": [15.67, 15.80, 15.84, 15.98, 16.00, 16.03, 16.12],
    "buco_2": [15.59, 15.73, 15.57, 15.47, 15.54, 15.74, 15.71],
    "buco_3": [15.85, 15.93, 15.96, 16.03, 15.93, 15.77, 15.93],
    "buco_4": [18.65, 18.65, 18.58, 18.51, 18.73, 18.47, 18.52],
    "buco_5": [38.20, 39.40, 42.49, 38.69, 38.01, 37.44, 39.27],
    "buco_6": [22.77, 22.80, 22.67, 22.77, 22.50, 22.54, 22.34],
    "buco_7": [16.70, 16.93, 16.65, 16.67, 16.70, 16.69, 16.53],
    "buco_8": [15.56, 15.55, 15.63, 15.53, 15.57, 15.44, 15.89],
    "buco_9": [15.61, 15.72, 15.83, 15.89, 15.67, 15.81, 15.73],
    "buco_10": [16.49, 16.42, 16.43, 16.45, 16.46, 16.37, 16.43]
}

# Calculate mean and error for each buco
T = np.array([calculate_mean_and_error(buco[el])[0] for el in buco]) / 10
err_T = np.array([calculate_mean_and_error(buco[el])[1] for el in buco]) / 10

# Print the period and its associated error for each buco
for i, key in enumerate(buco.keys()):
    mean, sigma = calculate_mean_and_error(buco[key])
    print(f"{key}: Periodo medio = {mean/10:.3f} s, Errore = {sigma/10:.3f} s")

# Define the model and its derivative
def period_model(d, l):
    return 2.0 * np.pi * np.sqrt((l**2.0 / 12.0 + d**2.0) / (g * d))

def derivative(d, l):
    return abs((np.pi * (1./12. * l**2 - d**2)) / (d**2 * g * np.sqrt((d**2 + 1./12. * l**2) / (d * g))))

# Iteratively fit the model and update errors
err_eff = err_T.copy()
for i in range(100):
    popt, pcov = curve_fit(period_model, d, T, sigma=err_eff)
    err_eff = np.sqrt(err_T**2 + (derivative(d, popt[0]) * err_dist)**2)

# Calculate residuals
res = T - period_model(d, *popt)

# Calcolo del chi^2 e del chi^2 ridotto
chi2 = np.sum(((T - period_model(d, popt[0]))**2) / (err_T**2))
dof = len(T) - len(popt)  # Gradi di libertà
chi2_reduced = chi2 / dof

# Creazione della figura
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))

# Plot principale: dati e modello di best-fit
ax1.errorbar(d, T, err_T, fmt="o", label="Dati")
xgrid = np.linspace(0.0, 0.5, 100)
ax1.plot(xgrid, period_model(xgrid, *popt), label=f'Best-fit model (χ² ridotto: {chi2_reduced:.2f})', color="orange")
ax1.set_ylabel("T [s]")
ax1.grid(color="lightgray", ls="dashed")
ax1.legend()

# Titolo del grafico principale
ax1.set_title(f"Fit per la stima dell'indice con χ²: {chi2:.2f}, Gradi di Libertà: {dof}")

# Plot dei residui
ax2.errorbar(d, res, err_T, fmt="o")
ax2.plot(xgrid, np.zeros_like(xgrid), color='orange')
ax2.set_xlabel("d [m]")
ax2.set_ylabel("Residui [s]")
ax2.grid(color="lightgray", ls="dashed")

# Finalizzazione del grafico
plt.xlim(0.0, 0.5)
fig.align_ylabels((ax1, ax2))
plt.savefig("Fit_e_residui.pdf")
plt.show()

# Calculate chi-squared
chi2 = np.sum(((T - period_model(d, popt[0]))**2) / (err_T**2))
print(f"l: {popt[0]} +- {np.sqrt(np.diag(pcov))}")
print(f"Chi-squared: {chi2}")
