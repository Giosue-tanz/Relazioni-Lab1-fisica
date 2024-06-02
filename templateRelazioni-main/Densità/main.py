import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# plt.rcParams.update({
#    'font.size': 8,
#    'text.usetex': True,
#    'text.latex.preamble': r'\usepackage{amsfonts, siunitx}'
#})
def mod_d(x, m, q):
    return( m*x + q)


print(np.pi)
class Parallelepipedo():
    def __init__(self, materiale, m, sigma_m, base, larghezza, altezza, sigma_b, sigma_l, sigma_h):
        self.materiale = materiale
        self.m = m
        self.sigma_m = sigma_m
        self.b = base
        self.l = larghezza
        self.h = altezza
        self.sigma_b = sigma_b
        self.sigma_l = sigma_l
        self.sigma_h = sigma_h        
        self.calc_volume()
    def get_mass(self):
        print(f"La massa: {self.m} +- {self.sigma_m}")
    def calc_volume(self):
        self.volume = self.b * self.l * self.h
        self.err = np.sqrt(np.power(self.sigma_b/self.b, 2) + np.power(self.sigma_l/self.l, 2) + np.power(self.sigma_h/self.h, 2)) * self.volume
    def get_volume(self):
        print(f"{self.volume} +- {self.err}")

class Sfera():
    def __init__(self, materiale, m, sigma_m, d, sigma_d):
        self.materiale = materiale
        self.m = m
        self.sigma_m = sigma_m
        self.d = d
        self.sigma_d = sigma_d
        self.calc_volume()
    def get_mass(self):
        print(f"{self.m} +- {self.sigma_m}")
    def calc_volume(self):
        self.volume = (1./6.) * np.pi * np.power(self.d, 3)
        self.err = 3 * (self.sigma_d/self.d) * self.volume
    def get_volume(self):
        print(f"Il volume: {self.volume} +- {self.err}")

class Cilindro():
    def __init__(self, materiale, m, sigma_m, d, sigma_d, h, sigma_h):
        self.materiale = materiale
        self.m = m
        self.sigma_m = sigma_m
        self.d = d
        self.sigma_d = sigma_d
        self.h = h
        self.sigma_h = sigma_h
        self.calc_volume()
    def get_mass(self):
        print(f"{self.m} +- {self.sigma_m}")
    def calc_volume(self):
        self.volume = np.pi * np.power(self.d, 2) * (self.h / 4)
        self.err = np.sqrt(4 * np.power(self.sigma_d/self.d, 2) + np.power(self.sigma_h/self.h, 2)) * self.volume
    def get_volume(self):
        print(f"Il volume: {self.volume} +- {self.err}")

p1 = Cilindro("alluminio", 15.858, 0.010, 1.983, 0.005, 2.001, 0.005)
p2 = Cilindro("alluminio", 5.827, 0.010, 1.929, 0.005, 1.196, 0.005)
p3 = Parallelepipedo("alluminio", 4.822, 0.010, 1.005, 1.006, 1.783, 0.005, 0.005, 0.005)
p4 = Parallelepipedo("alluminio", 7.709, 0.010, 1.764, 2.008, 0.814, 0.005, 0.005, 0.005)

data = [p1, p2, p3, p4]

alluminio_v = []
alluminio_sigma_v = []
alluminio_m = []
alluminio_sigma_m = []

# Ripartizione dati in array
for el in data:
    alluminio_v.append(el.volume)
    alluminio_sigma_v.append(el.err)
    alluminio_m.append(el.m)
    alluminio_sigma_m.append(el.sigma_m)

# Conversione to np.array
alluminio_v = np.array(alluminio_v, dtype=np.float64)
alluminio_sigma_v = np.array(alluminio_sigma_v, dtype=np.float64)
alluminio_m = np.array(alluminio_m, dtype=np.float64)
alluminio_sigma_m = np.array(alluminio_sigma_m, dtype=np.float64)

# Grafico e modello di best-fit alluminio
popt, pcov = curve_fit(mod_d, alluminio_m, alluminio_v, [1./2.710, 0], sigma=alluminio_sigma_v)
sigma_eff = alluminio_sigma_v
for i in range(10):
    sigma_eff = np.sqrt(np.power(alluminio_sigma_v, 2) + np.power((1./popt[0]) * alluminio_sigma_m, 2))
    popt, pcov = curve_fit(mod_d, alluminio_m, alluminio_v, [1./2.710, 0], sigma=sigma_eff)
    chisq = np.sum(((alluminio_v - mod_d(alluminio_m, *popt))/sigma_eff) ** 2)
    print(f"Step {i}")
    print(popt, np.sqrt(pcov.diagonal()))
    print(f"Chisquare = {chisq:.2f}")


res = alluminio_v - mod_d(alluminio_m, *popt)

fig = plt.figure("Un grafico dei residui")

ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))

ax1.errorbar(alluminio_m, alluminio_v, alluminio_sigma_v, fmt='o', label='Dati')
xgrid = np.linspace(0.0, 16.0, 100)
ax1.plot(xgrid, mod_d(xgrid, *popt), label='Modello di best-fit')
ax1.set_ylabel("V [m^3]")
ax1.grid(color="lightgray", ls="dashed")
ax1.legend()


ax2.errorbar(alluminio_m, res, alluminio_sigma_v, fmt='o')
ax2.plot(xgrid, np.full(xgrid.shape, 0.0))
ax2.set_xlabel("m [kg]")
ax2.set_ylabel("Residui [m^3]")
plt.xlim(0.0, 16.0)
fig.align_ylabels((ax1, ax2))

plt.show()

print(f"{popt[0]} +- {np.sqrt(pcov.diagonal()[0])}")
