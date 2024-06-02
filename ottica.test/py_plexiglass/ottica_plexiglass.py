import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2

def indice(x, m, q): 
    return m * x + q

def plot_data(sin_i, sin_r, sigma_eff, popt, chisq, p_value):
    i_inviluppo = np.linspace(0, max(sin_i), len(sin_i))
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[8, 3], top=0.95, bottom=0.11, left=0.11, right=0.95, hspace=0.2, wspace=0.2))
    ax1.errorbar(sin_i, sin_r, sigma_eff, fmt='o', label='Data')
    ax1.plot(i_inviluppo, indice(i_inviluppo, *popt), color='orange', label=f'Best-fit model (χ² ridotto: {chisq/(len(sin_i)-len(popt)):.2f})')
    ax1.set_title(f"Fit per la stima dell'indice con χ²: {chisq:.2f}, Gradi di Libertà: {len(sin_i)-len(popt)}")
    ax1.grid(which='both', ls='dashed', color='gray')
    ax1.set_ylabel('sin(r)')
    ax1.legend()
    residui = sin_r - indice(sin_i, *popt)
    ax2.errorbar(sin_i, residui, sigma_eff, fmt='o')
    ax2.plot(i_inviluppo, np.full(i_inviluppo.shape, 0.0), color='orange')
    ax2.set_xlabel('sin(i)')
    ax2.set_ylabel('Residues')
    ax2.grid(which='both', ls='dashed', color='gray')
    plt.savefig('fit e residui per plexiglass')
    plt.show()

def fit_model(sin_i, sin_r, di, dr):
    popt, pcov = curve_fit(indice, sin_i, sin_r, sigma=di)
    for i in range(11):
        sigma_eff = np.sqrt(di**2 + (popt[0] * dr)**2)
        popt, pcov = curve_fit(indice ,sin_i, sin_r, sigma=sigma_eff)
        chisq = (((sin_r - indice(sin_i, *popt))/sigma_eff)**2).sum()
        p_value = chi2.sf(chisq, df=5)  
        print(f'Step {i}', popt, np.sqrt(pcov.diagonal()), f'chi-square =', chisq, f'p-value =', p_value)
    return sigma_eff, popt, chisq, p_value

def main():
    sin_i = np.array([ 23.5, 22.5, 18.5, 14.5, 8.5, 1.5])
    sin_r = np.array([ 37.0, 34.5, 29.0, 21.5, 12.0, 3.5]) 
    sin_i.sort()
    sin_r.sort()
    di = np.array([0.5]*6) 
    dr = np.array([0.5]*6) 

    #mm e errore
    incertezza_41 = 0.5 / 41
    di = np.sqrt((di/sin_i)**2 + incertezza_41**2) * sin_i
    dr = np.sqrt((dr/sin_r)**2 + incertezza_41**2) * sin_r
    sin_i = sin_i / 41
    sin_r = sin_r / 41
    di = di / 41
    dr = dr / 41


    sigma_eff, popt, chisq, p_value = fit_model(sin_i, sin_r, di, dr)
    plot_data(sin_i, sin_r, sigma_eff, popt, chisq, p_value)

if __name__ == "__main__":
    main()
