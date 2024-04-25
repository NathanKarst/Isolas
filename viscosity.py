import numpy as np


def viscosity_arrhenius(h,delta):
    return np.exp(delta)**h

def viscosity_pries_vitro(h,d):
    eta_plasma  = 1.3
    
    eta_star    = 220*np.exp(-1.3*d) + 3.2 - 2.44*np.exp(-0.06*d**(0.645))
    c           = (0.8 + np.exp(-0.075*d))*(-1 + 1/(1 + 10**(-11)*d**12)) + 1/(1 + 10**(-11)*d**12)
    eta         = 1 + (eta_star - 1)*((1 - h)**c - 1)/((1 - 0.45)**c - 1)
    eta         = eta*eta_plasma
    
    return eta

def viscosity_pries_vitro_deriv(h,d):
    eta_plasma  = 1.3
    
    eta_star    = 220*np.exp(-1.3*d) + 3.2 - 2.44*np.exp(-0.06*d**(0.645))
    c           = (0.8 + np.exp(-0.075*d))*(-1 + 1/(1 + 10**(-11)*d**12)) + 1/(1 + 10**(-11)*d**12)
    eta         = (1 + (eta_star - 1)*((1 - h)**c - 1)/ ((1 - 0.45)**c - 1))
    eta         = eta*eta_plasma
    
    detadh      = -eta_plasma*c*(eta_star - 1)*(1 - h)**(c-1)/ ((1 - 0.45)**c - 1)
    return detadh

def viscosity_pries_vivo(h,d):
    eta_plasma  = 1.3;

    eta_star    = 6*np.exp(-0.085*d) + 3.2 - 2.44*np.exp(-0.06*d**(0.645));
    c           = (0.8 + np.exp(-0.075*d))*(-1 + 1/(1 + 10**(-11)*d**12)) + 1/(1 + 10**(-11)*d**12);
    eta         = (1 + (eta_star - 1)*((1 - h)**c - 1)/ ((1 - 0.45)**c - 1) * (d/(d-1.1))**2)*(d/(d-1.1))**2;
    eta         = eta*eta_plasma;

    return eta

def viscosity_pries_vivo_deriv(h,d):
    eta_plasma  = 1.3;

    eta_star    = 6*np.exp(-0.085*d) + 3.2 - 2.44*np.exp(-0.06*d**(0.645));
    c           = (0.8 + np.exp(-0.075*d))*(-1 + 1/(1 + 10**(-11)*d**12)) + 1/(1 + 10**(-11)*d**12);
    eta         = (1 + (eta_star - 1)*((1 - h)**c - 1)/ ((1 - 0.45)**c - 1) * (d/(d-1.1))**2)*(d/(d-1.1))**2;
    eta         = eta*eta_plasma;
    
    detadh      = -eta_plasma*c*(eta_star - 1)*(1 - h)**(c-1)/ ((1 - 0.45)**c - 1)*(d/(d-1.1))**2*(d/(d-1.1))**2;
    return detadh