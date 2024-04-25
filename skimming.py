import numpy as np

def skimming_pries(q,h_feed,d_feed,d_alpha,d_beta):
#     q += 1e-16*(q >= 0) - 1e-16*(q < 0)
    x_0 = 0.4/d_feed;
    a   = -6.96/d_feed*np.log(d_alpha/d_beta);
    b   = 1 + 6.98*(1-h_feed)/d_feed;

    f = np.zeros(np.shape(q))    
    f = 0*(q < x_0) + (1 + np.exp(-a - b*np.log(abs((q-x_0)/(1-q-x_0)))))**(-1)*(q >= x_0)*(q <= 1 - x_0) + 1*(q > 1-x_0)
    
    f   = f/q
    g   = (1 - f*q)/(1-q)

    return f,g

def skimming_pries_dq(q,h_feed,d_feed,d_alpha,d_beta):
    x_0 = 0.4/d_feed;
    a   = -6.96/d_feed*np.log(d_alpha/d_beta);
    b   = 1 + 6.98*(1-h_feed)/d_feed;

    f,g = skimming_pries(q,h_feed,d_feed,d_alpha,d_beta)
    
    dfdq = np.zeros(np.shape(q))
    dfdq = -f**2*(1 + np.exp(-a - b*np.log(abs((q-x_0)/(1-q-x_0)))) - b*q*np.exp(-a)*(np.abs((q-x_0)/(1-q-x_0)))**(-b-1)*(1-2*x_0)/(1-q-x_0)**2)*(q >= x_0)*(q <= 1 - x_0)
    dfdq += -1/q**2*(q > 1 - x_0)   
    
    dgdq = np.zeros(np.shape(q))
    dgdq = (1-f*q + (-f-q*dfdq)*(1-q))/(1-q)**2
    
    return dfdq,dgdq

def skimming_pries_dh(q,h_feed,d_feed,d_alpha,d_beta):
    x_0 = 0.4/d_feed;
    a   = -6.96/d_feed*np.log(d_alpha/d_beta);
    b   = 1 + 6.98*(1-h_feed)/d_feed;
    

    f,g = skimming_pries(q,h_feed,d_feed,d_alpha,d_beta)
    
    dfdh = np.zeros(np.shape(h_feed))
    dfdh = f**2*q*np.exp(-a)*np.log(abs((q-x_0)/(1-q-x_0)))*abs((q-x_0)/(1-q-x_0))**(-b)*-6.98/d_feed*(q >= x_0)*(q <= 1 - x_0)
    dgdh = -dfdh*q/(1-q)
    
    return dfdh, dgdh


def skimming_kj(q,p):
    f = q**(p-1)/(q**p + (1-q)**p)
    g = (1-q)**(p-1)/(q**p + (1-q)**p)
    
    return (f,g)

def skimming_kj_dq(q,p):
    num = q**(p-1)
    den = q**p + (1-q)**p 
    
    dnum = (p-1)*q**(p-2)
    dden = p*q**(p-1) - p*(1-q)**(p-1)
    
    dfdq = (dnum*den - num*dden)/den**2

    num = (1-q)**(p-1)
    dnum = -(p-1)*(1-q)**(p-2)
    
    dgdq =  (dnum*den - num*dden)/den**2   
    
    return (dfdq,dgdq)

def skimming_kj_dh(q,p):
    return (np.zeros(np.shape(q)),np.zeros(np.shape(q)))