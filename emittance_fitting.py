import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
from scipy.optimize import curve_fit

pixel = 0.046*0.001         # pixel to m
L = 0.1                     # Quadrupole effective length, unit is m
E = 16.116                  # Beam energy, unit is MeV
gama = E/0.511 + 1          # gama factor
beta = np.sqrt(1-1/gama**2) # beta factor
d1 = 1.05                   # MQ01 to DV08, unit is m
d2 = 0.5                    # MQ02 to DV08, unit is m
d3 = 0.19                   # MQ03 to DV08, unit is m
D = d1                      # chose the right distance between the quadrupole to screen

quadrupole = 'quad1 scan'
direction = 'horizontal'
filename = 'P:\\experiments\\quadrupole scan\\' + quadrupole + '\\' + direction + '\\processing\\dataprocessing.csv'
savepath = 'P:\\experiments\\quadrupole scan\\' + quadrupole + '\\' + direction + '\\emitatnce no thin approximation modify.png'
data = pd.read_csv(filename, delimiter=';')
theta = data['theta(pixels)'] * pixel
modify_factor = np.abs(data['quadrupole parameter k (m^-2)'])*(-3.94e-4)+1.062
quad_stength = np.sqrt(np.abs(data['quadrupole parameter k (m^-2)'])) * L * modify_factor

# divergence
def div_fit(x,a,b,c):
    return (a*L**2/D**2+b*D**2/L**2+c)*np.cosh(x)**2+(2*a*L**2/D+c*D)/L*x*np.sinh(x)*np.cosh(x) \
           +(2*b*D/L+c*L/D)/x*np.sinh(x)*np.cosh(x)+a*(x**2)*(np.sinh(x)**2) \
           +b*(np.sinh(x)**2)/(x**2)+c*np.sinh(x)**2

def div_show(x,p_fit):
    a,b,c = p_fit.tolist()
    return (a*L**2/D**2+b*D**2/L**2+c)*np.cosh(x)**2+(2*a*L**2/D+c*D)/L*x*np.sinh(x)*np.cosh(x) \
           +(2*b*D/L+c*L/D)/x*np.sinh(x)*np.cosh(x)+a*(x**2)*(np.sinh(x)**2) \
           +b*(np.sinh(x)**2)/(x**2)+c*np.sinh(x)**2

# convergence
def con_fit(x,a,b,c):
    return (a*L**2/D**2+b*D**2/L**2-c)*np.cos(x)**2+(-a*2*L/D+c*D/L)*x*np.sin(x)*np.cos(x) \
           +(b*2*D/L-c*L/D)/x*np.sin(x)*np.cos(x)+a*x**2*np.sin(x)**2+b*np.sin(x)**2/x**2+c*np.sin(x)**2

def con_show(x,p_fit):
    a,b,c = p_fit.tolist()
    return (a*L**2/D**2+b*D**2/L**2-c)*np.cos(x)**2+(-a*2*L/D+c*D/L)*x*np.sin(x)*np.cos(x) \
           +(b*2*D/L-c*L/D)/x*np.sin(x)*np.cos(x)+a*x**2*np.sin(x)**2+b*np.sin(x)**2/x**2+c*np.sin(x)**2


param_bounds = ([],[])
p0 = []
p_fit, pcov = curve_fit(div_fit, quad_stength, theta**2)
sigma11 = p_fit[0]*L**2/D**2
sigma22 = p_fit[1]/L**2
sigma12 = p_fit[2]/(2*D)
emittance = 1/D*np.sqrt(p_fit[0]*p_fit[1]-1/4*p_fit[2]**2)

# emittance error
delta_a = 1/(2*D**2*emittance)*p_fit[1]
delta_b = 1/(2*D**2*emittance)*p_fit[0]
delta_c = 1/(2*D**2*emittance)*(-1/2*p_fit[2])
delta_emittance = np.sqrt(delta_a**2*pcov[0,0]+delta_b**2*pcov[1,1]+delta_c**2*pcov[2,2] \
                          +2*delta_a*delta_b*pcov[0,1]+2*delta_a*delta_c*pcov[0,2]+2*delta_b*delta_c*pcov[1,2])

# plot and save figure
emittance_text = '$\epsilon_N$ = ' + str(round(beta*gama*emittance*1e6,5)) + r'$\pm$' + str(round(beta*gama*delta_emittance*1e6,5))
x1 = np.arange(np.min(quad_stength),np.max(quad_stength), 0.001)
y1 = div_show(x1,p_fit)
plt.plot(x1, y1, 'r', label='fitting')
plt.scatter(quad_stength, theta**2, c='g', label='original data')
plt.xlabel(r'$\sqrt{k}$'+' ' + 'L')
plt.ylabel(r'$\sigma^2$'+'/ m')
plt.title(quadrupole.title()+'_'+direction+' '+'no thin approximation modify')
plt.text(0.3,5e-7,emittance_text,family='serif',fontsize=12, style='italic', ha='left', wrap=True)
plt.legend()
plt.savefig(savepath, dpi=300)
plt.show()