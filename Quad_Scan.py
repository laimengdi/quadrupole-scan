import pandas as pd
import numpy as np
import ntpath
import os
import gc
from matplotlib import pyplot as plt
from matplotlib import rc
from scipy.optimize import curve_fit
from matplotlib.patches import Ellipse

# load parameters
dataPath = 'data path.txt'
with open(dataPath,'r') as path:
    basic = path.readlines()

basicPath = basic[0]
basicMethod = basic[1]
basicMethod = basicMethod.strip('\t\n')
basicPath = basicPath.strip('\t\n')            # quadrupole current and beam spot sigma, unit: mA and mm
headPath, tailPath = ntpath.split(basicPath)
basicParaPath = headPath + '\\basic parameters.txt'
basicParameters = np.loadtxt(basicParaPath)
E = basicParameters[0]                        # Beam total energy, unit is MeV
L = basicParameters[1]                        # Quadrupole effective length, unit is m
D = basicParameters[2]                        # the distance between the quadrupole to screen
Q = basicParameters[3]                        # bunch charge, unit: pC
g = basicParameters[4]                        # quadrupole gradient, unit: (T/m)/A
K = 299.8*g/E                                 # quadrupole strength, unit: m^-2/A
gama = E/0.511                                # gama factor
beta = np.sqrt(1-1/gama**2)                   # beta factor
pixel = 0.046*0.001                           # pixel to m
sigma_quadstr = np.loadtxt(basicPath)
sigma = sigma_quadstr[:,1]*1e-3               # unit: m
quad_strength = np.sqrt(abs(sigma_quadstr[:,0])/1000 * K) * L
quad_strength_thin = sigma_quadstr[:,0]/1000 * K * L

bunchaCharge = str(Q) + ' pC'

saveImagePath = headPath + '\\' + basicMethod

# thin-lens approximation
def thin_fit(x,a,b,c):
    return a*x**2-b*x+c

def thin_show(x,p_fit):
    a,b,c = p_fit.tolist()
    return a*x**2-b*x+c

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

# phase space ellipse
def f_ellipse(a,b,c,d,e,f):
    ta = b**2 - 4*a*c
    x0 = (b*e-2*c*d)/ta
    y0 = (b*d-2*a*e)/ta
    r = a * pow(x0,2) + b * x0 * y0 + c * pow(y0,2) + f
    aa = np.sqrt(r / a)
    bb = np.sqrt(-4 * a * r / ta)
    t = np.linspace(0, 2 * np.pi, 60)
    A = np.array([[1,-b/(2*a)],[0,1]])
    B = np.zeros([2,60])
    for i in range(0,60):
        B[0,i] = aa*np.cos(t[i])
        B[1,i] = bb*np.sin(t[i])
    xy0 = A.dot(B)
    ellipse = np.zeros([2,xy0.shape[1]])
    ellipse[0,:] = xy0[0,:]-x0
    ellipse[1,:] = xy0[1,:]-y0
    return ellipse

# plot normalized circle
def f_circle(r):
    t = np.linspace(0, 2 * np.pi, 60)
    a = np.zeros([2,60])
    for i in range(0,60):
        a[0,i] = r * np.cos(t[i])
        a[1,i] = r * np.sin(t[i])
    return a


def main():
    if basicMethod == 'Thin-lens approximation':
        p_fit, pcov = curve_fit(thin_fit, quad_strength_thin, sigma ** 2)
        x1 = np.arange(np.min(quad_strength_thin), np.max(quad_strength_thin), 0.001)
        y1 = thin_show(x1, p_fit)
        xlabel = r'$\frac{1}{f}$ /'+ r'$m^{-1}$'
        ylabel = r'$\sigma^2$'+'/ $m^2$'
        sigma11 = p_fit[0] / D ** 2
        sigma12 = (p_fit[1] - 2 * D * sigma11) / (2 * D ** 2)
        sigma22 = (p_fit[2] - sigma11 - 2 * D * sigma12) / (D ** 2)
        emittance = np.sqrt(sigma11 * sigma22 - sigma12 ** 2)
        # emittance error
        delta_a = 1 / (D ** 4 * emittance) * p_fit[2]
        delta_b = 1 / (D ** 4 * emittance) * (-1 / 2 * p_fit[1])
        delta_c = 1 / (D ** 2 * emittance) * p_fit[0]
        delta_emittance = np.sqrt(delta_a ** 2 * pcov[0, 0] + delta_b ** 2 * pcov[1, 1] + delta_c ** 2 * pcov[2, 2] \
                                  + 2 * delta_a * delta_b * pcov[0, 1] + 2 * delta_a * delta_c * pcov[ 0, 2] \
                                  + 2 * delta_b * delta_c * pcov[1, 2])

        Twiss_beta = sigma11 / emittance
        Twiss_gama = sigma22 / emittance
        Twiss_alpha = -sigma12 / emittance

        ## phase space normalized ellipse
        R11 = 1-D*quad_strength_thin  # linear matrix elements
        R12 = D
        x0 = np.arange(-1, 1, 0.01)
        x0p = np.zeros([x0.shape[0], sigma.shape[0]])
        for i in range(0, sigma.shape[0]):
            x0p[:, i] = (sigma[i] - x0 * R11[i]*np.sqrt(Twiss_beta*emittance))*Twiss_beta\
                        / (R12*np.sqrt(Twiss_beta*emittance)) + Twiss_alpha*x0

        circle = f_circle(1)
        plt.figure(1)
        plt.plot(circle[0, :], circle[1, :], c='k', label='normalized ellipse')  # emittance from fitting
        for i in range(0, sigma.shape[0]):
            string = str(sigma_quadstr[i, 0]) + ' mA'
            plt.plot(x0, x0p[:, i], label=string)
        # plt.legend(loc='upper left')
        plt.xlabel(r'$\frac{x}{\sqrt{\beta\epsilon}}$')
        plt.ylabel(r'$\frac{\alpha x + \beta x^{\prime}}{\sqrt{\beta\epsilon}}$')
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        plt.savefig(saveImagePath+' phase space normalized ellipse.jpg',dpi=300, bbox_inches='tight')
        # plt.show()
        # print(beta*gama*emittance*1e6)

        ## phase space ellipse
        F_para = f_ellipse(sigma22, -2 * sigma12, sigma11, 0, 0, emittance ** 2)
        plt.figure(2)
        plt.plot(F_para[0, :], F_para[1, :], c='k')  # emittance ellipse
        plt.xlabel('Position / m')
        plt.ylabel('Angle / rad')
        plt.savefig(saveImagePath+' phase space ellipse.jpg', dpi=300, bbox_inches='tight')
        # plt.show()

        ## fitting image
        emittance_text = '$\epsilon_N$ = ' + str(round(beta * gama * emittance * 1e6, 3)) + r'$\pm$' \
                         + str(round(beta * gama * delta_emittance * 1e6, 3)) + ' ' + r'$\mu$m'
        textPositionX = quad_strength_thin[1]
        textPositionY = (sigma[0] ** 2 + sigma[1] ** 2) / 2
        plt.figure(3)
        plt.plot(x1, y1, 'r', label='fitting')
        plt.scatter(quad_strength_thin, sigma ** 2, c='g', label='original data')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.text(textPositionX, textPositionY, emittance_text, family='serif', fontsize=15)
        plt.legend()
        plt.savefig(saveImagePath + ' fitting.jpg', dpi=300, bbox_inches='tight')
        # plt.show()


    if basicMethod == 'Thick-lens divergence':
        p_fit, pcov = curve_fit(div_fit, quad_strength, sigma ** 2)
        x1 = np.arange(np.min(quad_strength), np.max(quad_strength), 0.001)
        y1 = div_show(x1, p_fit)
        xlabel = r'$\sqrt{k}$'+' ' + 'L'
        ylabel = r'$\sigma^2$'+'/ $m^2$'
        sigma11 = p_fit[0] * L ** 2 / D ** 2
        sigma22 = p_fit[1] / L ** 2
        sigma12 = -p_fit[2] / (2 * D)
        emittance = 1 / D * np.sqrt(p_fit[0] * p_fit[1] - 1 / 4 * p_fit[2] ** 2)
        ## emittance error
        delta_a = 1 / (D ** 2 * emittance_thick) * p_fit_thick[1]
        delta_b = 1 / (D ** 2 * emittance_thick) * p_fit_thick[0]
        delta_c = -1 / (2 * D ** 2 * emittance_thick) * p_fit_thick[2]
        delta_emittance = np.sqrt(delta_a ** 2 * pcov[0, 0] + delta_b ** 2 * pcov[1, 1]
                                        + delta_c ** 2 * pcov[2, 2] + 2 * delta_a * delta_b * pcov[0, 1]
                                        + 2 * delta_a * delta_c * pcov[0, 2] + 2 * delta_b * delta_c * pcov[1, 2])

    if basicMethod == 'Thick-lens convergence':
        p_fit, pcov = curve_fit(con_fit, quad_strength, sigma ** 2)
        x1 = np.arange(np.min(quad_strength), np.max(quad_strength), 0.001)
        y1 = con_show(x1, p_fit)
        xlabel = r'$\sqrt{k}$'+' ' + 'L'
        ylabel = r'$\sigma^2$'+'/ $m^2$'
        sigma11 = p_fit[0] * L ** 2 / D ** 2
        sigma22 = p_fit[1] / L ** 2
        sigma12 = -p_fit[2] / (2 * D)
        emittance = 1 / D * np.sqrt(p_fit[0] * p_fit[1] - 1 / 4 * p_fit[2] ** 2)
        ## emittance error
        delta_a = 1 / (D ** 2 * emittance) * p_fit[1]
        delta_b = 1 / (D ** 2 * emittance) * p_fit[0]
        delta_c = -1 / (2 * D ** 2 * emittance) * p_fit[2]
        delta_emittance = np.sqrt(delta_a ** 2 * pcov[0, 0] + delta_b ** 2 * pcov[1, 1]
                                        + delta_c ** 2 * pcov[2, 2] + 2 * delta_a * delta_b * pcov[0, 1]
                                        + 2 * delta_a * delta_c * pcov[0, 2] + 2 * delta_b * delta_c * pcov[1, 2])
        Twiss_beta = sigma11 / emittance
        Twiss_gama = sigma22 / emittance
        Twiss_alpha = -sigma12 / emittance

        ## normalized ellipse
        R11 = np.cos(quad_strength) - D * quad_strength / L * np.sin(quad_strength)  # linear matrix elements
        R12 = 1 / (quad_strength / L) * np.sin(quad_strength) + D * np.cos(quad_strength)
        x0 = np.arange(-1, 1, 0.01)
        x0p = np.zeros([x0.shape[0], sigma.shape[0]])
        for i in range(0, sigma.shape[0]):
            x0p[:, i] = (sigma[i] - x0 * R11[i]*np.sqrt(Twiss_beta*emittance))*Twiss_beta / \
                        (R12[i]*np.sqrt(Twiss_beta*emittance)) + Twiss_alpha*x0

        ## normalized circle
        circle = f_circle(1)
        plt.figure(1)
        plt.plot(circle[0, :], circle[1, :], c='k', label='normalized ellipse')  # emittance from fitting
        for i in range(0, sigma.shape[0]):
            string = str(sigma_quadstr[i, 0]) + ' mA'
            plt.plot(x0, x0p[:, i], label=string)
        # plt.legend(loc='lower left')
        plt.xlabel(r'$\frac{x}{\sqrt{\beta\epsilon}}$')
        plt.ylabel(r'$\frac{\alpha x + \beta x^{\prime}}{\sqrt{\beta\epsilon}}$')
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        plt.savefig(saveImagePath+' phase space normalized ellipse.jpg',dpi=300, bbox_inches='tight')
        # plt.show()

        ## phase space ellipse
        F_para = f_ellipse(sigma22, -2 * sigma12, sigma11, 0, 0, emittance ** 2)
        plt.figure(2)
        plt.plot(F_para[0, :], F_para[1, :], c='k')  # emittance ellipse
        plt.xlabel('Position / m')
        plt.ylabel('Angle / rad')
        plt.savefig(saveImagePath+' phase space ellipse.jpg', dpi=300, bbox_inches='tight')
        # plt.show()

        ## fitting image
        emittance_text = '$\epsilon_N$ = ' + str(round(beta * gama * emittance * 1e6, 3)) + r'$\pm$' \
                         + str(round(beta * gama * delta_emittance * 1e6, 3)) + ' ' + r'$\mu$' + 'm'
        textPositionX = quad_strength[1]
        textPositionY = (sigma[0] ** 2 + sigma[1] ** 2) / 2
        plt.figure(3)
        plt.plot(x1, y1, 'r', label='fitting')
        plt.scatter(quad_strength, sigma ** 2, c='g', label='original data')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.text(textPositionX, textPositionY, emittance_text, family='serif', fontsize=15)
        plt.legend()
        plt.savefig(saveImagePath + ' fitting.jpg', dpi=300, bbox_inches='tight')
        # plt.show()

    #
    #
    #
    # fitting_alpha = 1/2*np.arctan(-2*sigma12/(sigma22-sigma11))                         # angle
    # # print(fitting_alpha)
    #
    # # plot and save figure
    # emittance_text = '$\epsilon_N$ = ' + str(round(beta*gama*emittance*1e6,5)) + r'$\pm$'\
    #              + str(round(beta*gama*delta_emittance*1e6,5)) + ' ' + r'$\mu$' + 'm'
    #
    # textPositionX = quad_strength[1]
    # textPositionY = (sigma[0]**2 + sigma[1]**2)/2
    #
    # plt.figure(3)
    # plt.plot(x1, y1, 'r', label='fitting')
    # plt.scatter(quad_strength_thin, sigma ** 2, c='g', label='original data')
    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    # # plt.text(textPositionX, textPositionY, emittance_text, family='serif', fontsize=11, style='italic', ha='left',
    # #          wrap=True)
    # plt.legend()
    # # plt.savefig(saveFitPath, dpi=100, bbox_inches='tight')
    # plt.show()

    results = np.array([[beta*gama*emittance*1e6,beta*gama*delta_emittance*1e6]])
    resultsPath = saveImagePath + ' results.txt'
    np.savetxt(resultsPath,results)

if __name__ == '__main__':
    main()