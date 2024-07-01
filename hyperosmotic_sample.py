import numpy as np
import pandas as pd
import scipy.optimize
import matplotlib.pyplot as plt

###############################################################################
#Define functions for parameters fixed by intitial conditions
###############################################################################

def epsilon_t(lamp,g,rs,r0,kc,P):
    return g*rs + P**2*lamp*(rs**2)*3/2 +kc*(0.5*(1/rs-1/r0)**2+(rs-r0)/(r0*rs**2))-P*rs

def muL_t(kappa,lamp,g,rs,r0,kc,P):
    return kappa/(-kc*(1/rs-1/r0)**2-g*rs+2*epsilon_t(lamp,g,rs,r0,kc,P)-P**2*lamp*rs**2+P*rs)

def kappa_L(r0,lamp,muL,epsilon,g,r,kc,P):
    return muL*(2*epsilon-g*r-(kc*(1/r-1/r0)**2+(P**2)*lamp*r**2-P*r))

def kappa_R(r0,lamp,muR,epsilon,g,r,kc,P):
    return muR*(2*(epsilon-g*r)-(-kc*(1/r-1/r0)*2/r+kc*(1/r-1/r0)**2+3*(P**2)*lamp*r**2-P*2*r))

def L_exp(kappa):
    return 1.262*np.exp(0.3288*kappa)

def r_exp(kappa):
    return 0.1*kappa+0.195

###############################################################################
#Define functions to simulate osmotic shocks
###############################################################################

def osmotic_shock(ts,ts2,kappai,pf,muR):
    #timestep (h)
    dt=.001
    #parameters as seen in Table 1
    r0=0.3
    P=0.3
    P0=0.3
    lam0=0.18
    lamp=lam0/0.09
    g=1.64
    kc=0.03
    #initialize lists for simulation
    #time
    tList=[0]
    #growth rate
    kappaList=[kappai]
    #radius
    rList=[r_exp(kappai)]
    #length
    lList=[L_exp(kappai)]
    #division proteins
    xList=[0]
    #mu_L
    muLList=[muL_t(kappaList[-1],lamp,g,rList[-1],r0,kc,P0)]
    #epsilon
    epList=[epsilon_t(lamp,g,rList[-1],r0,kc,P0)]
    #pressure
    PList=[P]
    #Switch to track whether or not shock has occured
    shockSwitch=0
    #set runtime of simulation (h)
    while tList[-1]<0.8:
        #integrate variables of interest
        tList.append(tList[-1]+dt)
        muLList.append(muL_t(kappai,lamp,g,rList[0],r0,kc,P0)) 
        epList.append(epsilon_t(lamp,g,rList[0],r0,kc,P0)) 
        rList.append(rList[-1]*(1+dt*kappa_R(r0,lamp,muR,epList[-1],g,rList[-1],kc,P)))
        kappaList.append(kappa_L(r0,lamp,muLList[-1],epList[-1],g,rList[-1],kc,P))
        #check to see if division protein threshold has been reached
        if xList[-1]>2*3.1415*364*rList[-1]:
            #divide cell
            xList.append(0)
            lList.append(lList[-1]/2)
        else:
            #integrate length and division proteins as normal 
            lList.append((1+dt*kappaList[-1])*lList[-1])
            xList.append(xList[-1]+dt*lList[-1]*2*3.1415*364*r_exp(kappaList[-1])*kappaList[-1]/L_exp(kappaList[-1]))
        #check to see if it is time for the osmotic shock
        if tList[-1]>=ts and shockSwitch==0:
            #reassign pressure
            P=pf
            shockSwitch=1
        PList.append(P)
    #return lists of simulated parameters
    return tList,kappaList,rList,muLList,epList,lList,PList

###############################################################################
#Plot
###############################################################################

#define parameters for sample hyperosmotic shock trajectory
tList,kappaList,rList,muLList,epList,lList,PList=osmotic_shock(0.165,0.23254287,2.08,-.9,5)

fig1=plt.figure(1,figsize=(5, 5))
plt.plot(tList,kappaList,color='black')
#plt.plot(timeDataHyper,kappaDataHyper,'o',color='black')
plt.xlabel('t [h]')
plt.ylabel('\u03BA [$h^{-1}$]')

fig, ax1 = plt.subplots(figsize=(6, 5))

color = 'tab:red'
plt.margins(x=0)
ax1.set_xlabel('t [h]')
ax1.set_ylabel('R [\u03BCm]', color=color)
ax1.plot(tList, rList, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('L [\u03BCm]', color=color)  #, color=color we already handled the x-label with ax1
ax2.plot(tList, lList, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()

fig3=plt.figure(3,figsize=(5, 5))
plt.plot(tList,np.array(PList)-PList[0],color='black')
#plt.plot(timeDataHyper,kappaDataHyper,'o',color='black')
plt.xlabel('t [h]')
plt.ylabel('ΔP [$MPa$]')

plt.show  