import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
import matplotlib.pyplot as plt

###############################################################################
#Import experimental data
###############################################################################

data1=pd.read_csv(r'downshift2.csv')
data=pd.DataFrame.to_numpy(data1)
timeDataDown=[]
kappaDataDown=[]
for i in range(len(data)):
    timeDataDown.append(data[i,0])    
    kappaDataDown.append(data[i,1])

###############################################################################
#Define functions for parameters fixed by intitial conditions
###############################################################################

def epsilon_t(lam,g,rs,r0,kc):
    return g*rs + lam*rs**2*3/2 +kc*(0.5*(1/rs-1/r0)**2+(rs-r0)/(r0*rs**2))

def muL_t(kappa,lam,g,rs,r0,kc):
    return kappa/(-kc*(1/rs-1/r0)**2-g*rs+2*epsilon_t(lam,g,rs,r0,kc)-lam*rs**2)

def kappa_L(r0,lam,muL,epsilon,g,r,kc):
    return muL*(2*epsilon-g*r-(kc*(1/r-1/r0)**2+lam*r**2))

def kappa_R(r0,lam,muR,epsilon,g,r,kc):
    return muR*(2*(epsilon-g*r)-(-kc*(1/r-1/r0)*2/r+kc*(1/r-1/r0)**2+3*lam*r**2))

def L_exp(kappa):
    return 1.262*np.exp(0.3288*kappa)

def r_exp(kappa):
    return 0.1*kappa+0.195

###############################################################################
#Define functions to simulate downshift
###############################################################################

#functional form for epsilon and mu_L as seen in eq. 4.1 
def logistic(t,ts,alpha,theta):
    return 1+alpha/(1+np.exp(-theta*(t-ts)))

#integration of our differential equations during the nutrient shift
def nutrient_shift(ts,kappai,kappaf,theta,muR,g):
    #timestep (h)
    dt=0.001
    #parameters as seen in Table 1
    r0=0.3
    lam=0.18
    kc=0.03
    #initialize list for simulation
    #time
    tList=[-3]
    #growth rate
    kappaList=[kappai]
    #radius
    rList=[r_exp(kappai)]
    #mu_L
    muLList=[muL_t(kappaList[-1],lam,g,rList[-1],r0,kc)]
    #epsilon
    epList=[epsilon_t(lam,g,rList[-1],r0,kc)]
    #g
    gList=[g]
    #calculate fractional change in initial conditions due to the shift
    alphamu=(muL_t(kappaf,lam,g,r_exp(kappaf),r0,kc)-muLList[-1])/muLList[-1]
    alphaepsilon=(epsilon_t(lam,g,r_exp(kappaf),r0,kc)-epList[-1])/epList[-1]
    #set runtime of simulation (h)
    while tList[-1]<4.5:
        #integrate variables of interest
        tList.append(tList[-1]+dt)
        muLList.append(logistic(tList[-1],ts,alphamu,theta)*muLList[0]) 
        epList.append(logistic(tList[-1],ts,alphaepsilon,theta)*epList[0])        
        rList.append(rList[-1]*(1+dt*kappa_R(r0,lam,muR,epList[-1],gList[-1],rList[-1],kc)))
        kappaList.append(kappa_L(r0,lam,muLList[-1],epList[-1],gList[-1],rList[-1],kc)) 
    #return lists of simulated parameters
    return tList,kappaList,rList,muLList,epList,gList

#function to minimize to fit experimental data
def optimize_this(arg):
    #minimization arguments
    ts,kappai,kappaf,theta,muR,g=arg
    #run the simulation for a set of arguments
    tList,kappaList,rList,muLList,epList,gList=nutrient_shift(ts,kappai,kappaf,theta,muR,g)
    #perform a least squares minimization between the simulated trajectory and the data
    loss=0
    dataMatchIndices=[]
    dataIndexTracker=0
    for n in tList:
        if dataIndexTracker<len(timeDataDown):
            if n>timeDataDown[dataIndexTracker]:
                dataMatchIndices.append(tList.index(n))
                dataIndexTracker+=1
    for m in range(len(kappaDataDown)):
        loss+=(kappaDataDown[m]-kappaList[dataMatchIndices[m]])**2
        #add extra weight to important features in the data (here we weight the points closest to the shift)
        if m in [6,7,8]:
           loss+=10*(kappaDataDown[m]-kappaList[dataMatchIndices[m]])**2
    #condition on the minimizer to not make the nutrient import rate negative
    if min(epList)<0.01:
        loss+=1000000000000
    return loss

###############################################################################
#Fit the model to data
###############################################################################

                                      #ts,kappai,kappaf,thetamuL,thetaep,muR,g
solDown=scipy.optimize.minimize(optimize_this,
                                [0,0.001,0.38,10,0.0006,3],
                                bounds=((-.5,-.08),(kappaDataDown[0]-0.005,kappaDataDown[0]+0.005),(0.38,.43),(0,240),(0.001,0.4),(0.1,4)),
                                method='nelder-mead',
                                options={'maxfev': 20000})
print(solDown.message)
print(solDown.x)

###############################################################################
#Plot resulting fit
###############################################################################

tList,kappaList,rList,muLList,epList,gList=nutrient_shift(solDown.x[0],solDown.x[1],
                                                         solDown.x[2],solDown.x[3],
                                                         solDown.x[4],solDown.x[5])


plt.rcParams.update({'font.size': 16})

fig1=plt.figure(1,figsize=(5, 5))
plt.margins(x=0)
plt.plot(tList,kappaList,color='black')
plt.plot(timeDataDown,kappaDataDown,'o',color='black')
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
#ax2.set_ylabel('$\textcolor{blue}{e^{-x/5}} + \textcolor{green}{e^{-x/1}}$')  #, color=color we already handled the x-label with ax1
ax2.set_ylabel('$\u03B5/\u03B5_i$', color='blue')
ax2.plot(tList, epList/epList[0], color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.show  