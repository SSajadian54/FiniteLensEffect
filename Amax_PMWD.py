import numpy as np 
from numpy import conj
import matplotlib.pyplot as plt
import matplotlib
import pylab as py 
from matplotlib import rcParams
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
rcParams["font.size"] = 11.5
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Computer Modern Sans"]
rcParams["text.usetex"] = True
rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"
rcParams['text.usetex'] = True
matplotlib.rc('text', usetex=True)
rcParams["text.latex.preamble"].join([r"\usepackage{dashbox}",r"\setmainfont{xcolor}",])
cmap=plt.get_cmap('viridis')
import scipy.special as ss
import warnings
warnings.filterwarnings("ignore")
################################################################################
G= 6.67430*pow(10.0,-11.0)
AU=1.495978707*pow(10.0,11)
Msun=1.989*pow(10.0,30.0)
Rsun =6.9634*pow(10.0,8.0)
KPC= 3.0857*pow(10.0,19.0)## meter 
velocity=299792458.0##m/s
Dl=1.0*KPC
tt=int(9)
cmap=plt.get_cmap('viridis')
v=   np.zeros((tt))

################################################################################
def tickfun(x, start, dd0, rho):
    return((start+x*dd0)/rho)
    

def tickfun2(x, start, dd0, rho):
    return( start+x*dd0/rho ) 
   
################################################################################
n1=100
n2=100
Magni=np.zeros((n2, n1))
Rinw=np.zeros((n2, n1))
fil0=open("Cancel.dat","w")
fil0.close();
peri=np.zeros((50,2))


ddt=np.log10(800.0)-np.log10(3.0)

for k in range(50): 
    period=pow(10.0, float(np.log10(3.0)+ddt*k/50.0) )*24.0*3600.0#[s]
    thrx=[]
    thry=[]
    for i in range(n1): 
        for j in range(n2): 
            Mwd  =float(0.1+1.3*i/n1)#[Msun]
            Mstar=pow(10.0,-1.096+1.16*j/n2)#[Msun] 
            Rstar=np.power(Mstar,0.8)#[Rsun]
            Rwd=0.01*np.power(1.0/Mwd,1.0/3.0)#[Rsun]
            semi= np.power(period*period*G*(Mwd+Mstar)*Msun/(4.0*np.pi*np.pi),1.0/3.0)##[m]
            RE=np.sqrt(4.0*G*Mwd*Msun*semi)/(velocity*Rsun)#[Rsun]
            Rstarp=Rstar*Dl/(Dl+semi)#[Rsun]
            Rho=Rstarp/RE
            Rin= abs(np.sqrt(Rstarp*Rstarp+4.0*RE*RE)-Rstarp)/2.0#[Rsun]  
            #xsiz=float(2.15*Rho)#[RE]
            #ysiz=float(2.15*Rho)#[RE]
            #nx = int(xsiz/dx+1.0); 
            #ny = int(ysiz/dy+1.0); 
            #Fbase=float(np.pi*Rho*Rho*(1.0-0.0/3.0)/(dx*dy))      
            #print ("nx, ny, dx, dy, rho:  ",  nx, ny, dx, dy,   Rho, Mwd,   Rwd)
            if(Rwd<Rin): flag=0;  
            else:        flag=1; 
            if(abs(Rwd/RE-1.415)<0.03): 
                thrx.append(i)
                thry.append(j)
                peri[k,1]+= Mwd
            Amax= float(2.0/(Rho*Rho)-flag*(Rwd*Rwd- Rin*Rin)/(Rstarp*Rstarp))*100.0;  
            print(Rstar, Rstarp,  RE, semi/AU,  Amax,  Rwd/Rin)
    print("*************************************************************")
    peri[k,1]=peri[k,1]/float(len(thrx))## MWD
    peri[k,0]=period/(24.0*3600.0)## days     
    fil0=open("Cancel.dat","a+")
    np.savetxt(fil0,peri[k,:].reshape((-1,2)),fmt ="%.7f    %.8f")
    fil0.close();

################################################################################


plt.cla()
plt.clf()
fig=plt.figure(figsize=(8,6))
plt.plot(peri[:,0], peri[:,1], 'g--', lw=2.5)##,label=r"$\rm{Eclipsing}~\rm{effect}$")
plt.xlabel(r"$T(\rm{days})$", fontsize=20)
plt.ylabel(r"$\overline{M_{\rm{WD}}}(M_{\odot})$", fontsize=20)
plt.xlim([ 1.0, 700.0])
plt.ylim([ 0.08, 0.8 ])
#plt.xscale('log')          
plt.xticks(fontsize=20, rotation=0)
plt.yticks(fontsize=20, rotation=0)
plt.title(r"$R_{\rm{WD}}=\sqrt{2}~R_{\rm{E}}$", fontsize=20)
plt.grid()
plt.grid(linestyle='dashed')
#plt.legend()
#plt.legend(prop={"size":16.0}, loc='best')
fig=plt.gcf()
fig.tight_layout()
plt.savefig("Cancel2.jpg" , dpi=200)


















