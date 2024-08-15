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


def tickfun(x, start, dd0, rho):
    return((start+x*dd0)/rho)
    

def tickfun2(x, start, dd0, rho):
    return( start+x*dd0/rho ) 
       
    
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
dx=float(0.0875)
dy=float(0.0875)

thrx=[]
thry=[]
n1=70
n2=70
Magni=np.zeros((n2, n1))
Rinw=np.zeros((n2, n1))
Delta1=np.zeros((n2, n1))
Delta2=np.zeros((n2, n1))
Delta3=np.zeros((n2, n1))

period=float(30.0*24.0*3600.0)#[s]


k=0
par=np.zeros((n1*n2, 8))
par=np.loadtxt("Amax{0:d}_{1:d}.dat".format(int(period/24.0/3600.0) , int(Dl*1000.0/KPC)) )
for i in range(n1): 
    for j in range(n2):
        Delta1[j,i]=float(par[k,4]-par[k,5])
        k+=1
        
        
        
        

plt.cla()
plt.clf()
fig=plt.figure(figsize=(7,6))
ax= plt.gca()  
plt.imshow(Delta1,cmap=cmap,interpolation='nearest',aspect='equal', origin='lower')
#plt.plot(thrx, thry, "m*", markersize=3)
#plt.scatter(Flag,cmap=discrete_cmap(n1*n2, 'cubehelix'),interpolation='nearest',aspect='equal', origin='lower')
plt.clim()
minn=np.min(Delta1)
maxx=np.max(Delta1)
step=float((maxx-minn)/(tt-1.0));
for m in range(tt):
    v[m]=round(float(minn+m*step),1)
    print(v[m], m)
cbar=plt.colorbar(orientation='vertical',shrink=0.9,pad=0.01, ticks=v)
cbar.ax.tick_params(labelsize=17)
cbar.set_label(r"$\rm{Error}\times 100$", rotation=270, fontsize=21, labelpad=20.0)
#contours=plt.contour(Delta1, 10,  colors='black')
#plt.clabel(contours, inline=5, fontsize=17)
plt.title( r"$T(\rm{days})=$"+str(int(period/24.0/3600.0)),fontsize=21)    
plt.xticks(fontsize=20, rotation=0)
plt.yticks(fontsize=20, rotation=0)
plt.xlim(0.0,n1)
plt.ylim(0.0,n2)
ticc=np.array([ int(n1*0.1), int(n1*0.3), int(n1*0.5), int(n1*0.7), int(n1*0.9) ])
ax.set_xticks(ticc,labels=[round(j,1) for j in tickfun( ticc,float(0.1), float(1.3/n1), 1.0)])
ax.set_yticks(ticc,labels=[round(j,1) for j in tickfun2(ticc,float(-1.096),float(1.16/n2), 1.0)])
ax.set_aspect('equal', adjustable='box')
plt.xlabel(r"$M_{\rm{WD}}(M_{\odot})$",fontsize=21,labelpad=0.05)
plt.ylabel(r"$\log_{10}[M_{\star}(M_{\odot})]$" ,fontsize=21,labelpad=0.05)
fig=plt.gcf()
fig.tight_layout(pad=0.15)
fig.savefig("RError{0:d}_{1:d}.jpg".format(int(period/24.0/3600.0), int(Dl*1000.0/KPC)), dpi=200)     
    
        
    
    










