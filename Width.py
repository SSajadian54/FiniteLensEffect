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
#dx=float(0.0875)
#dy=float(0.0875)

################################################################################

def tickfun(x, start, dd0, rho):
    return((start+x*dd0)/rho)
    

def tickfun2(x, start, dd0, rho):
    return( start+x*dd0/rho ) 
    
################################################################################

def Plot(width, perid, n1, n2):
    plt.cla()
    plt.clf()
    fig=plt.figure(figsize=(7,6))
    ax= plt.gca()  
    plt.imshow(width,cmap=cmap,interpolation='nearest',aspect='equal', origin='lower')
    plt.clim()
    minn=np.min(width)
    maxx=np.max(width)
    step=float((maxx-minn)/(tt-1.0));
    for m in range(tt):
        v[m]=round(float(minn+m*step),1)
        #print(v[m], m)
    cbar=plt.colorbar(orientation='vertical',shrink=0.9,pad=0.01, ticks=v)
    cbar.ax.tick_params(labelsize=17)
    cbar.set_label(r"$\rm{Width}(\rm{hours})$", rotation=270, fontsize=21, labelpad=20.0)
    contours=plt.contour(width, 10,  colors='black')
    plt.clabel(contours, inline=5, fontsize=17)
    plt.title(r"$T(\rm{days})=$"+str(int(perid/24.0/3600.0)),fontsize=21)    
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
    fig.savefig("Width{0:d}.jpg".format(int(perid/24.0/3600.0)), dpi=200)     
    print ("****************Plott:  ", int(perid/24.0/3600.0)) 
################################################################################
n1=100
n2=100

peri=np.array([3.0,5.0,10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0,130.0,150.0,180.0,200.0,300.0,400.0,500.0,600.0,700.0,800.0])
print ("peri, ", peri, len(peri)    )
minp=np.zeros(( len(peri) ))
maxp=np.zeros(( len(peri) ))
avep=np.zeros(( len(peri) ))        



for k in range(22): 
    width=np.zeros((n2, n1))
    period=float(peri[k]*24.0*3600.0)
    print ("period:  ", period, k, peri[k])
    for i in range(n1): 
        for j in range(n2): 
            Mwd  =float(0.1+1.3*i/n1)#[Msun]
            Mstar=pow(10.0,-1.096+1.16*j/n2)#[Msun] 
            Rstar=np.power(Mstar,0.8)#[Rsun]
            Rwd=0.01*np.power(1.0/Mwd,1.0/3.0)#[Rsun]
            semi=np.power(period*period*G*(Mwd+Mstar)*Msun/(4.0*np.pi*np.pi),1.0/3.0)##[m]
            #RE=np.sqrt(4.0*G*Mwd*Msun*semi)/(velocity*Rsun)#[Rsun]
            Rstarp=Rstar*Dl/(Dl+semi)#[Rsun]
            width[j,i]=np.abs(np.abs(np.arccos(Rstarp*Rsun/semi))*180.0/np.pi-90.0)*2.0*period/(360.0*3600.0)## hours
    Plot(width,period, n1, n2)
    minp[k]=np.min(width)
    maxp[k]=np.max(width)   
    avep[k]=np.mean(width)   
    print("Width, min, max:  ", width,   np.min(width),  np.max(width) , minp[k], maxp[k]  )        

################################################################################


plt.cla()
plt.clf()
fig=plt.figure(figsize=(8,6))
ax1=fig.add_subplot(111)
plt.plot(peri, maxp, "k-.",lw=1.5, label=r"$\rm{Maximum}$")
plt.plot(peri, minp, "k-.",lw=1.5, label=r"$\rm{Minimum}$")
plt.plot(peri, avep, "k--",lw=2.5, label=r"$\rm{Average}$")

#plt. axhline(y=60.0/60.0, color='g', linestyle='--', linewidth=1.8)##, label=r"$$") 
ax1.fill_between(peri, minp, maxp, color="b", alpha=0.2)
plt.xscale('log')
plt.xlim(3.0,800.0)
plt.ylim(0.0,16.0)
plt.xticks(fontsize=17, rotation=0)
plt.yticks(fontsize=17, rotation=0)
plt.xlabel(r"$\rm{Period}(\rm{days})$", fontsize=18)
plt.ylabel(r"$\rm{Width}(\rm{hours})$", fontsize=18)
ax1.grid("True")
ax1.grid(linestyle='dashed')
ax1.legend(prop={"size":14.5})
fig=plt.gcf()
fig.savefig("./UpDown.jpg",dpi=200)

