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
    
'''    
def LensEq2(xm, ym, Dls, Re, Mwd, Rwd):
    flag=-1; 
    b=np.sqrt(xm**2.0+ym**2.0)+1.0e-50#[m] Impact parameter
    angle=float(4.0*G*Mwd*Msun/(velocity*velocity*b))#radian  
    tant= float(b/Dl)## tan(theta)
    tana=float(np.tan(angle))## tan(alpha)
    tanb=tant-(tant + float(tana-tant)/(1.0+ tant*tana))*Dls/(Dl+Dls)
    beta= float(tanb*Dl)#[m]
    xs=   float(beta*xm/b)/Re#[RE]
    ys=   float(beta*ym/b)/Re#[RE]
    d2=   b*b/(Re*Re)
    xs0=  float(xm/Re-xm/Re/d2)
    ys0=  float(ym/Re-ym/Re/d2)
    if(float(angle*180.0/np.pi)>10.0 or abs(xs-xs0)>5.0 or abs(ys-ys0)>5.0): 
        print("deflection angle: ", angle*180.0/np.pi)
        print("new ccordinate : ",  xs,    ys)
        print("old coordinate:  ",  xs0,  ys0)
    if(b<float(Rwd*Rsun) or b==float(Rwd*Rsun) ): flag=0;
    else:                flag=1;
    #print (Mwd,  Rwd, b, angle,  xm/Re, ym/Re, xs0, ys0, xs, ys)
    #input("Enter a number ")
    return(xs, ys, xs0, ys0, flag)



def InRayShoot(Rho, Xsc, Ysc, Dls, Re, Mwd, Rwd):
    image1=0.0;   obscur1=0.0 
    image2=0.0;   obscur2=0.0 
    for i in range(nx):
        for j in range(ny):
            xi=float(i-nx/2.0)*dx+dx/1.996834354+0.00527435617465##[RE]
            yi=float(j-ny/2.0)*dy+dy/1.996834354+0.00527435617465##[RE]
            xsi,ysi, xsi0, ysi0,Flag=LensEq2(xi*Re, yi*Re, Dls, Re, Mwd, Rwd)
            diss=np.sqrt((xsi-Xsc)**2.0+(ysi-Ysc)**2.0)
            if(diss<Rho or diss==Rho):
                #mu=np.sqrt(1.0-diss**2.0/Rho**2.0) 
                ff=1.0#*abs(1.0-limb*abs(1.0-mu))  
                image1+=ff
                if(Flag==0): obscur1+=ff
            diss=np.sqrt((xsi0-Xsc)**2.0+(ysi0-Ysc)**2.0)
            if(diss<Rho or diss==Rho):
                #mu=np.sqrt(1.0-diss**2.0/Rho**2.0) 
                ff=1.0#*abs(1.0-limb*abs(1.0-mu))  
                image2+=ff
                if(Flag==0): obscur2+=ff
    return(image1, image2, obscur1, obscur2)                   
'''    
        
################################################################################

thrx=[]
thry=[]
n1=100
n2=100
Magni=np.zeros((n2, n1))
Rinw=np.zeros((n2, n1))
Rwre=np.zeros((n2, n1))
obsc=np.zeros((n2, n1))
Rwr1=np.zeros((n2, n1))
REE= np.zeros((n2, n1))
#Delta1=np.zeros((n2, n1))
#Delta2=np.zeros((n2, n1))
#Delta3=np.zeros((n2, n1))


period=float(500.0*24.0*3600.0)#[s]

#fil0=open("Amax{0:d}_{1:d}.dat".format(int(period/24.0/3600.0) , int(Dl*1000.0/KPC) ),"w")
#fil0.close();

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
        if(Rwd<Rin): 
            flag=0;  
            obsc[j,i]=float(0.0)
            Rwr1[j,i]=float(0.0)
        else:        
            flag=1;
            obsc[j,i]=float(Rwd*Rwd-Rin*Rin)*100.0/(Rstarp*Rstarp) 
            Rwr1[j,i]=float(Rwd*Rwd-Rin*Rin)*10000.0
        if(abs(Rwd/RE-np.sqrt(2.0))<0.05): 
            thrx.append(i)
            thry.append(j)
        Amax= float(2.0/(Rho*Rho)-flag*(Rwd*Rwd- Rin*Rin)/(Rstarp*Rstarp))*100.0;  
        #image1, image2, obscur1, obscur2=InRayShoot(Rho, 0.0, 0.0, semi, RE*Rsun, Mwd, Rwd)
        #Astar1=float(image1/Fbase)*100.0-float(obscur1/Fbase)*100.0-100.0##accurate
        #Astar2=float(image2/Fbase)*100.0-float(obscur2/Fbase)*100.0-100.0
        print(Rstar, Rstarp,  RE, semi/AU,  Amax,  Rwd/Rin)
        print("*************************************************************")
        Magni[j,i]=Amax
        Rinw[j,i]=np.log10(Rwd/Rin)
        Rwre[j,i]=float(Rwd/RE)
        REE[j,i] = float(RE)
        #Delta1[j,i]=float(Amax-Astar1)
        #Delta2[j,i]=float(Amax-Astar2)
        #Delta3[j,i]=float(Astar1-Astar2)
        #fil0=open("Amax{0:d}_{1:d}.dat".format(int(period/24.0/3600.0) , int(Dl*1000.0/KPC) ),"a+")
        #par=np.array([i,j,Mwd, Mstar, Amax, Astar1, Astar2, np.log10(Rwd/Rin) ])
        #fil0.close();

################################################################################
'''
###    MAGNIFICATION
plt.cla()
plt.clf()
fig=plt.figure(figsize=(7,6))
ax= plt.gca()  
plt.imshow(Magni,cmap=cmap,interpolation='nearest',aspect='equal', origin='lower')
plt.plot(thrx, thry, "m*", markersize=3)
#plt.scatter(Flag,cmap=discrete_cmap(n1*n2, 'cubehelix'),interpolation='nearest',aspect='equal', origin='lower')
plt.clim()
minn=np.min(Magni)
maxx=np.max(Magni)
step=float((maxx-minn)/(tt-1.0));
for m in range(tt):
    v[m]=round(float(minn+m*step),1)
    print(v[m], m)
cbar=plt.colorbar(orientation='vertical',shrink=0.9,pad=0.01, ticks=v)
cbar.ax.tick_params(labelsize=17)
cbar.set_label(r"$\Delta A_{\rm{m}}(\times 100)$", rotation=270, fontsize=21, labelpad=20.0)
contours=plt.contour(Magni, 10,  colors='black')
plt.clabel(contours, inline=5, fontsize=17)
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
fig.savefig("mapp{0:d}.jpg".format(int(period/24.0/3600.0)), dpi=200)     
    
################################################################################
####  RWD/RIn
plt.cla()
plt.clf()
fig=plt.figure(figsize=(7,6))
ax= plt.gca()  
plt.imshow(Rinw,cmap=cmap,interpolation='nearest',aspect='equal', origin='lower')
#plt.plot(thrx, thry, "m*", markersize=3)
plt.clim()
minn=np.min(Rinw)
maxx=np.max(Rinw)
step=float((maxx-minn)/(tt-1.0));
for m in range(tt):
    v[m]=round(float(minn+m*step),1)
    print(v[m], m)
cbar=plt.colorbar(orientation='vertical',shrink=0.9,pad=0.01, ticks=v)
cbar.ax.tick_params(labelsize=17)
cbar.set_label(r"$\log_{10}[R_{\rm{WD}}/R_{1}]$", rotation=270, fontsize=21, labelpad=20.0)
contours=plt.contour(Rinw, 10,  colors='black')
plt.clabel(contours, inline=5, fontsize=17)
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
fig.savefig("Rinw{0:d}.jpg".format(int(period/24.0/3600.0) ), dpi=200)     
    
################################################################################

####  RWD/RE
plt.cla()
plt.clf()
fig=plt.figure(figsize=(7,6))
ax= plt.gca()  
plt.imshow(Rwre,cmap=cmap,interpolation='nearest',aspect='equal', origin='lower')
#plt.plot(thrx, thry, "m*", markersize=3)
plt.clim()
minn=np.min(Rwre)
maxx=np.max(Rwre)
step=float((maxx-minn)/(tt-1.0));
for m in range(tt):
    v[m]=round(float(minn+m*step),1)
    print(v[m], m)
cbar=plt.colorbar(orientation='vertical',shrink=0.9,pad=0.01, ticks=v)
cbar.ax.tick_params(labelsize=17)
cbar.set_label(r"$\log_{10}[R_{\rm{WD}}/R_{\rm{E}}]$", rotation=270, fontsize=21, labelpad=20.0)
contours=plt.contour(Rwre, 10,  colors='black')
plt.clabel(contours, inline=5, fontsize=17)
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
fig.savefig("Rwre{0:d}.jpg".format(int(period/24.0/3600.0) ), dpi=200)     
    
################################################################################

#### Obscuration
plt.cla()
plt.clf()
fig=plt.figure(figsize=(7,6))
ax= plt.gca()  
plt.imshow(obsc,cmap=cmap,interpolation='nearest',aspect='equal', origin='lower')
#plt.plot(thrx, thry, "m*", markersize=3)
plt.clim()
minn=np.min(obsc)
maxx=np.max(obsc)
step=float((maxx-minn)/(tt-1.0));
for m in range(tt):
    v[m]=round(float(minn+m*step),1)
    print(v[m], m)
cbar=plt.colorbar(orientation='vertical',shrink=0.9,pad=0.01, ticks=v)
cbar.ax.tick_params(labelsize=17)
cbar.set_label(r"$\mathcal{O}\times 100$", rotation=270, fontsize=21, labelpad=20.0)
contours=plt.contour(obsc, 10,  colors='black')
plt.clabel(contours, inline=5, fontsize=17)
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
fig.savefig("Obsc{0:d}.jpg".format(int(period/24.0/3600.0) ), dpi=200)     
    


################################################################################


#### Obscuration Net
plt.cla()
plt.clf()
fig=plt.figure(figsize=(7,6))
ax= plt.gca()  
plt.imshow(Rwr1,cmap=cmap,interpolation='nearest',aspect='equal', origin='lower')
#plt.plot(thrx, thry, "m*", markersize=3)
plt.clim()
minn=np.min(Rwr1)
maxx=np.max(Rwr1)
step=float((maxx-minn)/(tt-1.0));
for m in range(tt):
    v[m]=round(float(minn+m*step),1)
    print(v[m], m)
cbar=plt.colorbar(orientation='vertical',shrink=0.9,pad=0.01, ticks=v)
cbar.ax.tick_params(labelsize=17)
cbar.set_label(r"$R_{\rm{WD}}^{2}-R_{1}^{2}$", rotation=270, fontsize=21, labelpad=20.0)
contours=plt.contour(Rwr1, 10,  colors='black')
plt.clabel(contours, inline=5, fontsize=17)
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
fig.savefig("ObsN{0:d}.jpg".format(int(period/24.0/3600.0) ), dpi=200)     
    


################################################################################
'''




### RE
plt.cla()
plt.clf()
fig=plt.figure(figsize=(7,6))
ax= plt.gca()  
plt.imshow(REE,cmap=cmap,interpolation='nearest',aspect='equal', origin='lower')
#plt.plot(thrx, thry, "m*", markersize=3)
plt.clim()
minn=np.min(REE)
maxx=np.max(REE)
step=float((maxx-minn)/(tt-1.0));
for m in range(tt):
    v[m]=round(float(minn+m*step),1)
    print(v[m], m)
cbar=plt.colorbar(orientation='vertical',shrink=0.9,pad=0.01, ticks=v)
cbar.ax.tick_params(labelsize=17)
cbar.set_label(r"$R_{\rm{E}}(R_{\odot})$", rotation=270, fontsize=21, labelpad=20.0)
contours=plt.contour(REE, 10,  colors='black')
plt.clabel(contours, inline=5, fontsize=17)
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
fig.savefig("RE{0:d}.jpg".format(int(period/24.0/3600.0) ), dpi=200)     
    


################################################################################
















