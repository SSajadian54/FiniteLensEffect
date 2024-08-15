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
#import VBBinaryLensingLibrary as vb
#VBB=vb.VBBinaryLensing()
#VBB.Tol=1.0e-4;
#VBB.SetLDprofile(VBB.LDlinear);
#VBB.LoadESPLTable("./ESPL.tbl"); 

from numba import jit, njit, prange
#@numba.jit(nopython=True, parallel=True)

direc="./Example5/"

################################################################################

G= 6.67430*pow(10.0,-11.0)
AU=1.495978707*pow(10.0,11)
Msun=1.989*pow(10.0,30.0)
Rsun =6.9634*pow(10.0,8.0)
KPC= 3.0857*pow(10.0,19.0)## meter 
velocity=299792458.0##m/s
const=float(np.sqrt(4.0*G)/velocity)
thre=float(0.001); 
epsi=float(0.0005)

################################################################################

Nb=int(1000); 
nbh=int(50)
Nm=int(30000)
nx=int(1801)
ny=int(1801)
tt=int(9)
sini=  np.zeros((nx,ny))
pro=   np.zeros((nx,ny))
Pro=   np.zeros((nx,ny))
phi=   np.zeros((Nm));  
ksi=   np.zeros((Nm));  
x1=    np.zeros((Nm));  
x0=    np.zeros((Nm));
y1=    np.zeros((Nm));  
y0=    np.zeros((Nm));
z1=    np.zeros((Nm));  
RE=    np.zeros((Nm));
Astar1= np.zeros((Nm));
Astar2= np.zeros((Nm));
Fintl1= np.zeros((Nm));
Fintl2=np.zeros((Nm));
Flux1=  np.zeros((Nm));
Flux2=  np.zeros((Nm));
Occul= np.zeros((Nm));
rho=   np.zeros((Nm));
u=     np.zeros((Nm));
dis=   np.zeros((Nm));
disp=  np.zeros((Nm));
tim=   np.zeros((Nm));
v=     np.zeros((tt))

################################################################################

def tickfun(x, start, dd0, rho):
    return((start+x*dd0)/rho)
    
#=====================================

def Thirdlaw(MSum, period):
    return(np.power(G*MSum*period*period*24.0*24.0*3600.0*3600.0/(4.0*np.pi*np.pi),1.0/3.0))
#=====================================

@jit
def Kepler(phi, ecen):
    phi=phi*180.0/np.pi
    for kk in range(len(phi)): 
        while(phi[kk]>360.0):
            phi[kk]=phi[kk]-360.0  
        while(phi[kk]<0.0):
            phi[kk]=phi[kk]+360.0       
        if(phi[kk]>180):  phi[kk]=float(phi[kk]-360)
        if(phi[kk]<-181.0 or phi[kk]>181.0):  
            print("Phi:  ",  phi[kk], ecen[kk])
            input("Enter a number ")
    phi=phi*np.pi/180.0##radian 
    ksi=phi; 
    for iw in range(Nb):
        term=2.0/(iw+1.0)*ss.jv(int(iw+1),(iw+1.0)*ecen)*np.sin((iw+1)*phi)
        if(iw==0):   term0=np.abs(term)
        ksi+=term
        if(np.mean(np.abs(term))<np.mean(abs(thre)*term0) and iw>5):  
            break 
    return(ksi) 
#=====================================

#@jit
def Fluxs(rho, xs0, ys0, sini):
    for i in range(nx):  
        for j in range(ny):
            sini[j,i]=0.0;   
            xi=float(i-nx/2.0)*dx#[RE]
            yi=float(j-ny/2.0)*dy#[RE]
            dis=np.sqrt((xi-xs0)**2.0+(yi-ys0)**2.0)
            if(dis<rho or dis==rho):               
                mu=np.sqrt(1.0-dis**2.0/rho**2.0) 
                sini[j,i]=1.0*abs(1.0-limb*abs(1.0-mu))
            else: 
                sini[j,i]=0.0;               
    return(sini); 
#=====================================
#@jit
def LensEq2(xm, ym, Dls, Re):#[m, m, m, m]
    flag=-1; 
    b=np.sqrt(xm**2.0+ym**2.0)+1.0e-50#[m] Impact parameter
    angle=float(4.0*G*MBH/(velocity*velocity*b))#radian  
   
    tant= float(b/Dl)## tan(theta)
    tana=float(np.tan(angle))## tan(alpha)
    tanb=tant-(tant + float(tana-tant)/(1.0+ tant*tana))*Dls/(Dl+Dls)
    
    beta= float(tanb*Dl)#[m] in lens plane
    xs=   float(beta*xm/b)/Re#[RE]
    ys=   float(beta*ym/b)/Re#[RE]
    d2=   b*b/(Re*Re)
    xs0=  float(xm/Re-xm/Re/d2)
    ys0=  float(ym/Re-ym/Re/d2)
    if(float(angle*180.0/np.pi)>1.0 or abs(xs-xs0)>0.1 or abs(ys-ys0)>0.1): 
        print("deflection angle: ", angle*180.0/np.pi)
        print("new ccordinate : ",  xs,    ys)
        print("old coordinate:  ",  xs0,  ys0)
        #input("Enter a number ")
    if(b<RBH or b==RBH): flag=0;##Occultation by Lens
    else:                flag=1;
    return(xs, ys, xs0, ys0, flag)

#=====================================

@jit
def Eclipsing(xs, ys, zs):
    frac=0.0;  frac0=1.0;#no eclipse      
    proj=float(Dl/(Dl-xs));
    Disp=np.sqrt(ys**2.0 + zs**2.0)#[m]
    us=float(Disp-RBH-Rstar*proj);#[m]
    if(us<=0.0):
        if(Disp<=abs(Rstar*proj-RBH)):# complete eclipse 
            frac=1.0;  frac0=1.0;
        else:#partial eclipse   
            frac=0.0;  frac0=0.0;    
            for i1 in range(nbh):
                for j1 in range(nbh):  
                    yb=float(-RBH + i1*stepb);#[m]  
                    zb=float(-RBH + j1*stepb);#[m] 
                    zlim=np.sqrt(RBH**2.0 - yb**2.0); 
                    if(abs(zb)<=zlim):#circle
                        frac0+=1.0; 
                        yc=float(ys-yb);#[m]    
                        zc=float(zs-zb);#[m]  
                        rstar=np.sqrt(yc**2.0+zc**2.0)/(Rstar*proj); 
                        if(rstar<=1.0): frac+=1.0; 
    return(float(1.0-frac/frac0));
    
#=====================================
@jit   
#@jit(nopython=True, parallel=True)
def InRayShoot(Rho, Xsc, Ysc, Dls, Re, pro, Pro, sini):
    num0=0.0;  num1=0.0;  fintl1=0.0 
    Num0=0.0;  Num1=0.0;  fintl2=0.0 
    for i in range(nx):
        for j in range(ny):
            pro[j,i]=0.0; Pro[j,i]=0.0
    for i in range(nx):
        for j in range(ny):
            xi=float(i-nx/2.0)*dx##[RE]
            yi=float(j-ny/2.0)*dy##[RE]
            xsi,ysi, xsi0, ysi0,Flag=LensEq2(xi*Re, yi*Re, Dls, Re)#[m,m,m,m]--> [RE, RE, RE, RE, flag]
            #########################################################
            diss=np.sqrt((xsi-Xsc)**2.0+(ysi-Ysc)**2.0)
            if(diss<Rho or diss==Rho):#True images
                mu=np.sqrt(1.0-diss**2.0/Rho**2.0) 
                ff=1.0*abs(1.0-limb*abs(1.0-mu))  
                px=int(round((xsi+xsiz*0.5)/dx,2))           
                py=int(round((ysi+ysiz*0.5)/dy,2))
                num0+=ff
                pro[j,i]+=sini[py,px]
                if(Flag==0): 
                    #num1+=ff 
                    fintl1+=sini[py,px]
                if((px<0 or px>=nx or py<0 or py>=ny or abs(ff-sini[py,px])>0.05) and abs(diss-Rho)>0.1 ):
                    print("Error1 , px, py, ff, sini: ",  px, py, ff, sini[py,px] , diss/Rho , mu, limb)
                    input("Enter a number ")
            #########################################################        
            diss=np.sqrt((xsi0-Xsc)**2.0+(ysi0-Ysc)**2.0)
            if(diss<Rho or diss==Rho):#True images
                mu=np.sqrt(1.0-diss**2.0/Rho**2.0) 
                ff=1.0*abs(1.0-limb*abs(1.0-mu))  
                px=int(round((xsi0+xsiz*0.5)/dx,2))           
                py=int(round((ysi0+ysiz*0.5)/dy,2))
                Num0+=ff
                Pro[j,i]+=sini[py,px]
                if(Flag==0): 
                    #Num1+=ff 
                    fintl2+=sini[py,px]  
                if((px<0 or px>=nx or py<0 or py>=ny or abs(ff-sini[py,px])>0.05) and abs(diss-Rho)>0.1 ):
                    print("Error 2, px, py, ff, sini: ",  px, py, ff, sini[py,px],  diss/Rho, mu , limb)
                    input("Enter a number ")
            #########################################################        
    if(abs(num0-np.sum(pro))>0.05 or abs(Num0-np.sum(Pro))>0.05 or abs(num1-fintl1)>0.05 or abs(Num1-fintl2)>0.05): 
        print("Error, num, pro, ",  num0, np.sum(pro),  num1,   fintl1,  Num0, np.sum(Pro) ,  Num1, fintl2 ) 
        #input("Enter a number ") 
    return(pro,Pro, fintl1, fintl2, num0, Num0)                   
#===============================================================================

def MapI(Xsc, Ysc, Rho, Re, Pro, Num, H):
    plt.cla()   
    plt.clf()
    fig=plt.figure(figsize=(8,8))
    ax=fig.add_subplot(111)
    circle1=plt.Circle((Xsc/dx+nx/2, Ysc/dy+ny/2), float(Rho/dx),fill = False,color='w',lw=2.4, linestyle='-')
    circle2=plt.Circle((nx/2,ny/2),float(RBH/Re/dx),fill = False,color='k', lw=1.9, linestyle='-')
    circle3=plt.Circle((nx/2,ny/2),float(Re/Re/dx),fill = False, color='r', lw=1.9, linestyle='--')
    plt.imshow(Pro,cmap=cmap,interpolation='nearest',aspect='equal', origin='lower')
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(circle3)
    plt.clim()
    minn=np.min(Pro)
    maxx=np.max(Pro)
    step=float((maxx-minn)/(tt-1.0));
    for m in range(tt):
        v[m]=round(float(minn+m*step),1)
    cbar=plt.colorbar(orientation='horizontal',shrink=0.85,pad=0.1,ticks=v)
    cbar.ax.tick_params(labelsize=17)
    plt.clim(v[0]-0.005*step,v[tt-1]+0.005*step)
    plt.xticks(fontsize=19, rotation=0)
    plt.yticks(fontsize=19, rotation=0)
    plt.xlim(0.0+nx*0.29,nx*0.71)
    plt.ylim(0.0+ny*0.29,ny*0.71)
    ticc=np.array([ int(nx*0.3), int(nx*0.4), int(nx*0.5), int(nx*0.6), int(nx*0.7) ])
    ax.set_xticks(ticc,labels=[round(j,1) for j in tickfun(ticc,float(-xsiz*0.5),dx,Rho ) ])
    ax.set_yticks(ticc,labels=[round(j,1) for j in tickfun(ticc,float(-ysiz*0.5),dy,Rho ) ])
    ax.set_aspect('equal', adjustable='box')
    plt.xlabel(r"$x-\rm{axis}[R_{\star,~\rm{p}}]$",fontsize=20,labelpad=0.05)
    plt.ylabel(r"$y-\rm{axis}[R_{\star,~\rm{p}}]$",fontsize=20,labelpad=0.05)
    fig=plt.gcf()
    fig.tight_layout()
    fig.savefig(direc+"map{0:d}_{1:d}.jpg".format(H,Num), dpi=200)     
    print ("One map is plotted ")

################################################################################
fil0=open(direc+"param.dat","w")
fil0.close();

for h in range(1):  
    fil1=open(direc+"light{0:d}.dat".format(h),"w")
    fil1.close();
    Mstar=0.2*Msun#[kg]
    Rstar=0.3*Rsun#[m]
    MBH= 1.3*Msun#[kg]#WD
    RBH=0.01*Rsun*np.power(Msun/MBH,1.0/3.0)#[m]
    period=60.0## days
    inc=0.0# degree
    teta=0.0# degree
    ecen=0.0
    Dl=0.01*KPC#[m]
    limb=0.5
    tp=-0.4; ##in [period]
    alfa=0.01
    stepb=float(2.0*RBH/nbh/1.0);     
    for i in range(Nm):  
        tim[i]=float(-0.5+i/Nm/1.0) #[-0.5,0.5]
        phi[i]=(tim[i]-tp)*2.0*np.pi 
    
    inc= float(inc*np.pi/180.0)
    teta=float(teta*np.pi/180.0)
    a=Thirdlaw(float(MBH+Mstar), period)#[m]
    print("Semi_major axis[AU]", a/AU)
    if(ecen<0.01): ksi=phi
    else:          ksi=Kepler(phi, ecen)
    x0=a*(np.cos(ksi)-ecen)#[m]
    y0=a*np.sin(ksi)*np.sqrt(1.0-ecen**2.0)#[m]
    y1=                y0*np.cos(teta)+x0*np.sin(teta)#[m]
    x1=  np.cos(inc)*(-y0*np.sin(teta)+x0*np.cos(teta))#[m]
    z1= -np.sin(inc)*(-y0*np.sin(teta)+x0*np.cos(teta))#[m] 
    dis= np.sqrt(x1**2.0 + y1**2.0 + z1**2.0)+1.0e-50#[m]
    disp=np.sqrt(y1**2.0 + z1**2.0)+1.0e-50;#[m]
    RE=const*np.sqrt(MBH)*np.sqrt(np.abs(x1)*Dl/np.abs(Dl-x1)) + 1.0e-50#[m]
    rho=np.abs(Rstar*Dl/(Dl-x1)/RE)+1.0e-50#[RE]
    u= disp/RE##[RE]
    
    fil0=open(direc+"param.dat","a+")
    par=np.array([h,Mstar/Msun,Rstar/Rsun,MBH/Msun,RBH/Rsun, period, inc, teta, ecen, Dl/KPC, limb, tp, alfa, a/AU])
    np.savetxt(fil0,par.reshape((-1,14)),fmt ="%d  %.4f  %.4f  %.4f  %.4f  %.4f  %.4f  %.4f  %.4f  %.4f  %.4f  %.4f %.4f  %.4f")
    fil0.close();
    
    
    rhom=0.0
    for i in range(Nm): 
        if(x1[i]<0.0 and u[i]<float(5.0*rho[i]) ):
            if(rho[i]>rhom):  
                rhom=float(rho[i])           
    xsiz=float(5.0*rhom)#[RE]
    ysiz=float(5.0*rhom)#[RE]
    dx=float(xsiz/(nx-1.0))#[RE]
    dy=float(ysiz/(ny-1.0))#[RE]
    print("RE[a], rhio, u:  ",  RE/a, rho, u, rhom, xsiz, ysiz, dx, dy)
    
    st=0; sp=0; o1=0; o2=0
    nsi=0; 
################################################################################
    for k in range(Nm): 
        Astar1[k]=1.0;  
        Astar2[k]=1.0;  
        Fintl1[k]=0.0;  
        Fintl2[k]=0.0;  
        Occul[k]=1.0;
        Self=0; Ecl=0;
        num0=1.000;  Num0=0.0;   
        Fbase=float(np.pi*rho[k]*rho[k]*(1.0-limb/3.0)/(dx*dy))
        #################################################
        if(x1[k]<0.0 and u[k]<float(1.25*rho[k])):#Self-lensing
            Self=1
            if(st==0):st=k
            if(st>0): sp=k 
            if(u[k]<float(1.25*rho[k])): 
                xsc=float(y1[k]/RE[k])#[RE] Projected & Normalized Source_center
                ysc=float(z1[k]/RE[k])#[RE]
                sini=Fluxs(rho[k], xsc, ysc, sini)   
                pro,Pro,Fintl1[k],Fintl2[k],num0,Num0=InRayShoot(float(rho[k]), xsc, ysc, abs(x1[k]), float(RE[k]), pro, Pro, sini)
                Astar1[k]=float(np.sum(pro)/Fbase)
                Astar2[k]=float(np.sum(Pro)/Fbase)
                Fintl1[k]=float(   Fintl1[k]/Fbase)
                Fintl2[k]=float(   Fintl2[k]/Fbase)
                nsi+=1
                MapI(xsc, ysc, rho[k], float(RE[k]), pro,nsi, h);#[RE, RE, m, m, , , ]
                MapI(xsc, ysc, rho[k], float(RE[k]), Pro,nsi, h+1);#[RE, RE, m, m, , , ]
                MapI(xsc, ysc, rho[k], float(RE[k]),np.log10(np.abs(pro-sini)+0.1),nsi, h+2);#[RE, RE, m, m, , , ]
                MapI(xsc, ysc, rho[k], float(RE[k]), np.log10(np.abs(Pro-pro)+0.1),nsi, h+3);#[RE, RE, m, m, , , ]
                print("Self:U,rho,x1[a],As,Fl",round(u[k],1),round(rho[k],1),round(x1[k]/a,1),round(Astar1[k],10),round(Fintl1[k],10))
                print("************************************************")
        if(x1[k]>0.0 and disp[k]<float(Rstar*float(Dl/(Dl-x1[k]))+RBH)): 
            Ecl=1;
            Occul[k]=Eclipsing(x1[k],y1[k],z1[k])#[m, m, m]   
            if(o1==0):    o1=k
            if(o1>0):     o2=k    
        #################################################    
        Flux1[k]=float(Astar1[k]-Fintl1[k]+alfa*Occul[k])/(1.0+alfa)    
        Flux2[k]=float(Astar2[k]-Fintl2[k]+alfa*Occul[k])/(1.0+alfa)      
        fil1=open(direc+"light{0:d}.dat".format(h),"a+")
        ssa=np.array([k,Self,Ecl,tim[k],Astar1[k],Fintl1[k],Fintl2[k],Occul[k],Flux1[k],Astar2[k],u[k],rho[k],float(RE[k]/RBH),Flux2[k]])
        np.savetxt(fil1,ssa.reshape((-1,14)),fmt ="%d %d %d %.5f %.8f %.8f  %.8f  %.8f  %.8f  %.8f  %.5f  %.4f  %.4f  %.8f")
        fil1.close()
    print("limits of self-lensing:", st, sp, tim[st],   tim[sp] ) 
    print("limits of eclipsing:",    o1, o2, tim[o1],   tim[o2] )            
    
    
    
    
    

