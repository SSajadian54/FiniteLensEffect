from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import gridspec
import numpy as np
import matplotlib as mpl
import pylab
rcParams["font.size"] = 18
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Computer Modern Sans"]
rcParams["text.usetex"] = True
rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"
from matplotlib.ticker import FormatStrFormatter
from matplotlib.gridspec import GridSpec
from scipy.signal import savgol_filter
#from scipy.interpolate import make_interp_spline, BSpline
#from tsmoothie.smoother import *
#smoother = ConvolutionSmoother(window_len=5, window_type='ones')
from scipy.ndimage import uniform_filter1d

################################################################################

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return (y_smooth)



def ErrorTESS(maga):
    emt=-1.0;     
    if(maga<7.5):       emt=float(0.22*maga-5.850); 
    elif(maga<12.5):    emt=float(0.27*maga-6.225);  
    else:               emt=float(0.31*maga-6.725);    
    #emt=emt+np.random.randn(0.0,0.1)
    if(emt<-5.0): emt=-5.0 
    emt=np.power(10.0,emt)
    if(emt<0.00001 or emt>0.5 or maga<0.0):
        print("Error emt:  ", emt, maga); 
    return(emt); 
################################################################################
sector=int(0)
nc=int(49+1) ## No.  parameters

direc="./"


f1=open(direc+"F{0:d}_-2.dat".format(sector) )
N0=int(len(f1.readlines()))
Tobs=27.4;  
Rsun =6.9634*pow(10.0,8.0)

################################################################################

for ll in range(1): 
    f1=open(direc+"F{0:d}_-2.dat".format(sector),"r")
    nf= sum(1 for line in f1)  
    par=np.zeros((nf,nc))
    par=np.loadtxt(direc+"F{0:d}_-2.dat".format(sector)) 
    for i in range(nf):
        #par[i,36]=np.log10(par[i,36])## log10(F)
        #par[i,44]=np.log10(par[i,44])## log10(rho*)
        #par[i,46]=np.log10(par[i,46])## log10(u0/rho*) Log(impactL)
        #par[i,47]=np.log10(par[i,47])## log10(dp/Rstar) Log(impactO)
        FlagD, nsim, signD, lat, lon  =int(par[i,0]),int(par[i,1]),par[i,2],par[i,3],par[i,4]
        Ds, mass, Tstar, Rstar, limb  =par[i,5],par[i,6],par[i,7],par[i,8],par[i,9]
        Mab, Map,magb,fb,Nbl,Ext,prior= par[i,10],par[i,11],par[i,12],par[i,13],par[i,14],par[i,15],par[i,16]
        Ml, tefl, Rl, loggl, Mapl, Mabl=par[i,17],par[i,18], par[i,19],par[i,20],par[i,21],par[i,22]
        magG,magBP,magRP,MG,MBP,MRP,numl =par[i,23],par[i,24], par[i,25],par[i,26],par[i,27],par[i,28], par[i,29]
        inc,teta,ecen,period,lsemi,tp,ratio=par[i,30],par[i,31],par[i,32],par[i,33],par[i,34],par[i,35],par[i,36]
        error, dchi, snr,cdpp       =par[i,37], par[i,38], par[i,39], par[i,40]
        depth,depthO,depthL,rho=par[i,41],par[i,42],par[i,43],par[i,44]
        loggs, impactL, impactO, RcRe, SourceA= par[i,45],par[i,46],par[i,47],par[i,48], par[i,49]
        if(par[i,2]<0): par[i,2]=0 ## Occultation
        par[i,2]=int(par[i,2])
        sigA= abs(1.0-pow(10.0,-0.4*error))
        if(signD==0):  dep=r"$,~\log_{10}[\Delta F_{\rm{L}}]=$"
        if(signD==1):  dep=r"$,~\log_{10}[\Delta F_{\rm{E}}]=$"
        if(signD==-1): dep=r"$,~\log_{10}[\Delta F_{\rm{O}}]=$"
        RE=float(Rl*Rsun/RcRe)/Rsun
        print ("RE: ", RE, nsim, FlagD, period, rho)
        ######################################################################## 
        Tobs= float(13.7*2.0)
        ntran=float(Tobs/period)
        snr=  float(np.sqrt(ntran*1.0)*depth*1000000.0/cdpp)+0.01
        stat=r"$\rm{Not}-\rm{Detected}$"
        col='r'
        if(snr>=3.0 and ntran>=1.0): 
            stat=r"$\rm{Detected}$"
            col='g'
        ######################################################################## 
        if(FlagD>0): 
            nd=-1;  nm=-1;  
            try: 
                f1=open(direc+'L_{0:d}_0.dat'.format(nsim) )
                nd=int(len(f1.readlines()))
                print(nd)
            except: 
                print("file does not exist",  nsim)    
            try:
                f2=open(direc+'M_{0:d}_0.dat'.format(nsim) )
                nm=int(len(f2.readlines()))  
                print(nm)
            except: 
                print("file does not exist",  nsim)        
            print("nsim,   nd,  nm:  ", nsim, nd, nm)   
            if(nd>1 and nm>1 and nsim==78): 
                dat=np.zeros((nd,5))
                dat=np.loadtxt(direc+'L_{0:d}_0.dat'.format(nsim)) 
                mod=np.zeros((nm,16))
                mod=np.loadtxt(direc+'M_{0:d}_0.dat'.format(nsim)) 

                a1=1.000/(1.0+ratio)
                a2=ratio/(1.0+ratio) 
                l1=-1;l2=0 
                vlin=[]   
                umin=100000.0;tstar=-10.0; t0=0.0
                for j in range(nm):  
                    #if(abs(mod[j,1]-mod[j-1,1])>float(5.0*abs(mod[j-1,1]-mod[j-2,1]))  and mod[j,13]<0.5): 
                    #    mod[j,1]=0.5*(mod[j-2,1]+mod[j-3,1]) 
                    #    mod[j,2]=0.5*(mod[j-2,2]+mod[j-3,2])
                    #if(mod[j,15]==0 or mod[j,13]>1.2): 
                    #    mod[j,3]=1.0; mod[j,5]=0.0; mod[j,2]=1.0
                    if(mod[j,15]>0 and l1<0): l1=j
                    if(mod[j,15]>0 and l1>0): l2=j
                    if(mod[j,15]>0 and float(mod[j,13]*mod[j,12])<umin):
                        umin=float(mod[j,13]*mod[j,12])
                        indm=int(j)
                    if(abs(mod[j,13]-1.0)<0.005  or abs(mod[j,13]-2.0)<0.005):  vlin.append(mod[j,0])
                t0=float(mod[indm,0])## when u is minimum       
                for j in range(l2-l1):    
                    if(abs(mod[j+l1,13]-1.0)<0.1  and tstar<0.0): 
                        tstar=abs(mod[j+l1,0]-t0)
                print("l1, l2 :  ", l1, l2 , len(vlin),   vlin,  t0  ,mod[l1,0]-t0, mod[l2,0]-t0, tstar)
                print("inclination, impacts:  ", inc  ,  impactL  ,   impactO,  ratio,  a1,   a2 )
                ttess=np.arange(mod[l1,0]-t0, mod[l2,0]-t0, float(2.0/(60.0*24.0)), dtype=float)
                Atess=np.zeros(len(ttess)) 
                Etess=np.zeros(len(ttess))
                tim=(mod[l1:l2,0]-t0)/tstar
                if(l1<0 or tstar<0):  
                    print ("Error, l1, tstar:  ", l1, tstar, l2,   np.min(mod[:,13])   )
                    input("Enter a number ")
                ################################################################
                #d.t,As,Astar,Astar2,Occul,finl,l.num1, l.num0,x1, y1, z1, RE,ros,u/ros,disp/a, self
                #0   1   2      3     4     5     6       7    8   9   10  11  12   13,   14,   15
                #ymax1,ymin1= np.max(mod[:,1:4]), np.min(mod[:,1:4])
                #ymax2,ymin2= np.max(dat[:,1]), np.min(dat[:,1])
                #ymax ,ymin= np.max(np.array([ymax1,ymax2])) , np.min(np.array([ymin1, ymin2]))  
                #mod[:,2] = savgol_filter(mod[:,2], 30, 5)  
                #mod[:,15] = savgol_filter(mod[:,15], 50, 2) 

                #mod[l1:l2,3]=uniform_filter1d(mod[l1:l2,3],size=5)
                #astot=mod[l1:l2,3]*a1-mod[l1:l2,5]*a1+mod[l1:l2,4]*a2
                #print("smoother star2, astot:  ", selfIRS,  astot,   a1, a2 )
                ################################################################
                plt.cla()
                plt.clf()
                fig=plt.figure(figsize=(8,6))
                ax1=fig.add_subplot(111)
                #plt.plot(tim,mod[l1:l2,1],'k--',label=r"$\rm{Overall}~\rm{Flux}$",lw=1.7)
                #plt.plot(tim,mod[l1:l2,2]*a1+a2,'g--',label=r"$\rm{Self}-\rm{Lensing}$",lw=1.5)
                #plt.plot(tim,(mod[l1:l2,3]-mod[l1:l2,5])*a1+mod[l1:l2,4]*a2,'k-',label=r"$\rm{Overall}~\rm{Flux}$",lw=1.7)
                #plt.plot(tim,mod[l1:l2,3]*a1+a2,'g--', label=r"$\rm{Self}-\rm{Lensing}$",lw=1.5)
                plt.plot(tim,(1.0-mod[l1:l2,5])*a1+a2,'r-.',label=r"$\rm{Finite}-\rm{Lens}$",lw=1.5)
                
                #!!!!!!!!!!!!!!!!!!!!!!!!!!!
                mod[l1:l2,3]=uniform_filter1d(mod[l1:l2,3],size=15)
                #mod[l1:l2,3] = savgol_filter(mod[l1:l2,3],17,3)## Self-lensing with IRShooting
                astot=mod[l1:l2,3]*a1-mod[l1:l2,5]*a1+mod[l1:l2,4]*a2
                for j in range(len(ttess)):
                    uu= int(np.argmin(np.abs(tim*tstar-ttess[j])))
                    emt=ErrorTESS( float(magb-2.5*np.log10(astot[uu])) )
                    deltaA=abs(np.power(10.0,-0.4*emt)-1.0)*astot[uu]; 
                    Atess[j]= float(astot[uu]+np.random.randn(1)*deltaA+0.0)
                    Etess[j]= float(deltaA)
                plt.plot(tim,astot,'k-',label=r"$\rm{Overall}-\rm{Flux}$",lw=1.7)
                plt.plot(tim,mod[l1:l2,3]*a1+a2,'g--', label=r"$\rm{Self}-\rm{Lensing}$",lw=1.5)
                #plt.errorbar(ttess/tstar,Atess,yerr=Etess,fmt=".",markersize=6.,color='m',ecolor='lime',elinewidth=0.2, capsize=0,alpha=0.42,label=r"$\rm{TESS}~\rm{Data}$")
                
                
                #for j in range(len(vlin)): 
                #    plt.axvline( float(vlin[j]-t0)/tstar, color="m", ls="dashed", lw=0.95)

                plt.title(
                r"$M_{\rm{WD}}(M_{\odot})=$"+'{0:.1f}'.format(Ml)+r"$,~R_{\rm{WD}}/R_{\rm{E}}=$"+'{0:.2f}'.format(RcRe)+    
                r"$,~M_{\star}(M_{\odot})=$"+'{0:.1f}'.format(mass)+r"$,~\rho_{\star}=$"+'{0:.1f}'.format(rho)+
                r"$,~T(\rm{days})=$"+'{0:.1f}'.format(float(period))+
                r"$,~\epsilon=$"+'{0:.1f}'.format(float(ecen)),fontsize=13.5,color='k')
                plt.xlim(np.min(tim),  np.max(tim) )
                plt.xlim(-1.012,1.012)
                plt.ylim(1.0-0.0005, np.max(mod[l1:l2,1])*1.0005 )
                plt.xticks(fontsize=17, rotation=0)
                plt.yticks(fontsize=17, rotation=0)
                plt.xlabel(r"$\rm{time}-t_{0}(t_{\star})$", fontsize=18)
                plt.ylabel(r"$\rm{Normalized}~\rm{Flux}$",fontsize=19)
                plt.legend(prop={"size":15.},loc='best')
                fig=plt.gcf()
                fig.tight_layout()
                fig.savefig(direc+"Lensing{0:d}.jpg".format(nsim), dpi=200)
                ################################################################              
                
                                       
                plt.clf()
                plt.cla()
                fig=plt.figure(figsize=(8,6))
                ax1=fig.add_subplot(111)
                plt.errorbar(dat[:,0],dat[:,1],yerr=dat[:,2],fmt=".",markersize=2.,color='m',ecolor='gray',elinewidth=0.02, capsize=0,alpha=0.42,label=r"$\rm{TESS}~\rm{Data}$")
                
                plt.plot(mod[:,0],(mod[:,3]-mod[:,5])*a1+mod[:,4]*a2,'k-',label=r"$\rm{Overall}~\rm{Flux}$",lw=1.7)
                plt.plot(mod[:,0], mod[:,3]*a1+a2,'g--',label=r"$\rm{Self}-\rm{lensing}$",lw=1.5)
                plt.plot(mod[:,0], a1+mod[:,4]*a2,'b:',label=r"$\rm{Eclipsing}$",lw=1.5)
                plt.plot(mod[:,0],(1.0-mod[:,5])*a1+a2,'r-.',label=r"$\rm{Finite}-\rm{lens}$",lw=1.5)
                
                plt.ylabel(r"$\rm{Normalized}~\rm{Flux}$",fontsize=19)
                plt.xlabel(r"$\rm{time}(\rm{days})$",fontsize=19)
                
                plt.title(
                r"$M_{\rm{WD}}(M_{\odot})=$"+'{0:.1f}'.format(Ml)+
                r"$,~M_{\star}(M_{\odot})=$"+'{0:.1f}'.format(mass)+
                r"$,~T(\rm{days})=$"+'{0:.2f}'.format(period)+
                r"$,~\log_{10}[\mathcal{F}]=$"+'{0:.2f}'.format(np.log10(ratio))+
                r"$,~\rho_{\star}=$"+'{0:.1f}'.format(rho)+ "\n"+ 
                r"$b(R_{\star})=$"+'{0:.2f}'.format(impactL)+   
                r"$,~R_{\rm{WD}}/R_{\rm{E}}=$"+'{0:.2f}'.format(RcRe)+    
                str(dep)+'{0:.2f}'.format(np.log10(depth))+     
                r"$,~\rm{SNR}=$"+'{0:.2f}'.format(snr),fontsize=17.0,color='k')
                
                #pylab.ylim([ymin, ymax])
                #plt.xlim([0.0,Tobs-1.0])
                plt.xticks(fontsize=18, rotation=0)
                plt.yticks(fontsize=18, rotation=0)
                #plt.gca().invert_yaxis()
                ax1.grid("True")
                ax1.grid(linestyle='dashed')
                ax1.legend(title=stat,prop={"size":15.})
                fig=plt.gcf()
                fig.tight_layout()  
                fig.savefig(direc+"LightW{0:d}.jpg".format(nsim),dpi=200)
                
                ################################################################
                
                
                plt.cla() 
                plt.clf()
                plt.figure(figsize=(8, 6))
                plt.plot(mod[:,0]/period,mod[:, 7]/np.max(mod[:,7]), color="r",label="x1(LoS)",lw=1.2, alpha=0.95)
                plt.plot(mod[:,0]/period,mod[:, 8]/np.max(mod[:,8]), color="b",label="Phase",  lw=1.2, alpha=0.95)
                plt.plot(mod[:,0]/period,mod[:, 9]/np.max(mod[:,9]), color="g",label="RE",     lw=1.2, alpha=0.95)
                plt.plot(mod[:,0]/period,mod[:,10]/np.max(mod[:,10]),color="m",label="Rostar", lw=1.2, alpha=0.95)
                plt.plot(mod[:,0]/period,mod[:,11]/np.max(mod[:,11]),color="k",label="u",      lw=1.2, alpha=0.95)
                plt.plot(mod[:,0]/period,mod[:,12]/np.max(mod[:,12]),color="c",label="disp",     lw=1.2, alpha=0.95)
                plt.xticks(fontsize=17, rotation=0)
                plt.yticks(fontsize=17, rotation=0)
                plt.xlabel(r"$time/Period$", fontsize=18)
                plt.legend(prop={"size":15.},loc='best')
                fig=plt.gcf()
                fig.tight_layout()
                fig.savefig(direc+"xyz{0:d}.jpg".format(nsim), dpi=200)
                print("*****************************************************")                 
                ################################################################
                input("Enter a number ")























