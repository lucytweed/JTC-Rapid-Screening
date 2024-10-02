import numpy as np
import pandas as pd
import os
import shutil
import time

from scipy.integrate import solve_bvp
from scipy.optimize import fsolve
from scipy.special import expi
from scipy.interpolate import UnivariateSpline

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Polygon
mpl.rc('xtick', labelsize=16) 
mpl.rc('ytick', labelsize=16) 

input_file = input("Enter input spreadsheet filename: ")
filename = input("Enter output directory name: ")

st = time.time()

# Reading input file
print('\nReading input file')
# Reading options
df0 = pd.read_excel(input_file,sheet_name=0,header=None,index_col=0)
refPT_option = df0[1]['Automatically set reference pressure and temperature?']
fluidprops_option = df0[1]['Automatically calculate fluid properties?']
paramsweep_option = df0[1]['Do parameter sweep?']
plots_option = df0[1]['Generate plots?']
contourplots_option = df0[1]['Generate thermodynamic contour plots?']

# Reading reservoir properties
df1 = pd.read_excel(input_file,sheet_name=1,header=0,usecols=[1,2,3,4,5,6,7,8,9,10,11],skiprows=[1])
df1['T0'].loc[0:2]+=273.15
df1['Tw'].loc[0:2]+=273.15
resprops = list(df1.iloc[0,:])
[P0,T0,Tw,Qw,H,rw,phi,k,kT,rho_s,Cp_s] = resprops
index_to_vary = [i for i in range(len(resprops)) if list(df1.loc[3])[i] == 'Y'][0]
respropsmax = [P0,T0,Tw,Qw,H,rw,phi,k,kT,rho_s,Cp_s]
respropsmax[index_to_vary] = list(df1.iloc[1,:])[index_to_vary]
respropsmin = [P0,T0,Tw,Qw,H,rw,phi,k,kT,rho_s,Cp_s]
respropsmin[index_to_vary] = list(df1.iloc[2,:])[index_to_vary]

# Reading fluid properties
df2 = pd.read_excel(input_file,sheet_name=2,header=0,usecols=[1,2,3,4,5,6,7,8],skiprows=[1])
[Pref,Tref,rho_r,beta_f,alpha_f,Cp_f,mu,JT] = list(df2.iloc[0,:])
Tref = Tref+273.15

# Reading time intervals
df3 = pd.read_excel(input_file,sheet_name=3,header=None,index_col=0).transpose()
t = np.array(list(df3.dropna()['Time value (s)']))
tlabels = np.array(list(df3.dropna()['Time label']))
N = len(t)

# Creating output directory
print('\nCreating run directory')
path = '../JTC_screening_runs/' + filename
os.makedirs(path, exist_ok=True)
print('\nCopying input file to run directory')
shutil.copyfile(input_file, path + '/input.xlsx')

# Reference conditions
if refPT_option not in ["Y", "N"]:
    print("Incorrect option selected for 'Automatically set reference pressure and temperature?'. Enter Y or N")
elif refPT_option=="Y":
    print('\nCalculating reference conditions')
    import CoolProp
    CP = CoolProp.CoolProp
    fluid = 'CarbonDioxide'
    rho0 = CP.PropsSI('Dmass','P',P0,'T',T0,fluid)
    beta_f0 = CP.PropsSI('isothermal_compressibility','P',P0,'T',T0,fluid)
    mu0 = CP.PropsSI('V','P',P0,'T',T0,fluid)
    L = 2.e3
    Dp = k/(phi*mu0*beta_f0)
    tL = L**2/(4*Dp)
    eta_wL = rw/np.sqrt(Dp*tL)
    eta_L = L/np.sqrt(Dp*tL)
    etalin = np.logspace(np.log10(eta_wL),np.log10(eta_L),1000)
    rlin = etalin*np.sqrt(Dp*tL)
    Plin = -np.exp((eta_wL**2)/4)*mu0*Qw/(4*np.pi*H*k*rho0)*expi(-(etalin**2)/4) + P0
    Pav = (np.trapz(Plin, rlin)/L)
    Pref = round(Pav*1.e-6,2)*1.e6
    Tref = Tw

# Reference thermodynamic properties
if fluidprops_option not in ["Y", "N"]:
    print("Incorrect option selected for 'Automatically calculate fluid properties?'. Enter Y or N")
elif fluidprops_option=="Y":
    print('\nCalculating reference fluid properties')
    import CoolProp
    CP = CoolProp.CoolProp
    fluid = 'CarbonDioxide'
    rho_r = CP.PropsSI('Dmass','P',Pref,'T',Tref,fluid)
    beta_f = CP.PropsSI('isothermal_compressibility','P',Pref,'T',Tref,fluid)
    alpha_f = CP.PropsSI('isobaric_expansion_coefficient','P',Pref,'T',Tref,fluid)
    Cp_f = CP.PropsSI('Cpmass','P',Pref,'T',Tref,fluid)
    JT = CP.PropsSI('d(T)/d(P)|H','P',Pref,'T',Tref,fluid)
    mu = CP.PropsSI('V','P',Pref,'T',Tref,fluid)

dim_params = {'P0':P0, 'T0':T0, 'Tw':Tw, 'Qw':Qw, 'H':H, 'rw':rw, 'phi':phi, 'k':k, 'kT':kT, 'rho_s':rho_s, 'Cp_s':Cp_s, 
       'Pref':Pref, 'Tref':Tref, 'rho_r':rho_r, 'beta_f':beta_f, 'alpha_f':alpha_f, 'Cp_f':Cp_f, 'mu':mu, 'JT':JT}

# Reference parameter values at Pref, Tref:
rhoCp_av = phi*rho_r*Cp_f+(1-phi)*rho_s*Cp_s
Dp = k/(phi*mu*beta_f)
G = (phi*rho_r*Cp_f)/rhoCp_av
J = alpha_f*JT/beta_f
A = phi*alpha_f*alpha_f*Tref/(beta_f*rhoCp_av)
PeT = k*rhoCp_av/(kT*phi*mu*beta_f)
Chi = mu*beta_f*Qw/(2*np.pi*H*k*rho_r)
Twd = alpha_f*(Tw-T0)

nondim_params = {'G':G, 'J':J, 'A':A, 'PeT':PeT, 'Chi':Chi, 'Twd':Twd}

def solver(resparams):
    [P0,T0,Tw,Qw,H,rw,phi,k,kT,rho_s,Cp_s] = resparams
    Dp = k/(phi*mu*beta_f)
    Twd = alpha_f*(Tw-T0)
    Prd = beta_f*(Pref-P0)
    Trd = alpha_f*(Tref-T0)
    
    # Wellbore eta
    eta_w = rw/np.sqrt(Dp*t)
    
    # ODEs
    def F(eta,y):
        # Parameter values
        rho_f = rho_r*np.exp(y[0]-Prd-y[1]+Trd)
        rhoCp_av = phi*rho_f*Cp_f+(1-phi)*rho_s*Cp_s
        G = phi*rho_f*Cp_f/rhoCp_av
        J = alpha_f*JT/beta_f
        A = phi*alpha_f*(y[1]+alpha_f*T0)/(beta_f*rhoCp_av)
        PeT = k*rhoCp_av/(phi*mu*beta_f*kT)
        # 1st order ODEs
        dPdeta = y[2]
        dTdeta = y[3]
        d2Pdeta2 = -(eta/2)*(dPdeta-(1-eta_wi/eta)*dTdeta) - dPdeta*(dPdeta-dTdeta) - dPdeta/eta
        d2Tdeta2 = -PeT*((eta/2)*((1-eta_wi/eta)*dTdeta-A*dPdeta) + G*dPdeta*(dTdeta-J*dPdeta)) - dTdeta/eta
        return np.array((dPdeta,dTdeta,d2Pdeta2,d2Tdeta2))
    # BCs
    def bcs(ya,yb):
        rho_fa = rho_r*np.exp(ya[0]-Prd-ya[1]+Trd)
        rhoCp_ava = phi*rho_fa*Cp_f+(1-phi)*rho_s*Cp_s
        rho_fb = rho_r*np.exp(yb[0]-Prd-yb[1]+Trd)
        rhoCp_avb = phi*rho_fb*Cp_f+(1-phi)*rho_s*Cp_s
        Ga = phi*rho_fa*Cp_f/rhoCp_ava
        Chi = mu*beta_f*Qw/(2*np.pi*H*k*rho_fa)
        r1 = ya[2]+Chi/eta_wi
        r2 = ya[1]-Twd
        r3 = yb[0]
        r4 = yb[1]
        return np.array([r1,r2,r3,r4])
    
    # Skeletons for full solutions
    sol = []
    eta = []
    P = []
    T = []
    dPdeta = []
    dTdeta = []
    rho = []
    
    print('\nCalculating coupled temperature and pressure fields')
    for i,eta_wi in enumerate(eta_w):
        # domain
        eta0 = np.logspace(np.log10(eta_wi),1.,1000)
        y0 = np.zeros((4,eta0.size))
        # BVP Solver
        solution = solve_bvp(F, bcs, eta0, y0,tol=1.e-7,max_nodes=1000000,verbose=2)
        sol.append(solution)
        eta.append(sol[i].x)
        P.append(sol[i].y[0]/beta_f+P0)
        T.append(sol[i].y[1]/alpha_f+T0)
        dPdeta.append(sol[i].y[2]/beta_f)
        dTdeta.append(sol[i].y[3]/alpha_f)
        rho.append(rho0*np.exp(beta_f*(P[i]-Pref)-alpha_f*(T[i]-Tref)))
        sol[i].message
    
    print('\nCalculating front positions')
    def co2_front(eta_f,i):
        Psol = sol[i].sol.__call__(eta_f)
        return eta_f+2*Psol[2]
    
    def thermal_front(eta_f,eta_w,i):
        Psol = sol[i].sol.__call__(eta_f)
        rho_f = rho_r*np.exp(Psol[0]-Prd-(Psol[1]-Trd))
        rhoCp_av = phi*rho_f*Cp_f+(1-phi)*rho_s*Cp_s
        G = phi*rho_f*Cp_f/rhoCp_av
        return eta_f-eta_w+2*G*Psol[2]
    
    def thermal_max(eta_f,i):
        Psol = sol[i].sol.__call__(eta_f)
        rho_f = rho_r*np.exp(Psol[0]-Prd-(Psol[1]-Trd))
        rhoCp_av = phi*rho_f*Cp_f+(1-phi)*rho_s*Cp_s
        G = phi*rho_f*Cp_f/rhoCp_av
        J = alpha_f*JT/beta_f
        A = phi*alpha_f*alpha_f*T0/(beta_f*rhoCp_av)
        return A*eta_f+2*G*J*Psol[2]
    
    eta_C = []
    for i in range(N):
        y = fsolve(co2_front,0.49,args=(i),xtol=1.e-6,full_output=True)
        eta_C.append(y[0][0])
        print('\neta_C \n sol={}, nfev={}, error={} \n {} \n'.format(y[0],y[1]['nfev'],y[1]['fvec'][0],y[3]))
    
    eta_T = []
    for i in range(N):
        y = fsolve(thermal_front,0.2,args=(eta_w[i],i),xtol=1.e-6,full_output=True)
        eta_T.append(y[0][0])
        print('eta_T \n sol={}, nfev={}, error={} \n {} \n'.format(y[0],y[1]['nfev'],y[1]['fvec'][0],y[3]))
    
    eta_M = []
    for i in range(N):
        y = fsolve(thermal_max,0.39,args=(i),xtol=1.e-6,full_output=True)
        eta_M.append(y[0][0])
        print('eta_M \n sol={}, nfev={}, error={} \n {} \n'.format(y[0],y[1]['nfev'],y[1]['fvec'][0],y[3]))
    
    P_T = sol[-1].sol.__call__(eta_T)[0]
    print('P_T \n sol={} MPa'.format((P_T/beta_f+P0)*1.e-6))
    
    
    print('\nCalculating CO2 concentration profile') 
    def CO2_concentration(eta, eta_C):
        return np.where(eta>eta_C,np.zeros(len(eta)),np.ones(len(eta)))
    c = [CO2_concentration(eta[i],eta_C[i]) for i in range(N)]
    
    print('\nCalculating r')
    r = np.array([eta[i]*np.sqrt(Dp*t[i]) for i in range(N)],dtype=object)
    r_C = np.array([eta_C[i]*np.sqrt(Dp*t[i]) for i in range(N)],dtype=object)
    r_M = np.array([eta_M[i]*np.sqrt(Dp*t[i]) for i in range(N)],dtype=object)
    r_T = np.array([eta_T[i]*np.sqrt(Dp*t[i]) for i in range(N)],dtype=object)

    dPdt = np.array([-eta[i]/(2*t[i])*dPdeta[i] for i in range(N)],dtype=object)
    dPdr = np.array([dPdeta[i]/np.sqrt(Dp*t[i]) for i in range(N)],dtype=object)
    dTdt = np.array([(eta_w[i]-eta[i])/(2*t[i])*dTdeta[i] for i in range(N)],dtype=object)
    dTdr = np.array([dTdeta[i]/np.sqrt(Dp*t[i]) for i in range(N)],dtype=object)
    drhodt = np.array([rho[i]*(beta_f*dPdt[i] - alpha_f*dTdt[i]) for i in range(N)],dtype=object)
    drhodr = np.array([rho[i]*(beta_f*dPdr[i] - alpha_f*dTdr[i]) for i in range(N)],dtype=object)
                    
    print('\nSaving solutions to dictionary')
    results = [eta, r, P, T, c, dPdeta, dTdeta, dPdt, dPdr, dTdt, dTdr, drhodt, drhodr, rho, eta_w, eta_C, eta_T, eta_M, r_C, r_T, r_M]
    resultsdict = [{'eta': eta[i], 'r':r[i], 'P':P[i], 'T':T[i], 'c':c[i], 'dPdeta':dPdeta[i], 'dTdeta':dTdeta[i], 'dPdt':dPdt[i], 'dPdr':dPdr[i], 'dTdt':dTdt[i], 'dTdr':dTdr[i], 'drhodt':drhodt[i], 'drhodr':drhodr[i], 'rho':rho[i], 'eta_w':eta_w[i], 'eta_C': eta_C[i], 'eta_T': eta_T[i], 'eta_M': eta_M[i],'r_C': r_C[i], 'r_T': r_T[i], 'r_M': r_M[i]} for i in range(N)]
    
    return results, resultsdict

# Calculate reference model
results, resultsdict = solver(resprops)

if paramsweep_option not in ["Y", "N"]:
    print("Incorrect option selected for 'Do parameter sweep?'. Enter Y or N")
elif paramsweep_option=="Y":
    print('\nCalculating upper and lower bounds')
    # Calculate model with max parameter value
    results_max, results_maxdict = solver(respropsmax)
    
    # Calculate model with min parameter value
    results_min, results_mindict = solver(respropsmin)

et = time.time()
pt = et-st
print('\nexecution time (wall time) = {}'.format(pt))

print('\nSaving solutions to output file')
df_dim_params = pd.DataFrame(dim_params,index=['values'])
df_nondim_params = pd.DataFrame(nondim_params,index=['values'])
df_results = [pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in resultsdict[i].items() ])) for i in range(N)]

with pd.ExcelWriter(path + '/output.xlsx') as writer:
    df_dim_params.to_excel(writer, sheet_name='input parameters', index=False)
    df_nondim_params.to_excel(writer, sheet_name='non-dimensional parameters', index=False)
    for i in range(N):
        df_results[i].to_excel(writer, sheet_name=tlabels[i], index=False)

resprops[1]-=273.15
resprops[2]-=273.15
respropsmax[1]-=273.15
respropsmax[2]-=273.15
respropsmin[1]-=273.15
respropsmin[2]-=273.15

# Plotting functions
[eta, r, P, T, c, dPdeta, dTdeta, dPdt, dPdr, dTdt, dTdr, drhodt, drhodr, rho, eta_w, eta_C, eta_T, eta_M, r_C, r_T, r_M] = results
def eta_plot():
    col_list = ['g','b','r','y','c','m']
    col = col_list[0:N]
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1,sharex=True,figsize=(8,9), height_ratios=[3,3,3,1])
    fig.subplots_adjust(hspace=0)
    
    ax1.axvline(eta_C[-1],color='k',alpha=0.1,linewidth=3)
    ax1.axvline(eta_M[-1],linestyle='-',color='sandybrown',alpha=0.3,linewidth=3)
    ax1.axvline(eta_T[-1],linestyle='-',color='cornflowerblue',alpha=0.3,linewidth=3)
    ax2.axvline(eta_C[-1],color='k',alpha=0.1,linewidth=3)
    ax2.axvline(eta_M[-1],linestyle='-',color='sandybrown',alpha=0.3,linewidth=3)
    ax2.axvline(eta_T[-1],linestyle='-',color='cornflowerblue',alpha=0.3,linewidth=3)
    ax3.axvline(eta_C[-1],color='k',alpha=0.1,linewidth=3)
    ax3.axvline(eta_M[-1],linestyle='-',color='sandybrown',alpha=0.3,linewidth=3)
    ax3.axvline(eta_T[-1],linestyle='-',color='cornflowerblue',alpha=0.3,linewidth=3)
    ax4.axvline(eta_M[-1],linestyle='-',color='sandybrown',alpha=0.3,linewidth=3)
    ax4.axvline(eta_T[-1],linestyle='-',color='cornflowerblue',alpha=0.3,linewidth=3)
    ax4.text(eta_T[-1],-1.2,'$\eta_T$',color='k',fontsize=14)
    ax4.text(eta_M[-1],-1.2,'$\eta_M$',color='k',fontsize=14)
    ax4.text(eta_C[-1],-1.2,'$\eta_C$',color='k',fontsize=14)

    for i in range(N):
        j = N-i-1
        ax1.semilogx(eta[j], P[j]*1.e-6,'-',label=tlabels[j],color=col[j])
        ax1.semilogx(eta[j][0],P[j][0]*1.e-6,'x',color=col[j],markersize=8)
        ax2.semilogx(eta[j], T[j]-273.15,'-',label=tlabels[j],color=col[j])
        ax2.semilogx(eta[j][0],T[j][0]-273.15,'x',color=col[j],markersize=8)
        ax3.semilogx(eta[j], rho[j], '-', color=col[j])
        ax3.semilogx(eta[j][0], rho[j][0], 'x', color=col[j],markersize=8)
        ax4.semilogx(eta[j][0],1,'x',color=col[j])
        ax4.text(eta[j][0],0.65,'$\eta_W$',color=col[j],fontsize=14)
        ax4.semilogx(eta[j],c[j],'-',color=col[j])
        
    ax2.add_patch(Polygon([(eta_T[-1],T0-273.15-1),(3,T0-273.15-1),(3,T0-273.15+1),(eta_T[-1],T0-273.15+1)],fill=False,linestyle=':',linewidth=1))
    
    handles, axlabels = ax1.get_legend_handles_labels()
    ax1.legend(handles[::-1], axlabels[::-1], fontsize=18,bbox_to_anchor=(1,1),facecolor='white',framealpha=0)
    ax1.grid('on',linestyle=':',alpha=0.6)
    # ax1.set_ylim([1.5,6.8])
    ax1.tick_params(axis='both', which='major', labelsize=14, direction='in', length=5)
    ax1.tick_params(axis='both', which='minor', direction='in', length=3)
    ax1.set_ylabel("p (MPa)",fontsize=18)
    
    ax2.grid('on',linestyle=':',alpha=0.6)
    # ax2.set_ylim([0,89])
    ax2.tick_params(axis='both', which='major', labelsize=14, direction='in', length=5)
    ax2.tick_params(axis='both', which='minor', direction='in', length=3)
    ax2.set_ylabel("T ($^{\circ}$C)",fontsize=18)
    
    ax3.grid('on',linestyle=':',alpha=0.6)
    # ax3.set_xlim([8.e-5,10])
    # ax3.set_ylim([1.5,6.8])
    ax3.tick_params(axis='both', which='major', labelsize=14, direction='in', length=5)
    ax3.tick_params(axis='both', which='minor', direction='in', length=3)
    ax3.set_ylabel(r"$\rho$ (kg m$^{-3}$)",fontsize=18)

    ax4.grid('on',linestyle=':',alpha=0.6)
    ax4.tick_params(axis='both', which='major', labelsize=14, direction='in', length=5)
    ax4.tick_params(axis='both', which='minor', direction='in', length=3)
    # ax4.set_xlim([8.e-5,10])
    ax4.set_ylim([-0.1,1.3])
    ax4.set_xlabel("$\eta$",fontsize=18)
    ax4.set_ylabel("$c$",fontsize=18)
    
    plt.savefig(path +'/'+ 'fig1.pdf',dpi=300, bbox_inches='tight') 

def Tmax_plot():
    col_list = ['g','b','r','y','c','m']
    col = col_list[0:N]
    plt.figure(figsize=(7,2.5))
    plt.axvline(eta_C[-1],color='k',alpha=0.1,linewidth=3)
    plt.axvline(eta_M[-1],linestyle='-',color='sandybrown',alpha=0.3,linewidth=3)
    plt.axhline(T0-273.15,linestyle='--',color='k',linewidth=1)
    plt.semilogx(eta[-1], T[-1]-273.15,'-',color='g')
    plt.grid('on',linestyle=':',alpha=0.6)
    plt.tick_params(axis='both', which='major', labelsize=14, direction='in', length=5)
    plt.tick_params(axis='both', which='minor', direction='in', length=3)
    plt.ylim([T0-273.15-1,T0-273.15+1])
    plt.xlim([eta_T[-1],3])
    plt.xlabel("$\eta$",fontsize=18)
    plt.ylabel("T ($^{\circ}$C)",fontsize=18)
    
    plt.savefig(path +'/'+ 'fig1b.pdf',dpi=300, bbox_inches='tight') 

def r_plot():
    col_list = ['g','b','r','y','c','m']
    col = col_list[0:N]
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1,sharex=True,figsize=(8,9), height_ratios=[3,3,3,1])
    fig.subplots_adjust(hspace=0)

    for i in range(N):
        j = N-i-1
        ax1.semilogx(r[j], P[j]*1.e-6,'-',label=tlabels[j],color=col[j])
        ax2.semilogx(r[j], T[j]-273.15,'-',label=tlabels[j],color=col[j])
        ax3.semilogx(r[j], rho[j], '-', color=col[j])
        ax4.semilogx(r[j],c[j],'-',color=col[j])

    ax1.grid('on',linestyle=':',alpha=0.6)
    #ax1.set_ylim([1.5,4.9])
    ax1.tick_params(axis='both', which='major', labelsize=14, direction='in', length=5)
    ax1.tick_params(axis='both', which='minor', direction='in', length=3)
    ax1.set_ylabel("p (MPa)",fontsize=18)
    
    handles, axlabels = ax2.get_legend_handles_labels()
    ax2.legend(handles[::-1], axlabels[::-1], fontsize=16, loc='lower right')
    ax2.grid('on',linestyle=':',alpha=0.6)
    ax2.set_ylim([min([min(T[i]) for i in range(N)])-273.15 - 2,max([max(T[i]) for i in range(N)])-273.15 + 2])
    ax2.tick_params(axis='both', which='major', labelsize=14, direction='in', length=5)
    ax2.tick_params(axis='both', which='minor', direction='in', length=3)
    ax2.set_ylabel("T ($^{\circ}$C)",fontsize=18)
    
    ax3.grid('on',linestyle=':',alpha=0.6)
    # ax3.set_xlim([8.e-5,10])
    # ax3.set_ylim([1.5,6.8])
    ax3.tick_params(axis='both', which='major', labelsize=14, direction='in', length=5)
    ax3.tick_params(axis='both', which='minor', direction='in', length=3)
    # ax3.set_xlabel("$\eta$",fontsize=18)
    # ax3.set_xticks([])
    ax3.set_ylabel(r"$\rho$ (kg m$^{-3}$)",fontsize=18)

    ax4.grid('on',linestyle=':',alpha=0.6)
    ax4.tick_params(axis='both', which='major', labelsize=14, direction='in', length=5)
    ax4.tick_params(axis='both', which='minor', direction='in', length=3)
    ax4.set_xlim([1.e-1,2.e3])
    ax4.set_ylim([-0.1,1.3])
    ax4.set_xlabel("$r \ (m)$",fontsize=18)
    ax4.set_ylabel("$c$",fontsize=18)
    
    plt.savefig(path +'/'+ 'fig2.pdf',dpi=300, bbox_inches='tight') 

def r_plot_paramvar(max, min):
    [etamax, rmax, Pmax, Tmax, cmax, dPdetamax, dTdetamax, dPdtmax, dPdrmax, dTdtmax, dTdrmax, drhodtmax, drhodrmax, rhomax, eta_wmax, eta_Cmax, eta_Tmax, eta_Mmax, r_Cmax, r_Tmax, r_Mmax] = results_max
    [etamin, rmin, Pmin, Tmin, cmin, dPdetamin, dTdetamin, dPdtmin, dPdrmin, dTdtmin, dTdrmin, drhodtmin, drhodrmin, rhomin, eta_wmin, eta_Cmin, eta_Tmin, eta_Mmin, r_Cmin, r_Tmin, r_Mmin] = results_min
    
    col_list = ['g','b','r','y','c','m']
    col = col_list[0:N]
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1,sharex=True,figsize=(8,9), height_ratios=[3,3,3,1])
    fig.subplots_adjust(hspace=0)

    for i in range(N):
        j = N-i-1
        ax1.semilogx(r[j], P[j]*1.e-6,'-',color=col[j])
        ax2.semilogx(r[j], T[j]-273.15,'-',label=tlabels[j],color=col[j])
        ax3.semilogx(r[j], rho[j], '-', color=col[j])
        ax4.semilogx(r[j],c[j],'-',color=col[j])
        ax1.semilogx(rmax[j], Pmax[j]*1.e-6,'--',color=col[j])
        ax2.semilogx(rmax[j], Tmax[j]-273.15,'--',color=col[j])
        ax3.semilogx(rmax[j], rhomax[j], '--', color=col[j])
        ax4.semilogx(rmax[j],cmax[j],'--',color=col[j])
        ax1.semilogx(rmin[j], Pmin[j]*1.e-6,':',color=col[j])
        ax2.semilogx(rmin[j], Tmin[j]-273.15,':',color=col[j])
        ax3.semilogx(rmin[j], rhomin[j], ':', color=col[j])
        ax4.semilogx(rmin[j],cmin[j],':',color=col[j])
    ax1.semilogx(r[0], P[0]*1.e-6,'-',label='{}={}'.format(df1.columns[index_to_vary],resprops[index_to_vary]),color=col[j])
    ax1.semilogx(rmax[0], Pmax[0]*1.e-6,'--',label='{}={}'.format(df1.columns[index_to_vary],respropsmax[index_to_vary]),color=col[j])
    ax1.semilogx(rmin[0], Pmin[0]*1.e-6,':',label='{}={}'.format(df1.columns[index_to_vary],respropsmin[index_to_vary]),color=col[j])

    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=3, fontsize=16)
    ax1.grid('on',linestyle=':',alpha=0.6)
    #ax1.set_ylim([1.5,4.9])
    ax1.tick_params(axis='both', which='major', labelsize=14, direction='in', length=5)
    ax1.tick_params(axis='both', which='minor', direction='in', length=3)
    ax1.set_ylabel("p (MPa)",fontsize=18)
    
    handles, axlabels = ax2.get_legend_handles_labels()
    ax2.legend(handles[::-1], axlabels[::-1], fontsize=16, loc='lower right')
    ax2.grid('on',linestyle=':',alpha=0.6)
    #ax2.set_ylim([min([min(T[i]) for i in range(N)])-273.15 - 2,max([max(T[i]) for i in range(N)])-273.15 + 2])
    ax2.tick_params(axis='both', which='major', labelsize=14, direction='in', length=5)
    ax2.tick_params(axis='both', which='minor', direction='in', length=3)
    ax2.set_ylabel("T ($^{\circ}$C)",fontsize=18)
    
    ax3.grid('on',linestyle=':',alpha=0.6)
    # ax3.set_xlim([8.e-5,10])
    # ax3.set_ylim([1.5,6.8])
    ax3.tick_params(axis='both', which='major', labelsize=14, direction='in', length=5)
    ax3.tick_params(axis='both', which='minor', direction='in', length=3)
    # ax3.set_xlabel("$\eta$",fontsize=18)
    # ax3.set_xticks([])
    ax3.set_ylabel(r"$\rho$ (kg m$^{-3}$)",fontsize=18)

    ax4.grid('on',linestyle=':',alpha=0.6)
    ax4.tick_params(axis='both', which='major', labelsize=14, direction='in', length=5)
    ax4.tick_params(axis='both', which='minor', direction='in', length=3)
    ax4.set_xlim([1.e-1,2.e3])
    ax4.set_ylim([-0.1,1.3])
    ax4.set_xlabel("$r \ (m)$",fontsize=18)
    ax4.set_ylabel("$c$",fontsize=18)
    
    plt.savefig(path +'/'+ 'fig5.pdf',dpi=300, bbox_inches='tight') 

def rates_plot():
    col_list = ['g','b','r','y','c','m']
    col = col_list[0:N]
    dPdt = np.array([-eta[i]/(2*t[i])*dPdeta[i] for i in range(N)],dtype=object)
    DPDt = np.array([-(eta[i]/(2*t[i]) + beta_f/t[i]*dPdeta[i])*dPdeta[i] for i in range(N)],dtype=object)
    dTdt = np.array([(eta_w[i]-eta[i])/(2*t[i])*dTdeta[i] for i in range(N)],dtype=object)
    DTDt = np.array([((eta_w[i]-eta[i])/(2*t[i]) - beta_f/t[i]*dPdeta[i])*dTdeta[i] for i in range(N)],dtype=object)
    drhodt = np.array([rho[i]*(beta_f*dPdt[i] - alpha_f*dTdt[i]) for i in range(N)],dtype=object)
    DrhoDt = np.array([rho[i]*(beta_f*DPDt[i] - alpha_f*DTDt[i]) for i in range(N)],dtype=object)

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2, sharex=True, figsize=(18,9))
    fig.subplots_adjust(hspace=0)

    ax1.axvline(eta_C[0],color='k',alpha=0.1,linewidth=3)
    ax1.axvline(eta_M[0],linestyle='-',color='sandybrown',alpha=0.3,linewidth=3)
    ax1.axvline(eta_T[0],linestyle='-',color='cornflowerblue',alpha=0.3,linewidth=3)
    ax2.axvline(eta_C[0],color='k',alpha=0.1,linewidth=3)
    ax2.axvline(eta_M[0],linestyle='-',color='sandybrown',alpha=0.3,linewidth=3)
    ax2.axvline(eta_T[0],linestyle='-',color='cornflowerblue',alpha=0.3,linewidth=3)
    ax3.axvline(eta_C[0],color='k',alpha=0.1,linewidth=3)
    ax3.axvline(eta_M[0],linestyle='-',color='sandybrown',alpha=0.3,linewidth=3)
    ax3.axvline(eta_T[0],linestyle='-',color='cornflowerblue',alpha=0.3,linewidth=3)
    ax4.axvline(eta_C[0],color='k',alpha=0.1,linewidth=3)
    ax4.axvline(eta_M[0],linestyle='-',color='sandybrown',alpha=0.3,linewidth=3)
    ax4.axvline(eta_T[0],linestyle='-',color='cornflowerblue',alpha=0.3,linewidth=3)
    ax5.axvline(eta_C[0],color='k',alpha=0.1,linewidth=3)
    ax5.axvline(eta_M[0],linestyle='-',color='sandybrown',alpha=0.3,linewidth=3)
    ax5.axvline(eta_T[0],linestyle='-',color='cornflowerblue',alpha=0.3,linewidth=3)
    ax6.axvline(eta_C[0],color='k',alpha=0.1,linewidth=3)
    ax6.axvline(eta_M[0],linestyle='-',color='sandybrown',alpha=0.3,linewidth=3)
    ax6.axvline(eta_T[0],linestyle='-',color='cornflowerblue',alpha=0.3,linewidth=3)

    for i in range(N):
        ax1.loglog(eta[i], dPdt[i], '-', c=col[i],label=tlabels[i])
        ax2.semilogx(eta[i], np.sign(DPDt[i])*np.log10(abs(DPDt[i])+1),'-',c=col[i])#np.sign(DPDt[i])*np.log10(abs(DPDt[i])+1)
        ax3.semilogx(eta[i], np.sign(dTdt[i])*np.log10(1.e5*abs(dTdt[i])+1),'-',c=col[i])#np.sign(dTdt[i])*np.log10(abs(dTdt[i])+1)
        ax4.semilogx(eta[i], np.sign(DTDt[i])*np.log10(1.e4*abs(DTDt[i])+1),'-',c=col[i])
        ax5.loglog(eta[i], drhodt[i],'-',c=col[i],label=tlabels[i])
        ax6.semilogx(eta[i], np.sign(DrhoDt[i])*np.log10(1.e4*abs(DrhoDt[i])+1),'-',c=col[i])

    ax1.legend(fontsize=16, loc='lower left')
    ax1.grid('on',linestyle=':',alpha=0.6)
    # ax1.set_ylim([1.5,6.8])
    ax1.tick_params(axis='both', which='major', labelsize=14, direction='in', length=5)
    ax1.tick_params(axis='both', which='minor', direction='in', length=3)
    ax1.set_ylabel(r"d$p$/d$t$ (Pa s$^{-1}$)",fontsize=18)
    ax1.set_title("In solid frame",fontsize=20)

    ax2.grid('on',linestyle=':',alpha=0.6)
    # ax2.set_ylim([-1,1])
    ax2.tick_params(axis='both', which='major', labelsize=14, direction='in', length=5)
    ax2.tick_params(axis='both', which='minor', direction='in', length=3)
    ax2.set_yticks([2,0,-2,-4],['10$^2$','0','-10$^2$','-10$^4$'])
    ax2.set_ylabel(r"D$p$/D$t$ (kPa s$^{-1}$)",fontsize=18)
    ax2.set_title("In fluid frame",fontsize=20)

    ax3.grid('on',linestyle=':',alpha=0.6)
    # ax3.set_ylim([1.5,6.8])
    ax3.tick_params(axis='both', which='major', labelsize=14, direction='in', length=5)
    ax3.tick_params(axis='both', which='minor', direction='in', length=3)
    ax3.set_yticks([-3.00043408,-2.00432137,-1.04139269,0.,1.04139269],['-10$^{-2}$','-10$^{-3}$','-10$^{-4}$','0','10$^{-4}$'])
    ax3.set_ylabel(r"d$T$/d$t$ ($^{\circ}$C s$^{-1}$)",fontsize=18)

    ax4.grid('on',linestyle=':',alpha=0.6)
    # ax4.set_ylim([1.5,6.8])
    ax4.tick_params(axis='both', which='major', labelsize=14, direction='in', length=5)
    ax4.tick_params(axis='both', which='minor', direction='in', length=3)
    ax4.set_yticks([-4.00004343, -2.00432137, 0.,2.00432137,4.00004343],['-10$^0$','-10$^{-2}$','0','10$^{-2}$','10$^0$'])
    ax4.set_ylabel(r"D$T$/D$t$ ($^{\circ}$C s$^{-1}$)",fontsize=18)

    ax5.grid('on',linestyle=':',alpha=0.6)
    ax5.set_xlim([6.e-5,10])
    ax5.tick_params(axis='both', which='major', labelsize=14, direction='in', length=5)
    ax5.tick_params(axis='both', which='minor', direction='in', length=3)
    ax5.set_ylabel(r"d$\rho$/d$t$ (kg m$^{-3}$ s$^{-1}$)",fontsize=18)
    ax5.set_xlabel("$\eta$",fontsize=18)

    ax6.grid('on',linestyle=':',alpha=0.6)
    ax6.set_xlim([6.e-5,10])
    ax6.tick_params(axis='both', which='major', labelsize=14, direction='in', length=5)
    ax6.tick_params(axis='both', which='minor', direction='in', length=3)
    ax6.set_yticks([-4.00004343, -2.00432137, 0.],['-10$^0$','-10$^{-2}$','0'])
    ax6.set_ylabel(r"D$\rho$/D$t$ (kg m$^{-3}$ s$^{-1}$)",fontsize=18)
    ax6.set_xlabel("$\eta$",fontsize=18)

    plt.savefig(path +'/'+ 'fig3.pdf',dpi=300, bbox_inches='tight') 

def phase_boundaries():
    pt_df = pd.read_excel('data.xlsx',sheet_name='CO2 phase boundary')
    P_pt = pt_df['P (MPa)'].to_numpy()
    T_pt = pt_df['T (C)'].to_numpy()

    ice = pd.read_excel('data.xlsx',sheet_name='Ice')
    P_iceraw = ice['P (MPa)'].to_numpy()
    T_iceraw = ice['T (C)'].to_numpy()
    spl_ice = UnivariateSpline(P_iceraw,T_iceraw,s=10)
    P_ice = np.linspace(0,22,101)
    T_ice = spl_ice(P_ice)

    CO2 = pd.read_excel('data.xlsx',sheet_name='CO2 hydrate')
    P_CO2raw = CO2['P (MPa)'].to_numpy()
    T_CO2raw = CO2['T (C)'].to_numpy()
    spl_CO2 = UnivariateSpline(P_CO2raw,T_CO2raw,s=10)
    P_CO2 = np.linspace(0,22,101)
    T_CO2 = spl_CO2(P_CO2)
    
    CH4 = pd.read_excel('data.xlsx',sheet_name='CH4 hydrate')
    P_CH4raw = CH4['P (MPa)'].to_numpy()
    T_CH4raw = CH4['T (C)'].to_numpy()
    spl_CH4 = UnivariateSpline(P_CH4raw,T_CH4raw,s=10)
    P_CH4 = np.linspace(0,22,101)
    T_CH4 = spl_CH4(P_CH4)

    return P_pt, T_pt, P_ice, T_ice, P_CO2, T_CO2, P_CH4, T_CH4

def PT_plot():
    col_list = ['g','b','r','y','c','m']
    col = col_list[0:N]
    P_pt, T_pt, P_ice, T_ice, P_CO2, T_CO2, P_CH4, T_CH4 = phase_boundaries()
    
    fig = plt.figure(figsize=(8,8))
    
    plt.plot(T_pt,P_pt, '-',linewidth=3,color='k')
    plt.plot([30.978,130],[7.3773,7.3773], '--',linewidth=3,color='k')
    plt.plot([30.978,30.978],[7.3773,22], '--',linewidth=3,color='k')
    plt.fill_betweenx(P_CH4,T_CH4,-20*np.ones(len(P_CH4)),color='aliceblue')
    plt.fill_betweenx(P_CH4,T_CH4,T_CO2,color='aliceblue')
    plt.fill_betweenx(P_CO2,T_CO2,T_ice,color='aliceblue')
    plt.plot(T[0][0]-273.15,P[0][0]*1.e-6,'ko',markersize=10,fillstyle='none',label='well-bore')
    for i in range(N):
        plt.plot(T[i]-273.15,P[i]*1.e-6,'-',c=col[i],label=tlabels[i])
        plt.plot(T[i][0]-273.15,P[i][0]*1.e-6,'o',markersize=10,fillstyle='none',c=col[i])
    plt.plot(T_CH4,P_CH4,':',c='slateblue')
    plt.plot(T_CO2,P_CO2,':',c='teal')
    plt.plot(T_ice,P_ice,'g:')
    plt.text(T_CH4[32]-0.8,P_CH4[32],'CH4-hydrate',rotation=85,horizontalalignment='center',c='slateblue',fontsize=12)
    plt.text(T_CO2[22],P_CO2[22],'CO2-hydrate',rotation='vertical',horizontalalignment='right',c='teal',fontsize=12)
    plt.text(T_ice[32],P_ice[32],'H2O-ice',rotation='vertical',horizontalalignment='right',c='green',fontsize=12)
    plt.plot(T[0][-1]-273.15,P[0][-1]*1.e-6,'ko',markersize=10,label='far-field reservoir')
    handles, labels = plt.gca().get_legend_handles_labels()
    order=[i+1 for i in range(N)]+[0,N+1]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],bbox_to_anchor=(1,1),facecolor='white',frameon=False,fontsize=16)#bbox_to_anchor=(1,1)
    plt.xlim([-20,130])
    plt.ylim([0,10])
    plt.xlabel("$T$ ($^{\circ}$C)",fontsize=18)
    plt.ylabel("$P$ (MPa)",fontsize=18)
    
    plt.savefig(path +'/'+ 'fig4.pdf',dpi=300, bbox_inches='tight') 

def PT_plot_paramvar(results_max,results_min):
    [etamax, rmax, Pmax, Tmax, cmax, dPdetamax, dTdetamax, dPdtmax, dPdrmax, dTdtmax, dTdrmax, drhodtmax, drhodrmax, rhomax, eta_wmax, eta_Cmax, eta_Tmax, eta_Mmax, r_Cmax, r_Tmax, r_Mmax] = results_max
    [etamin, rmin, Pmin, Tmin, cmin, dPdetamin, dTdetamin, dPdtmin, dPdrmin, dTdtmin, dTdrmin, drhodtmin, drhodrmin, rhomin, eta_wmin, eta_Cmin, eta_Tmin, eta_Mmin, r_Cmin, r_Tmin, r_Mmin] = results_min
    
    col_list = ['g','b','r','y','c','m']
    col = col_list[0:N]
    P_pt, T_pt, P_ice, T_ice, P_CO2, T_CO2, P_CH4, T_CH4 = phase_boundaries()
    
    fig = plt.figure(figsize=(8,8))
    
    plt.plot(T_pt,P_pt, '-',linewidth=3,color='k')
    plt.plot([30.978,130],[7.3773,7.3773], '--',linewidth=3,color='k')
    plt.plot([30.978,30.978],[7.3773,22], '--',linewidth=3,color='k')
    plt.fill_betweenx(P_CH4,T_CH4,-20*np.ones(len(P_CH4)),color='aliceblue')
    plt.fill_betweenx(P_CH4,T_CH4,T_CO2,color='aliceblue')
    plt.fill_betweenx(P_CO2,T_CO2,T_ice,color='aliceblue')
    plt.plot(T[0][0]-273.15,P[0][0]*1.e-6,'ko',markersize=10,fillstyle='none',label='well-bore')
    plt.plot(T[0]-273.15,P[0]*1.e-6,'-',c=col[0],label='{}={}'.format(df1.columns[index_to_vary],resprops[index_to_vary]))
    plt.plot(Tmax[0]-273.15,Pmax[0]*1.e-6,'--',c=col[0],label='{}={}'.format(df1.columns[index_to_vary],respropsmax[index_to_vary]))
    plt.plot(Tmin[0]-273.15,Pmin[0]*1.e-6,':',c=col[0],label='{}={}'.format(df1.columns[index_to_vary],respropsmin[index_to_vary]))
    for i in range(N):
        plt.plot(T[i]-273.15,P[i]*1.e-6,'-',c=col[i],label=tlabels[i])
        plt.plot(T[i][0]-273.15,P[i][0]*1.e-6,'o',markersize=10,fillstyle='none',c=col[i])
        plt.plot(Tmax[i]-273.15,Pmax[i]*1.e-6,'--',c=col[i])
        plt.plot(Tmax[i][0]-273.15,Pmax[i][0]*1.e-6,'o',markersize=10,fillstyle='none',c=col[i])
        plt.plot(Tmin[i]-273.15,Pmin[i]*1.e-6,':',c=col[i])
        plt.plot(Tmin[i][0]-273.15,Pmin[i][0]*1.e-6,'o',markersize=10,fillstyle='none',c=col[i])
    plt.plot(T_CH4,P_CH4,':',c='slateblue')
    plt.plot(T_CO2,P_CO2,':',c='teal')
    plt.plot(T_ice,P_ice,'g:')
    plt.text(T_CH4[32]-0.8,P_CH4[32],'CH4-hydrate',rotation=85,horizontalalignment='center',c='slateblue',fontsize=12)
    plt.text(T_CO2[22],P_CO2[22],'CO2-hydrate',rotation='vertical',horizontalalignment='right',c='teal',fontsize=12)
    plt.text(T_ice[32],P_ice[32],'H2O-ice',rotation='vertical',horizontalalignment='right',c='green',fontsize=12)
    plt.plot(T[0][-1]-273.15,P[0][-1]*1.e-6,'ko',markersize=10,label='far-field reservoir')
    handles, labels = plt.gca().get_legend_handles_labels()
    order=[1,2,3]+[i+4 for i in range(N)]+[0,N+3+1]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],bbox_to_anchor=(1,1),facecolor='white',frameon=False,fontsize=16)#bbox_to_anchor=(1,1)
    plt.xlim([-20,130])
    plt.ylim([0,10])
    plt.xlabel("$T$ ($^{\circ}$C)",fontsize=18)
    plt.ylabel("$P$ (MPa)",fontsize=18)
    
    plt.savefig(path +'/'+ 'fig6.pdf',dpi=300, bbox_inches='tight')

# Plotting
if plots_option not in ["Y", "N"]:
    print("Incorrect option selected for 'Generate plots?'. Enter Y or N")
elif plots_option=="Y":
    print('\nPlotting solutions')
    eta_plot()
    Tmax_plot()
    r_plot()
    rates_plot()
    PT_plot()
    if paramsweep_option=="Y":
        r_plot_paramvar(results_max,results_min)
        PT_plot_paramvar(results_max,results_min)

# Generating contour plots
if contourplots_option not in ["Y", "N"]:
    print("Incorrect option selected for 'Generate thermodynamic contour plots?'. Enter Y or N")
elif contourplots_option=="Y":
    print('\nGenerating contour plots')
    Tcon = np.arange(273.15-20,273.15+120.5,0.5)
    Pcon = np.arange(1e6,20.1e6,0.1e6)
    Tgrid, Pgrid = np.meshgrid(Tcon,Pcon,indexing='xy')
    Tco = Tgrid.reshape(len(Pcon)*len(Tcon))
    Pco = Pgrid.reshape(len(Pcon)*len(Tcon))

    fluid = 'CarbonDioxide'
    
    rho = np.empty(Tgrid.shape)
    for i in range(len(Pcon)):
        for j in range(len(Tcon)):
            rho[i,j] = CP.PropsSI('Dmass','P',Pcon[i],'T',Tcon[j],fluid)
    rhoco = rho.reshape(len(Pcon)*len(Tcon))
    
    beta = np.empty(Tgrid.shape)
    for i in range(len(Pcon)):
        for j in range(len(Tcon)):
            beta[i,j] = CP.PropsSI('isothermal_compressibility','P',Pcon[i],'T',Tcon[j],fluid)
    betaco = beta.reshape(len(Pcon)*len(Tcon))
    
    alpha = np.empty(Tgrid.shape)
    for i in range(len(Pcon)):
        for j in range(len(Tcon)):
            alpha[i,j] = CP.PropsSI('isobaric_expansion_coefficient','P',Pcon[i],'T',Tcon[j],fluid)
    alphaco = alpha.reshape(len(Pcon)*len(Tcon))
    
    cp = np.empty(Tgrid.shape)
    for i in range(len(Pcon)):
        for j in range(len(Tcon)):
            cp[i,j] = CP.PropsSI('Cpmass','P',Pcon[i],'T',Tcon[j],fluid)
    cpco = cp.reshape(len(Pcon)*len(Tcon))
    
    JT = np.empty(Tgrid.shape)
    for i in range(len(Pcon)):
        for j in range(len(Tcon)):
            JT[i,j] = CP.PropsSI('d(T)/d(P)|H','P',Pcon[i],'T',Tcon[j],fluid)
    JTco = JT.reshape(len(Pcon)*len(Tcon))
    
    mu = np.empty(Tgrid.shape)
    for i in range(len(Pcon)):
        for j in range(len(Tcon)):
            mu[i,j] = CP.PropsSI('V','P',Pcon[i],'T',Tcon[j],fluid)
    muco = mu.reshape(len(Pcon)*len(Tcon))

    Ppb, Tpb, P_ice, T_ice, P_CO2, T_CO2, P_CH4, T_CH4 = phase_boundaries()

    fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,figsize=(30,15))
    fig.tight_layout(pad=6.0)
    
    # ax.tricontour(T, P, alpha/beta, levels=[467.6], linewidths=0.5, colors='k')
    cntr1 = ax1.tricontourf(Tco-273.15, Pco*1.e-6, rhoco, levels=60, cmap="RdBu_r")
    ax1.plot(Tpb,Ppb,'k-',linewidth=5)
    ax1.plot([30.978,120],[7.3773,7.3773], 'k--',linewidth=4)
    ax1.plot([30.978,30.978],[7.3773,20], 'k--',linewidth=4)
    ax1.plot(Tref-273.15,Pref*1.e-6,'ko',markerfacecolor='w',markersize=10)
    ax1.annotate('Tr, Pr', (Tref-273.15+2, Pref*1.e-6+0.05),fontsize=16)
    ax1.plot(T0-273.15,P0*1.e-6,'ko',markerfacecolor='w',markersize=10)
    ax1.annotate('T0, P0', (T0-273.15+2, P0*1.e-6+0.05),fontsize=16)
    # ax.plot(30.978,7.3773,'kx',markersize=10)
    ax1.set_title('Density (kg m$^{-3}$)',fontsize=22)
    fig.colorbar(cntr1, ax=ax1)
    ax1.set_ylim([1,20])
    ax1.set_xlim([-20,120])
    ax1.set_xlabel('T ($^{\circ}$C)', fontsize=18)
    ax1.set_ylabel('P (MPa)', fontsize=18)
    
    levels = np.linspace(0,1.5e-6,60)
    cntr2 = ax2.tricontourf(Tco-273.15, Pco*1.e-6, betaco, levels=levels, cmap="RdBu_r")
    ax2.plot(Tpb,Ppb,'k-',linewidth=5)
    ax2.plot([30.978,120],[7.3773,7.3773], 'k--',linewidth=4)
    ax2.plot([30.978,30.978],[7.3773,20], 'k--',linewidth=4)
    ax2.plot(Tref-273.15,Pref*1.e-6,'ko',markerfacecolor='w',markersize=10)
    ax2.annotate('Tr, Pr', (Tref-273.15+2, Pref*1.e-6+0.05),fontsize=16)
    ax2.plot(T0-273.15,P0*1.e-6,'ko',markerfacecolor='w',markersize=10)
    ax2.annotate('T0, P0', (T0-273.15+2, P0*1.e-6+0.05),fontsize=16)
    # ax.plot(30.978,7.3773,'kx',markersize=10)
    ax2.set_title('Compressibility (Pa$^{-1}$)',fontsize=22)
    fig.colorbar(cntr2, ax=ax2)
    ax2.set_ylim([1,20])
    ax2.set_xlim([-20,120])
    ax2.set_xlabel('T ($^{\circ}$C)', fontsize=18)
    ax2.set_ylabel('P (MPa)', fontsize=18)
    
    levels = np.linspace(0,0.25,60)
    cntr3 = ax3.tricontourf(Tco-273.15, Pco*1.e-6, alphaco, levels=levels, cmap="RdBu_r")
    ax3.plot(Tpb,Ppb,'k-',linewidth=5)
    ax3.plot([30.978,120],[7.3773,7.3773], 'k--',linewidth=4)
    ax3.plot([30.978,30.978],[7.3773,20], 'k--',linewidth=4)
    ax3.plot(Tref-273.15,Pref*1.e-6,'ko',markerfacecolor='w',markersize=10)
    ax3.annotate('Tr, Pr', (Tref-273.15+2, Pref*1.e-6+0.05),fontsize=16)
    ax3.plot(T0-273.15,P0*1.e-6,'ko',markerfacecolor='w',markersize=10)
    ax3.annotate('T0, P0', (T0-273.15+2, P0*1.e-6+0.05),fontsize=16)
    # ax.plot(30.978,7.3773,'kx',markersize=10)
    ax3.set_title('Expansivity (K$^{-1}$)',fontsize=22)
    fig.colorbar(cntr3, ax=ax3)
    ax3.set_ylim([1,20])
    ax3.set_xlim([-20,120])
    ax3.set_xlabel('T ($^{\circ}$C)', fontsize=18)
    ax3.set_ylabel('P (MPa)', fontsize=18)
    
    levels = np.linspace(500,30000,60)
    cntr4 = ax4.tricontourf(Tco-273.15, Pco*1.e-6, cpco, levels=levels, cmap="RdBu_r")
    ax4.plot(Tpb,Ppb,'k-',linewidth=5)
    ax4.plot([30.978,120],[7.3773,7.3773], 'k--',linewidth=4)
    ax4.plot([30.978,30.978],[7.3773,20], 'k--',linewidth=4)
    ax4.plot(Tref-273.15,Pref*1.e-6,'ko',markerfacecolor='w',markersize=10)
    ax4.annotate('Tr, Pr', (Tref-273.15+2, Pref*1.e-6+0.05),fontsize=16)
    ax4.plot(T0-273.15,P0*1.e-6,'ko',markerfacecolor='w',markersize=10)
    ax4.annotate('T0, P0', (T0-273.15+2, P0*1.e-6+0.05),fontsize=16)
    # ax.plot(30.978,7.3773,'kx',markersize=10)
    ax4.set_title('Heat Capacity (J kg$^{-1}$ K$^{-1}$)',fontsize=22)
    fig.colorbar(cntr4, ax=ax4)
    ax4.set_ylim([1,20])
    ax4.set_xlim([-20,120])
    ax4.set_xlabel('T ($^{\circ}$C)', fontsize=18)
    ax4.set_ylabel('P (MPa)', fontsize=18)
    
    cntr5 = ax5.tricontourf(Tco-273.15, Pco*1.e-6, JTco, levels=60, cmap="RdBu_r")
    ax5.plot(Tpb,Ppb,'k-',linewidth=5)
    ax5.plot([30.978,120],[7.3773,7.3773], 'k--',linewidth=4)
    ax5.plot([30.978,30.978],[7.3773,20], 'k--',linewidth=4)
    ax5.plot(Tref-273.15,Pref*1.e-6,'ko',markerfacecolor='w',markersize=10)
    ax5.annotate('Tr, Pr', (Tref-273.15+2, Pref*1.e-6+0.05),fontsize=16)
    ax5.plot(T0-273.15,P0*1.e-6,'ko',markerfacecolor='w',markersize=10)
    ax5.annotate('T0, P0', (T0-273.15+2, P0*1.e-6+0.05),fontsize=16)
    # ax.plot(30.978,7.3773,'kx',markersize=10)
    ax5.set_title('Joule-Thomson Coefficient (K Pa$^{-1}$)',fontsize=22)
    fig.colorbar(cntr5, ax=ax5)
    ax5.set_ylim([1,20])
    ax5.set_xlim([-20,120])
    ax5.set_xlabel('T ($^{\circ}$C)', fontsize=18)
    ax5.set_ylabel('P (MPa)', fontsize=18)
    
    cntr6 = ax6.tricontourf(Tco-273.15, Pco*1.e-6, muco, levels=60, cmap="RdBu_r")
    ax6.plot(Tpb,Ppb,'k-',linewidth=5)
    ax6.plot([30.978,120],[7.3773,7.3773], 'k--',linewidth=4)
    ax6.plot([30.978,30.978],[7.3773,20], 'k--',linewidth=4)
    ax6.plot(Tref-273.15,Pref*1.e-6,'ko',markerfacecolor='w',markersize=10)
    ax6.annotate('Tr, Pr', (Tref-273.15+2, Pref*1.e-6+0.05),fontsize=16)
    ax6.plot(T0-273.15,P0*1.e-6,'ko',markerfacecolor='w',markersize=10)
    ax6.annotate('T0, P0', (T0-273.15+2, P0*1.e-6+0.05),fontsize=16)
    # ax.plot(30.978,7.3773,'kx',markersize=10)
    ax6.set_title('Viscosity (Pa s)',fontsize=22)
    fig.colorbar(cntr6, ax=ax6)
    ax6.set_ylim([1,20])
    ax6.set_xlim([-20,120])
    ax6.set_xlabel('T ($^{\circ}$C)', fontsize=18)
    ax6.set_ylabel('P (MPa)', fontsize=18)
    
    plt.savefig(path +'/'+ 'fig7.pdf',dpi=300, bbox_inches='tight')