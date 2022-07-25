
#%% Import 
import numpy as np
import sympy
from matplotlib import pyplot as plt
from scipy import integrate

#%% sympy define symbols and functions
t,tau_l0,Chi = sympy.symbols("t, τ_l^0, χ")
Pi0,beta_Pi,C_a0, C_l0 = sympy.symbols("Π^0, β_Π, C_a^0, C_l^0")
gam,k_a,kappa_0,EC,delta = sympy.symbols ("γ,k_a,κ_0,ηC,δ")
tau_l = sympy.symbols("τ_l")
R_h = sympy.symbols("R_h")
P = sympy.symbols("P")
c_star,lam,beta,E_H,lam_star= sympy.symbols("c_*,λ,β,η_H,λ_*")

C_a = sympy.Function("C_a")
C_s = sympy.Function("C_s")
C_d = sympy.Function("C_d")
C_l = sympy.Function("C_l")
J   = sympy.Function("J")
T_s = sympy.Function("T_s")
T_d = sympy.Function("T_d")

# %% Define the six equations
yr2s = 1   # change year to seconds, but here we don't use it. 

# 1. Energy Ts
ode_Ts = 1 / (delta*c_star) * (-lam*T_s(t*yr2s) + beta*sympy.log(C_a(t)/C_a0 + 1) 
                               - E_H*(T_s(t*yr2s)-T_d(t*yr2s)) 
                               - lam_star * (T_s(t*yr2s) - T_d(t*yr2s)))

ode_Ts   # here the sympy output even the same equations as the Latex. but it's not the excutable equation yet.
# %% 2. Energy Td
ode_Td = E_H*(T_s(t*yr2s)-T_d(t*yr2s))/((1-delta) * c_star)
ode_Td
# %% 3. function 
tau_l = tau_l0 * Chi ** (-T_s(t)/10)
R_h = (C_l0+C_l(t))/tau_l
P = Pi0 * (1+beta_Pi*sympy.ln(C_a(t)/C_a0 +1))
ode_Cl = P - R_h
ode_Cs = gam * (C_a(t)/k_a - C_s(t)/kappa_0) - EC*(C_s(t)/delta - C_d(t)/(1-delta))
ode_Cd = EC*(C_s(t)/delta - C_d(t)/(1-delta))

# %% 6. function of Ca
A_tot,t_opt = sympy.symbols ("A_tot,t_opt")
A = sympy.Function("A")
A = A_tot*( 1/(1+2.5*sympy.exp((t_opt-t)/50)) - 1/(1+2.5*sympy.exp(t_opt/50))) # rcp8.5 emission
A_ = []
for i in range(600):
    A_.append(A.subs({t:i,t_opt:250,A_tot:5000}))
plt.plot(A_)

# %%
J=A.diff(t)
J
# %%
ode_Ca = J - ode_Cl-ode_Cs-ode_Cd
ode_Ca
# %% Equations To functions
# 上述代码定义了6个方程，接下来，我们要将以上六个方程转变成可以执行的函数，sympy函数sympy.lambdify可以直接调用。
ode_sys = [ode_Ts, ode_Td,ode_Cl, ode_Cs, ode_Cd, ode_Ca]
ode_sys_np = sympy.lambdify(([T_s(t), T_d(t), C_l(t), C_s(t), C_d(t), C_a(t)],
                            t, 
                             c_star, 
                             lam, 
                             beta, 
                             E_H, 
                             lam_star,
                             tau_l0, 
                             Chi, 
                             Pi0, 
                             beta_Pi, 
                             C_a0, 
                             C_l0, 
                             gam, 
                             k_a, 
                             kappa_0, 
                             EC, 
                             delta, 
                             A_tot,
                             t_opt),
                            ode_sys)
# %% Solve the equations 
# 1. initial condition
ics = { 
    'TC0': [0,0,0,0,0,0], # ode_Ts,ode_Td,ode_Cl, ode_Cs, ode_Cd, ode_Ca
                              
    # var     # values.             # unit (origin)
    c_star:   10.8*1e9,             # JK**-1m**-2 
    lam:      1.77*365*24*3600,     # W m**-2K**-1
    beta:     5.77*365*24*3600,     # W m**-2
    E_H:      0.73*(365*24*3600),   # None
    lam_star: 0,                    # None

    tau_l0:   41,                   # yr
    Chi:      1.8,                  # None     
    Pi0:      60 ,                  # GtC/yr 
    beta_Pi:  0.4,                  # None 
    C_a0:     589,                  # GtC
    C_l0:     2500,                 # GtC
    gam:      0.005,                # GtC/yr/ppm
    k_a:      2.12,                 # GtC/ppm
    kappa_0:  1.98,                 # GtC/ppm
    EC:       60*1e-12*365*24*3600, # s ** -1 
    delta:    0.015 ,               # None
    A_tot:    5000,                 # GtC
    t_opt:    250 ,                 # yr   
    
}
t_ = np.arange(0,5000,1)   # year

# 2. Solve the ode
xy_t = integrate.odeint(ode_sys_np, ics['TC0'],
                        t_, 
                        args=(
                        ics[c_star],
                        ics[lam],
                        ics[beta],
                        ics[E_H],
                        ics[lam_star],
                        ics[tau_l0],
                        ics[Chi],
                        ics[Pi0],
                        ics[beta_Pi],
                        ics[C_a0],
                        ics[C_l0],                        
                        ics[gam],
                        ics[k_a],
                        ics[kappa_0],  
                        ics[EC],
                        ics[delta],
                        ics[A_tot],
                        ics[t_opt]
                        ))
xy_t.shape
# %% Results
fig,axes = plt.subplots(3,2,figsize = (12,10),dpi = 100)

colors = ['--k','--b','g','k','b','r']
labels = [r'$T_s$',r'$T_d$',r'$C_l$',r'$C_s$',r'$C_d$',r'$C_a$']

for i,ax in enumerate(axes.flat):
    ax.plot(xy_t[:,i],colors[i],label = labels[i])
    ax.legend(loc = "upper left")
    ax.set_xlabel("year")
    
for ax in axes.flat[:2]:
    ax.set_ylabel("T anomaly (K)")
for ax in axes.flat[2:]:
    ax.set_ylabel("C anomaly (Gt)")

# %%
