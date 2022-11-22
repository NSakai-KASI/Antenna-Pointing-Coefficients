# Written by Nobuyuki Sakai on 4th November 2022@NARIT
# 221104: Minor modifications
# 221104: Correcting bags in subroutines (i.e., definitions of functions) 
# 221110: Adding a subroutine for the calculation of the reduced chi square
# 221114: Modifications for the plot
# 221114: Adding a sigma clip of outlier data in the control file.
# 221116: Sign flip for P5
# 221118: Modification for data loading

# References:
  # https://nbviewer.org/url/jakevdp.github.io/downloads/notebooks/FreqBayes2.ipynb
  # http://jakevdp.github.io/blog/2014/06/14/frequentism-and-bayesianism-4-bayesian-in-python/
  # https://qiita.com/jyoppomu/items/e9a124b0e5d5a3caa78f
  # https://lmfit.github.io/lmfit-py/fitting.html
  # https://gist.github.com/andyfaff/373cc989d3fa2de8260b5b2aa22eb60e
  # For multi independent variables
  # https://stackoverflow.com/questions/32714055/using-multiple-independent-variables-in-python-lmfit
  # Minimization of residual with lmfit & emcee
  #https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.Minimizer.emcee

#Python start
#python3

import numpy as np
import matplotlib.pyplot as plt
import csv
#from matplotlib import pyplot as plt

import statistics
from statistics import stdev
from statistics import median
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import lmfit as lf
from lmfit import Parameter, Parameters, Minimizer
from lmfit import Model,minimize
#Least squares for non-linear functions

#For emcee
import emcee # 2.2.1
import corner # 2.0.1
import progressbar # 3.34.3

#For loading text file
from io import StringIO


################################
#########Phase 0:  Definitions##
################################



def func_4_3(theta, X, data2_id:int):
    P1_1, P2_1, P3_1, P4_1, P5_1, P7_2, P4_2, P5_2, P8_2, P9_2  = ( theta['P1_1'], theta['P2_1'],
                                                                  theta['P3_1'], theta['P4_1'], theta['P5_1'],
                                                                  theta['P7_2'], theta['P4_2'], theta['P5_2'],
        							  theta['P8_2'], theta['P9_2'] )
    if data2_id == 1:
        return delta_Az2(P1_1, P2_1, P3_1, P4_1, P5_1, X)
    if data2_id == 2:
        return delta_El2(P7_2, P4_2, P5_2, P8_2, P9_2, X)



#Method for calculating the residual between a model and data
#Method (=Objective function) should be defined as function(params, *args, **kws)



def objective_3(theta, X:np.ndarray, data2:np.ndarray, uncertainty2:np.ndarray):
    # make residual per data set
    residual = 0.0 * data2
    for n in range(data2.shape[0]):
        residual[n] = ( data2[n] - func_4_3(theta, X, n+1) ) / uncertainty2[n]
    # now flatten this to a 1D array, as minimize() needs
    return residual.flatten()




###############################
##### Likelihood functions ####
###############################


#Az and EL data for Minimizer.emcee
def log_likelihood_5(theta, X:np.ndarray, data2:np.ndarray, uncertainty2:np.ndarray):
    scale_az,scale_el  = (theta['scale_az'],theta['scale_el'])
    El_deg = X[1]
    El_rad = El_deg * np.pi / 180.0
    model_az   = func_4_3(theta, X, 1)
    model_el   = func_4_3(theta, X, 2)
    sigma_az   = uncertainty2[0] ** 2 * (scale_az) ** 2 
    sigma_el   = uncertainty2[1] ** 2 * (scale_el) ** 2 
    return  ( -0.5 *  ( np.sum((data2[0] - model_az) ** 2 / sigma_az + np.log(2 * np.pi * sigma_az ))
                      + np.sum((data2[1] - model_el) ** 2 / sigma_el + np.log(2 * np.pi * sigma_el )) ) )




#########################
#### Prior functions ####
#########################
# Gaussian errors + Systematic (=constant) error

# Az and El data with lmfit.emcee
#def log_prior_5(theta):
    #sys_az, sys_el  = theta['sys_az'], theta['sys_el']
    #return - np.log(sys_az) - np.log(sys_el)


###################
#### Posterior functios
###################
# Gaussian errors + Systematic (=constant) error #


#Az and El data with limfit.emcee
def log_posterior_5(theta, X:np.ndarray, data2:np.ndarray, uncertainty2:np.ndarray):
    #lp = log_prior_5(theta)
    #if not np.isfinite(lp):
    #    return -np.inf
    return log_likelihood_5(theta, X, data2, uncertainty2)




##################
# Functions for antenna pointing
##################
# For the offset of Az (deg)
#6-2

#
def deg_rad_az(X):
    Az_deg = X[0]
    Az_rad = Az_deg * np.pi / 180.0
    return Az_rad


def deg_rad_el(X):
    El_deg = X[1]
    El_rad = El_deg * np.pi / 180.0
    return El_rad



def collimation_term(X):
    El_rad = deg_rad_el(X)
    collimation = 1.0/np.cos(El_rad)
    return collimation


#6-3
def orthogonality_term(X):
    El_rad = deg_rad_el(X)
    orthogonality = np.tan(El_rad)
    return orthogonality


#6-4
def tilt_east(X):
    Az_rad = deg_rad_az(X)
    El_rad = deg_rad_el(X)
    tilt_East = -1.0 * np.cos(Az_rad) * np.tan(El_rad)
    return tilt_East


#6-5
def tilt_south(X):
    Az_rad = deg_rad_az(X)
    El_rad = deg_rad_el(X)
    tilt_South = np.sin(Az_rad) * np.tan(El_rad)
    return tilt_South


#6-6
def delta_Az2(P1, P2, P3, P4, P5, X):
    return ( P1 + P2 * collimation_term(X)
                     + P3 * orthogonality_term(X)
                     + P4 * tilt_east(X)
		     + P5 * tilt_south(X) )


#############################
# For the offset of El (deg)
#############################
#6-7
def tilt_east_south(X):
    Az_rad = deg_rad_az(X)
    tilt_East_South = 1.0 * np.sin(Az_rad)
    return tilt_East_South


#6-8
def tilt_south_east(X):
    Az_rad = deg_rad_az(X)
    #tilt_South_East = -1.0 * np.cos(Az_rad) #Typo in original document
    tilt_South_East = +1.0 * np.cos(Az_rad)  #
    return tilt_South_East


#6-9
def gravitation_term1(X):
    El_rad = deg_rad_el(X)
    gravitation_1 = np.sin(El_rad)
    return gravitation_1


#6-10
def gravitation_term2(X):
    El_rad = deg_rad_el(X)
    gravitation_2 = np.cos(El_rad)
    return gravitation_2


#6-11
def delta_El2(P7, P4, P5, P8, P9, X):
    return ( P7 + P4 * tilt_east_south(X)
                     + P5 * tilt_south_east(X)
                     + P8 * gravitation_term1(X)
		     + P9 * gravitation_term2(X) )






#####################################
######## Phase 1: Data load for all data #########
#####################################
with open("Input_file.txt") as f:
    content = f.read().replace(' #Name of Input file\n', '')



print("Input file:%s"% content)

#Data = np.loadtxt(content, comments="#",delimiter=',',skiprows=1)
Data = np.genfromtxt(content, comments="Az",delimiter=',')




Az = []
El = []

dAz_error = []
dEl_error = []

d_Az = []
d_El = []

line = 1

for i in Data:
    if ( (str(i[0]) == "nan") or (str(i[1]) == "nan") or (str(i[2]) == "nan") or
         (str(i[3]) == "nan") or (str(i[4]) == "nan") or (str(i[5]) == "nan") or
         (str(i[6]) == "nan") or (str(i[7]) == "nan") or (str(i[8]) == "nan") or
         (str(i[9]) == "nan") ):
        line = line + 1
        print('The line %d is skipped'% line)
    else:
       Az.append((float(i[0])+float(i[5]))/2.0)
       El.append((float(i[1])+float(i[6]))/2.0)
       dAz_error.append(float(i[3]))
       dEl_error.append(float(i[8]))
       d_Az.append(float(i[2]))
       d_El.append(float(i[7]))
       line = line + 1



num = len(d_Az)+len(d_El)
print('The number of all data:%d'% num)



######################################################################################
######## Phase 2: Weighted leaset squares for initial estimates of model parameters ##
######################################################################################

Az_deg = np.array(Az)
El_deg = np.array(El)
X = np.array([Az_deg,El_deg])
d_Az = np.array(d_Az)
d_El = np.array(d_El)
dAz_error = np.array(dAz_error)
dEl_error = np.array(dEl_error)
data2 = np.array([d_Az,d_El])
uncertainty2 = np.array([dAz_error,dEl_error])


##############################
# Call for the class object 'Parameters'
##############################
fit_params2 = lf.Parameters()


################################################################################
# Initial values and serach ranges for individual model parameters
#####################################################################

fit_params2.add("P1_1", value=0.1)
fit_params2.add("P2_1", value=0.1)
fit_params2.add("P3_1", value=0.1)
fit_params2.add("P4_1", value=-0.2)
fit_params2.add("P5_1", value=0.1)
fit_params2.add("P7_2", value=-0.1)
fit_params2.add("P4_2", value=-0.2)
fit_params2.add("P5_2", value=0.1)
fit_params2.add("P8_2", value=1.0)
fit_params2.add("P9_2", value=-1.0)


#####################################
#Defenition for a common parameter
######################################

fit_params2[f'P4_2'].expr = f'P4_1'
fit_params2[f'P5_2'].expr = f'P5_1'
fit_params2

###########################################################################
#Minimization with Levenberg-Marquardt algorithm (i.e., the least squares)
#Objective function is minimized with minimize() method
############################################################################

result2 = lf.minimize(objective_3, params=fit_params2, args=(X, data2, uncertainty2),scale_covar=True)

#Confirmation of the results
print(lf.fit_report(result2))

f = open('Least-squares.txt', 'w')
f.write(lf.fit_report(result2))
f.close()

###################
# Calculation of stndard errors for (delta_az - model ) and (delta_el - model)
#####################
num_Az = len(d_Az)
num_El = len(d_El)
d_Az_se   = stdev(d_Az-func_4_3(result2.params, X, 1))/np.sqrt(num_Az)
d_El_se   = stdev(d_El-func_4_3(result2.params, X, 2))/np.sqrt(num_El)



######################
# Clip data with a sigma clip and max/min values of FMHM for Az and EL
######################
Control = np.loadtxt('Control_v2.txt', comments="#")

clip_d_Az = Control[0]
clip_d_El = Control[1]
az_fwhm = Control[2]
el_fwhm = Control[3]
az_min_fwhm = Control[4]
el_min_fwhm = Control[5]


#Data2 = np.loadtxt(content, comments="#", delimiter=',')
Data2 = np.genfromtxt(content, comments="Az",delimiter=',')


Az = []
El = []

dAz_error = []
dEl_error = []

d_Az = []
d_El = []

Az_flag = []
El_flag = []
d_Az_flag = []
d_El_flag = []
dAz_error_flag = []
dEl_error_flag = []

output_flag = []
line = 1

for i in Data2:
    if ( (str(i[0]) == "nan") or (str(i[1]) == "nan") or (str(i[2]) == "nan") or
         (str(i[3]) == "nan") or (str(i[4]) == "nan") or (str(i[5]) == "nan") or
         (str(i[6]) == "nan") or (str(i[7]) == "nan") or (str(i[8]) == "nan") or
         (str(i[9]) == "nan") ):
        line = line + 1
        print('The line %d is skipped'% line)
    else:
         if ( np.abs( ( float(i[2]) - func_4_3(result2.params, [float(i[0]),float(i[1])], 1) ) /
                  (d_Az_se)  )
                  < clip_d_Az and np.abs( ( float(i[7]) - func_4_3(result2.params, [float(i[0]),float(i[1])], 2) ) /
                  (d_El_se) )
                  < clip_d_El and (np.abs(float(i[4])) ) < (az_fwhm) and
                  (np.abs(float(i[9])) ) < (el_fwhm) and (np.abs(float(i[4])) ) > (az_min_fwhm) and
                  (np.abs(float(i[9])) ) > (el_min_fwhm)):
            Az.append((float(i[0])+float(i[5]))/2.0)
            El.append((float(i[1])+float(i[6]))/2.0)
            dAz_error.append(float(i[3]))
            dEl_error.append(float(i[8]))
            d_Az.append(float(i[2]))
            d_El.append(float(i[7]))
            line = line + 1 
         else:
            Az_flag.append((float(i[0])+float(i[5]))/2.0)
            El_flag.append((float(i[1])+float(i[6]))/2.0)
            d_Az_flag.append(float(i[2]))
            d_El_flag.append(float(i[7]))
            dAz_error_flag.append(float(i[3]))
            dEl_error_flag.append(float(i[8]))
            line = line + 1
            output_flag.append([line,float(i[0]),float(i[1]),float(i[2]),float(i[3])
                           ,float(i[4]),float(i[5]),float(i[6]),float(i[7])
                           ,float(i[8]),float(i[9])])




f = open('Output_flag.txt', 'w')
f.write('#line number, Az (deg),El (deg),dAz (arcsec),error_dAz (arcsec),FWHM_az (arcsec),Az (deg),El (deg),dEl (arcsec),error_dEl (arcsec),FWHM_el (arcsec) ')
for x in output_flag:
    f.write(str(x) + "\n")


f.close()

flag = len(Az_flag)+len(El_flag)
print('The number of all data:%d'% num)
print('The number of flagged data:%d'% flag)




Az_deg = np.array(Az)
El_deg = np.array(El)
X = np.array([Az_deg,El_deg])
d_Az = np.array(d_Az)
d_El = np.array(d_El)
dAz_error = np.array(dAz_error)
dEl_error = np.array(dEl_error)
data2 = np.array([d_Az,d_El])
uncertainty2 = np.array([dAz_error,dEl_error])

Az_flag = np.array(Az_flag)
El_flag = np.array(El_flag)
dAz_error_flag = np.array(dAz_error_flag)
dEl_error_flag = np.array(dEl_error_flag)
X2 = np.array([Az_flag,El_flag])
d_Az_flag = np.array(d_Az_flag)
d_El_flag = np.array(d_El_flag)


##############################################################################
######## MCMC fit with y erros + a scaling factor for the error ##########
##############################################################################
nwalkers = 100  # number of MCMC walkers
nburn =  1600   # "burn-in" period to let chains stabilize
nsteps = 8000  # number of MCMC steps to take
thin =   100

emcee_params = result2.params.copy()
emcee_params.add('scale_az', value=0.1, min=0, max=1000)
emcee_params.add('scale_el', value=0.1, min=0, max=1000)



mini2 = Minimizer(log_posterior_5, emcee_params, fcn_args=(X, data2, uncertainty2))


res = mini2.emcee(burn=nburn,steps=nsteps, nwalkers=nwalkers,thin=thin,is_weighted=True,progress=True,float_behavior='posterior')

#print('Auto correlation time for each parameter:')
#res.acor
#auto correlation time for each parameter


#Above means that walkers require 100 steps to forget initial values
#Let’s discard the initial 100 steps, thin by about half the autocorrelation time (50 steps),


print('median of posterior probability distribution')
print('--------------------------------------------')                                   
lf.report_fit(res.params)


#########################################
# Caclulation of the reduced chi-square, AIC and BIC
########################################
k = 10 # The number of model parameters
Az_term     =  (d_Az-func_4_3(res.params, X, 1))**2 / ( (res.params['scale_az'])**2 * uncertainty2[0]**2 )
Az_term_sum = np.sum( (d_Az-func_4_3(res.params, X, 1))**2 / ( (res.params['scale_az'])**2 * uncertainty2[0]**2))
Sigma_az    = ( (res.params['scale_az'])**2 * uncertainty2[0]**2 )
El_term     =  (d_El-func_4_3(res.params, X, 2))**2 / ( (res.params['scale_el'])**2 * uncertainty2[1]**2)
El_term_sum = np.sum( (d_El-func_4_3(res.params, X, 2))**2 / ( (res.params['scale_el'])**2 * uncertainty2[1]**2))
Sigma_el    = ( (res.params['scale_el'])**2 * uncertainty2[1]**2)
Reduced_chi_square = (Az_term_sum + El_term_sum)/((num-flag)-k)


Az_aic = -2 * ( -0.5 * np.sum( Az_term + np.log(2 * np.pi * Sigma_az) ) )
El_aic = -2 * ( -0.5 * np.sum( El_term + np.log(2 * np.pi * Sigma_el) ) )
AIC = Az_aic + El_aic + 2 * k
BIC = Az_aic + El_aic + k * np.log((num-flag))
                                   
print('Reduced chi-square:%.2f'% Reduced_chi_square)
print('AIC:%.1f'% AIC)
print('BIC:%.1f'% BIC)

f = open('MCMC.txt', 'w')
f.write('reduced chi-square:%.2f\n'% Reduced_chi_square)
f.write('AIC:%.1f\n'% AIC)
f.write('BIC:%.1f\n'% BIC)
f.write(lf.fit_report(res.params))
f.close()



###############
# Making plots
################
plt.plot(res.acceptance_fraction, 'o')
plt.xlabel('walker')
plt.ylabel('acceptance fraction')
plt.savefig("Acceptance-fraction.png")
#plt.show()
plt.clf()

import corner
emcee_plot = corner.corner(res.flatchain, labels=res.var_names,quantiles=[0.15865, 0.5, 0.84135])


plt.savefig("Posteriors.png")
#plt.show()
plt.clf()





# Az vs. d_Az | El vs. d_Az
fig, ax = plt.subplots(2,2)
fig.suptitle('Antenna pointing data\n Cyan:Flagged data',fontsize=12)

max_min=np.max([np.abs(d_Az_flag)])
ax[0,0].set_ylim([-max_min,max_min])

ax[0,0].errorbar(Az_flag,d_Az_flag,yerr=dAz_error_flag,fmt=".c", capsize=0,markersize=8)
ax[0,0].errorbar(Az,d_Az,yerr=dAz_error,fmt=".r", capsize=0,markersize=12)
ax[0,0].set_ylabel('$\Delta$Az (arcsec)')
ax[0,0].set_xlabel('Az (deg)')
ax[0,0].tick_params(labelsize=10)

max_min=np.max([np.abs(d_Az)])
ax[1,0].set_ylim([-1.1*max_min,1.1*max_min])

ax[1,0].errorbar(Az_flag,d_Az_flag,yerr=dAz_error_flag,fmt=".c", capsize=0,markersize=8)
ax[1,0].errorbar(Az,d_Az,yerr=dAz_error,fmt=".r", capsize=0,markersize=12)
ax[1,0].set_ylabel('$\Delta$Az (arcsec)')
ax[1,0].set_xlabel('Az (deg)')
ax[1,0].tick_params(labelsize=10)

#Margin
plt.subplots_adjust(hspace=0.45)

# El vs. d_Az
max_min=np.max([np.abs(d_Az_flag)])
ax[0,1].set_ylim([-max_min,max_min])

ax[0,1].errorbar(El_flag,d_Az_flag,yerr=dAz_error_flag,fmt=".c", capsize=0,markersize=8)
ax[0,1].errorbar(El,d_Az,yerr=dAz_error,fmt=".r", capsize=0,markersize=12)
ax[0,1].set_ylabel('$\Delta$Az (arcsec)')
ax[0,1].set_xlabel('El (deg)')
ax[0,1].tick_params(labelsize=10)

max_min=np.max([np.abs(d_Az)])
ax[1,1].set_ylim([-1.1*max_min,1.1*max_min])

ax[1,1].errorbar(El_flag,d_Az_flag,yerr=dAz_error_flag,fmt=".c", capsize=0,markersize=8)
ax[1,1].errorbar(El,d_Az,yerr=dAz_error,fmt=".r", capsize=0,markersize=12)
ax[1,1].set_ylabel('$\Delta$Az (arcsec)')
ax[1,1].set_xlabel('El (deg)')
ax[1,1].tick_params(labelsize=10)

plt.tight_layout()
plt.savefig("d_Az.png")
#plt.show()
plt.clf()

############################
# Az vs. d_El | El vs. d_El
############################
fig, ax = plt.subplots(2,2)
fig.suptitle('Antenna pointing data\n Cyan:Flagged data',fontsize=12)

max_min=np.max([np.abs(d_El_flag)])
ax[0,0].set_ylim([-max_min,max_min])

ax[0,0].errorbar(Az_flag,d_El_flag,yerr=dAz_error_flag,fmt=".c", capsize=0,markersize=8)
ax[0,0].errorbar(Az,d_El,yerr=dAz_error,fmt=".r", capsize=0,markersize=12)
ax[0,0].set_ylabel('$\Delta$El (arcsec)')
ax[0,0].set_xlabel('Az (deg)')
ax[0,0].tick_params(labelsize=10)

max_min=np.max([np.abs(d_El)])
ax[1,0].set_ylim([-1.1*max_min,1.1*max_min])

ax[1,0].errorbar(Az_flag,d_El_flag,yerr=dAz_error_flag,fmt=".c", capsize=0,markersize=8)
ax[1,0].errorbar(Az,d_El,yerr=dAz_error,fmt=".r", capsize=0,markersize=12)
ax[1,0].set_ylabel('$\Delta$El (arcsec)')
ax[1,0].set_xlabel('Az (deg)')
ax[1,0].tick_params(labelsize=10)

#Margin
plt.subplots_adjust(hspace=0.45)

# El vs. d_El
max_min=np.max([np.abs(d_El_flag)])
ax[0,1].set_ylim([-max_min,max_min])

ax[0,1].errorbar(El_flag,d_El_flag,yerr=dAz_error_flag,fmt=".c", capsize=0,markersize=8)
ax[0,1].errorbar(El,d_El,yerr=dAz_error,fmt=".r", capsize=0,markersize=12)
ax[0,1].set_ylabel('$\Delta$El (arcsec)')
ax[0,1].set_xlabel('El (deg)')
ax[0,1].tick_params(labelsize=10)

max_min=np.max([np.abs(d_El)])
ax[1,1].set_ylim([-1.1*max_min,1.1*max_min])

ax[1,1].errorbar(El_flag,d_El_flag,yerr=dAz_error_flag,fmt=".c", capsize=0,markersize=8)
ax[1,1].errorbar(El,d_El,yerr=dAz_error,fmt=".r", capsize=0,markersize=12)
ax[1,1].set_ylabel('$\Delta$El (arcsec)')
ax[1,1].set_xlabel('El (deg)')
ax[1,1].tick_params(labelsize=10)

plt.tight_layout()
plt.savefig("d_El.png")
#plt.show()
plt.clf()





fig, ax = plt.subplots(2,1)
fig.suptitle('Antenna pointing data\n Top:raw data; Bottom:subtraction of a model')

ax[0].errorbar(Az_flag,El_flag, fmt=".c", capsize=0,markersize=8)
ax[0].errorbar(Az,El, fmt=".r", capsize=0,markersize=12)

ax[0].set_ylabel('El (deg)')
ax[0].set_ylim([0,90])
ax[0].set_xlim([0,360])

if np.max([d_Az,d_El]) > 180.0:
      ax[0].quiver(Az,El,d_Az,d_El,angles='xy',scale_units='xy',scale=36.0,color='black')
      ax[0].text(5, 10, "0.5 deg")
else:
      ax[0].quiver(Az,El,d_Az,d_El,angles='xy',scale_units='xy',scale=3.6,color='black')
      ax[0].text(5, 10, "3 arcmin")



x1 = 55 #Endpoint of x
y1 = 5 #Endpoint of y

x2 = 5 #Start of x
y2 = 5 #Start of y

xy = (x1,y1)
xytext = (x2,y2)

ax[0].annotate("" , xy=xy , xytext=xytext , arrowprops=dict(arrowstyle='->',facecolor="black",edgecolor="black") )

ax[1].errorbar(Az_flag,El_flag, fmt=".c", capsize=0,markersize=12)
ax[1].errorbar(Az,El, fmt=".r", capsize=0,markersize=12)
ax[1].set_ylabel('El (deg)')
ax[1].set_xlabel('Az (deg)')

if np.max([d_Az,d_El]) > 180.0:
      ax[1].quiver(Az,El,d_Az-func_4_3(res.params, X, 1),d_El-func_4_3(res.params, X, 2),angles='xy',scale_units='xy',scale=36.0,color='black')
      ax[1].text(5, 10, "0.5 deg")
else:
      ax[1].quiver(Az,El,d_Az-func_4_3(res.params, X, 1),d_El-func_4_3(res.params, X, 2),angles='xy',scale_units='xy',scale=3.6,color='black')
      ax[1].text(5, 10, "3 arcmin")



ax[1].set_ylim([0,90])
ax[1].set_xlim([0,360])

ax[1].annotate("" , xy=xy , xytext=xytext , arrowprops=dict(arrowstyle='->',facecolor="black",edgecolor="black") )


plt.savefig("Az-El.png")
#plt.show()
plt.clf()


fig, ax = plt.subplots(2,2)
fig.suptitle('Antenna pointing data\n Cyan:Flagged data')

ax[0,0].errorbar(d_Az_flag,d_El_flag, yerr=dEl_error_flag,xerr=dAz_error_flag,fmt=".c", capsize=0,markersize=12)
ax[0,0].errorbar(d_Az,d_El,yerr=dEl_error,xerr=dAz_error, fmt=".r", capsize=0,markersize=12)

ax[0,0].set_ylabel('$\Delta$El (arcsec)')
ax[0,0].set_xlabel('$\Delta$Az (arcsec)')
ax[0,0].set_aspect('equal')
ax[0,0].axhline(0.0, ls='--', color='black')
ax[0,0].axvline(0.0, ls='--', color='black')
max_min=np.max([np.abs(d_Az_flag),np.abs(d_El_flag)])
ax[0,0].set_ylim([-max_min,max_min])
ax[0,0].set_xlim([-max_min,max_min])

ax[0,1].errorbar(d_Az_flag,d_El_flag, yerr=dEl_error_flag,xerr=dAz_error_flag,fmt=".c", capsize=0,markersize=12)
ax[0,1].errorbar(d_Az,d_El,yerr=dEl_error,xerr=dAz_error, fmt=".r", capsize=0,markersize=12)

ax[0,1].set_ylabel('$\Delta$El (arcsec)')
ax[0,1].set_xlabel('$\Delta$Az (arcsec)')
ax[0,1].set_aspect('equal')
ax[0,1].axhline(0.0, ls='--', color='black')
ax[0,1].axvline(0.0, ls='--', color='black')
max_min=np.max([np.abs(d_Az),np.abs(d_El)])
ax[0,1].set_ylim([-max_min,max_min])
ax[0,1].set_xlim([-max_min,max_min])

#Margin
plt.subplots_adjust(hspace=0.4)


ax[1,0].errorbar(d_Az_flag-func_4_3(res.params, X2, 1),d_El_flag-func_4_3(res.params, X2, 2), yerr=dEl_error_flag,xerr=dAz_error_flag,fmt=".c", capsize=0,markersize=12)
ax[1,0].errorbar(d_Az-func_4_3(res.params, X, 1),d_El-func_4_3(res.params, X, 2), yerr=dEl_error,xerr=dAz_error,fmt=".r", capsize=0,markersize=12)
ax[1,0].set_ylabel('$\Delta$El - model (arcsec)')
ax[1,0].set_xlabel('$\Delta$Az - model (arcsec)')
ax[1,0].axhline(0.0, ls='--', color='black')
ax[1,0].axvline(0.0, ls='--', color='black')
ax[1,0].set_aspect('equal')
max_min=np.max([np.abs(d_Az_flag),np.abs(d_El_flag)])
ax[1,0].set_ylim([-max_min,max_min])
ax[1,0].set_xlim([-max_min,max_min])

ax[1,1].errorbar(d_Az_flag-func_4_3(res.params, X2, 1),d_El_flag-func_4_3(res.params, X2, 2), yerr=dEl_error_flag,xerr=dAz_error_flag,fmt=".c", capsize=0,markersize=12)
ax[1,1].errorbar(d_Az-func_4_3(res.params, X, 1),d_El-func_4_3(res.params, X, 2), yerr=dEl_error,xerr=dAz_error,fmt=".r", capsize=0,markersize=12)
ax[1,1].set_ylabel('$\Delta$El - model (arcsec)')
ax[1,1].set_xlabel('$\Delta$Az - model (arcsec)')
ax[1,1].axhline(0.0, ls='--', color='black')
ax[1,1].axvline(0.0, ls='--', color='black')
ax[1,1].set_aspect('equal')
max_min=np.max([np.abs(d_Az),np.abs(d_El)])
ax[1,1].set_ylim([-max_min,max_min])
ax[1,1].set_xlim([-max_min,max_min])


plt.savefig("d_Az-d_El.png")
#plt.show()
plt.clf()



###################
# Calculation of mean and stndard deviation for delta az and delta el
#####################

d_Az_mean = np.mean(d_Az)
d_Az_stdev = stdev(d_Az)
d_El_mean = np.mean(d_El)
d_El_stdev = stdev(d_El)
d_Az_mean_res = np.mean(d_Az-func_4_3(res.params, X, 1))
d_Az_stdev_res   = stdev(d_Az-func_4_3(res.params, X, 1))
d_El_mean_res = np.mean(d_El-func_4_3(res.params, X, 2))
d_El_stdev_res   = stdev(d_El-func_4_3(res.params, X, 2))


print('Mean, standard deviation and (standard error)')
print('d_Az:','{:.3g}'.format(d_Az_mean),'+/-','{:.3g}'.format(d_Az_stdev),'(','{:.3g}'.format(d_Az_stdev/(np.sqrt(num-flag))),')','arcsec')
print('d_El:','{:.3g}'.format(d_El_mean),'+/-','{:.3g}'.format(d_El_stdev),'(','{:.3g}'.format(d_El_stdev/(np.sqrt(num-flag))),')','arcsec')
print('d_Az - model:','{:.3g}'.format(d_Az_mean_res),'+/-','{:.3g}'.format(d_Az_stdev_res),'(','{:.3g}'.format(d_Az_stdev_res/(np.sqrt(num-flag))),')','arcsec')
print('d_El - model:','{:.3g}'.format(d_El_mean_res),'+/-','{:.3g}'.format(d_El_stdev_res),'(','{:.3g}'.format(d_El_stdev_res/(np.sqrt(num-flag))),')','arcsec')

