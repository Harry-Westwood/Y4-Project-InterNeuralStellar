# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 15:46:04 2020

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def predict(model, inputs):
    inputs[2]=10**inputs[2]
    outputs = model.predict(np.log10(inputs).T).T
    [rad, Teff, delnu] = 10**outputs
    Teff = Teff*5000
    L = rad**2*(Teff/5776.02970722)**4
    return [L, Teff, delnu]

def clusterPlot(df, ax, colour, label):
    log_cal_lum_err = df['cal_lum_err']/(df['cal_lum']*np.log(10))
    log_Teff_err = df['Teff_err']/(df['Teff']*np.log(10))
    ax.errorbar(np.log10(df['Teff']), np.log10(df['cal_lum']), xerr=log_Teff_err, yerr=log_cal_lum_err, fmt='.', zorder=2, c='black')
    ax.scatter(np.log10(df['Teff']), np.log10(df['cal_lum']), s=15, zorder=3, c=colour, label=label)

def plotClusterLocation(train_df, cluster_df, label):
    fig, ax=plt.subplots(1,1, figsize=(10,10))
    ax.scatter(np.log10(train_df['effective_T']),np.log10(train_df['luminosity']),s=5,zorder=1,c='lightgrey',alpha=0.2,label='training data')
    clusterPlot(cluster_df, ax, 'blue', label)
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_xlabel(r'$\log10 T_{eff}$')
    ax.set_ylabel(r'$\log10(L/L_{\odot})$')
    ax.legend()
    plt.show()
    
def plotSample(trace, train_df, cluster_df, title, zoom_in=False):
    FL = np.mean(trace['true_L'],axis=0)
    FTeff = np.mean(trace['true_Teff'],axis=0)
    log_FL_err = np.std(trace['true_L'],axis=0)/(FL*np.log(10))
    log_FTeff_err = np.std(trace['true_Teff'],axis=0)/(FTeff*np.log(10))
    
    fig, ax=plt.subplots(1,1,figsize=[10,10])
    clusterPlot(cluster_df, ax, 'blue', 'cluster data')
    ax.errorbar(np.log10(FTeff), np.log10(FL), xerr=log_FTeff_err, yerr=log_FL_err, fmt='.', zorder=5, c='red', label='HBM guesses')
    ax.scatter(np.log10(trace['true_Teff']), np.log10(trace['true_L']), s=10, zorder=1, alpha=0.7, color=[0.5,0.5,0.5], label='sample regions')
    lims = [ax.get_xlim(), ax.get_ylim()]
    ax.scatter(np.log10(train_df['effective_T']),np.log10(train_df['luminosity']),s=5,zorder=0,c='lightgrey',alpha=0.2,label='training data')
    if zoom_in==True:
        ax.set_xlim(lims[0][::-1])
        ax.set_ylim(lims[1])
    else:
        ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_xlabel(r'$\log10 T_{eff}$')
    ax.set_ylabel(r'$\log10(L/L_{\odot})$')
    ax.legend()
    ax.set_title(title)
    plt.show()
    
def plotIsochrone(model, ax, max_mass, age, feh, Y, MLT, N=1000):
    Tage = np.ones(N)*age
    Tmass = np.linspace(0.8,max_mass,N)
    Tfeh = np.ones(N)*feh
    TY = np.ones(N)*Y
    TMLT = np.ones(N)*MLT
    [TL, TTeff, Tdelnu] = predict(model, [Tmass, Tage, Tfeh, TY, TMLT])
    value = 10*np.log10(TTeff)-np.log10(TL)-35.5
    for i,v in enumerate(value):
        if v<0.3:
            TTeff = TTeff[:i]
            TL = TL[:i]
            break
    ax.plot(np.log10(TTeff),np.log10(TL),'red',zorder=1, label='Best fit isochrone')

def predictCut(model, inputs):
    [TL, TTeff, Tdelnu] = predict(model, inputs)
    value = 10*np.log10(TTeff)-np.log10(TL)-35.5
    for i,v in enumerate(value):
        if v<0.65:
            TTeff = TTeff[:i]
            TL = TL[:i]
            break
    return TL, TTeff
    
def plotFill(model, ax, age, feh, Y, MLT, colour, zorder, N=1000, label=None):
    Tmass = np.linspace(0.8,1.7,N)
    A_list = []
    B_list = []
    for param in [age, feh, Y]:
        if type(param)==list:
            A = np.ones(N)*param[0]
            B = np.ones(N)*param[1]
        else:
            A = np.ones(N)*param
            B = A
        A_list.append(A)
        B_list.append(B)
    TMLT = np.ones(N)*MLT
    TLA, TTeffA = predictCut(model, [Tmass, *A_list, TMLT])
    TLB, TTeffB = predictCut(model, [Tmass, *B_list, TMLT])
    TL = np.concatenate([TLA, TLB[::-1]])
    TTeff = np.concatenate([TTeffA, TTeffB[::-1]])
    if label is not None:
        ax.fill(np.log10(TTeff),np.log10(TL),color=colour,zorder=zorder, label=label)
    else:
        ax.fill(np.log10(TTeff),np.log10(TL),color=colour,zorder=zorder)

def sigmaPlot(model, ax, sigma, MLT, colour, zorder, splits, label):
    for iso_age in np.linspace(sigma[0][0],sigma[0][1],splits):
        for iso_feh in np.linspace(sigma[1][0],sigma[1][1],splits):
            plotFill(model, ax, iso_age, iso_feh, [sigma[2][0],sigma[2][1]], MLT, colour, zorder, label=label)
            label = None
        for iso_Y in np.linspace(sigma[2][0],sigma[2][1],splits):
            plotFill(model, ax, iso_age, [sigma[1][0],sigma[1][1]], iso_Y, MLT, colour, zorder)
    for iso_feh in np.linspace(sigma[1][0],sigma[1][1],splits):
        for iso_Y in np.linspace(sigma[2][0],sigma[2][1],splits):
            plotFill(model, ax, [sigma[0][0],sigma[0][1]], iso_feh, iso_Y, MLT, colour, zorder)

            
def fittedIso(trace, model, cluster_df, title, max_mass=1.7):
    ma = np.percentile(trace['mean_age'],50)
    sa = np.percentile(trace['spread_age'],50)
    mf = np.percentile(trace['mean_feh'],50)
    sf = np.percentile(trace['spread_feh'],50)
    my = np.percentile(trace['mean_Y'],50)
    sy = np.percentile(trace['spread_Y'],50)
    mmlt = np.percentile(trace['mean_MLT'],50)

    fig, ax=plt.subplots(1,1,figsize=[10,10])
    clusterPlot(cluster_df, ax, 'blue', 'cluster data')
    plotIsochrone(model, ax, max_mass, ma, mf, my, mmlt)
    lims = [ax.get_xlim(),ax.get_ylim()]
    one_sigma = [[ma-sa, ma+sa],[mf-sf, mf+sf],[my-sy, my+sy]]
    sigmaPlot(model, ax, one_sigma, mmlt, ([0.5,0.5,0.5]), 0, 5, r'1$\sigma$ spread')
    two_sigma = [[ma-2*sa, ma+2*sa],[mf-2*sf, mf+2*sf],[my-2*sy, my+2*sy]]
    sigmaPlot(model, ax, two_sigma, mmlt, ([0.8,0.8,0.8]), -1, 9, r'2$\sigma$ spread')
    ax.set_xlim(lims[0][::-1])
    ax.set_ylim(lims[1])
    ax.set_xlabel(r'$\log10 T_{eff}$')
    ax.set_ylabel(r'$\log10(L/L_{\odot})$')
    ax.legend()
    ax.set_title(title)
    plt.show()

def saveResults(trace, path):
    result_dict={}
    for parameter in ['mean_age','spread_age','mean_feh','spread_feh','mean_Y','spread_Y','mean_MLT']:
        median = np.percentile(trace[parameter],50)
        low_err = median-np.percentile(trace[parameter],15.9)
        high_err = np.percentile(trace[parameter],84.1)-median
        result_dict[parameter+'_est'] = median
        result_dict[parameter+'_low_err'] = low_err
        result_dict[parameter+'_high_err'] = high_err
    result_df = pd.DataFrame(result_dict, index=[1])
    result_df.to_csv(path,index=False)