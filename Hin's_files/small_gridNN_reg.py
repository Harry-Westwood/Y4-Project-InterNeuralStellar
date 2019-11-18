# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:01:05 2019

@author: User
"""

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
#from tensorflow.keras import backend
from matplotlib import rc
rc("font", family="serif", size=14)
from datetime import datetime
import os
import pickle

def datatoplot(tracks):
    plot_tracks=[tracks[0][:,7],tracks[0][:,8]]
    plot_m=[tracks[0][:,2]]
    for track in tracks:
        plot_tracks[0]=np.append(plot_tracks[0],track[:,7])
        plot_tracks[1]=np.append(plot_tracks[1],track[:,8])
        plot_m=np.append(plot_m,track[:,2])
    return plot_tracks, plot_m

def fetchData(data,parameters):
    indexes=[]
    for p in parameters:
        if p == 'Mass' or p == 'mass' or p == 'M':
            indexes.append(2)
        elif p == 'Age' or p == 'age':
            indexes.append(3)
        elif p == 'Metalicity' or p == 'metalicity' or p == 'Z':
            indexes.append(4)
        elif p == 'MLT':
            indexes.append(5)
        elif p == 'Teff':
            indexes.append(7)
        elif p == 'Luminosity' or p == 'luminosity' or p == 'L':
            indexes.append(8)
    return_array=[]
    if np.shape(data[0])==(27,):
        for i,ind in enumerate(indexes):
            return_array.append(np.log10(data[:,ind]).astype('float32'))
    else:
        for i,ind in enumerate(indexes):
            dummy_array=[]
            for d in data:
                dummy_array=np.append(dummy_array,np.log10(d[:,ind]).astype('float32'))
            return_array.append(dummy_array)
    return return_array

def buildModel(New_model,inout_shape=[0,0],no_layers=0,no_nodes=0,reg=None, call_name=None):
    if reg!=None:
        if reg[0]=='l1':
            regu=keras.regularizers.l1(reg[1])
        elif reg[0]=='l2':
            regu=keras.regularizers.l2(reg[1])
    if New_model:
        inputs=keras.Input(shape=(inout_shape[0],))
        if reg==None:
            xx=keras.layers.Dense(no_nodes,activation='relu')(inputs)
        else: xx=keras.layers.Dense(no_nodes,activation='relu',kernel_regularizer=regu)(inputs)
        for i in range(no_layers-1):
            if reg==None:
                xx=keras.layers.Dense(no_nodes,activation='relu')(xx)
            else: xx=keras.layers.Dense(no_nodes,activation='relu',kernel_regularizer=regu)(xx)
        outputs=keras.layers.Dense(inout_shape[1],activation='linear')(xx)
        model = keras.Model(inputs=inputs, outputs=outputs)
    else:
        model = keras.models.load_model(call_name)
    model.summary()
    return model

def compileModel(model, lr, loss, metrics=None, beta_1=0.9, beta_2=0.999):
    optimizer=keras.optimizers.Nadam(lr=lr, beta_1=beta_1, beta_2=beta_2)
    if metrics!=None:
        model.compile(optimizer=optimizer,loss=loss, metrics=metrics)
    else: model.compile(optimizer=optimizer,loss=loss)

def fitModel(model, inputs, outputs, epoch_no, batch_size, save_name, keep_log=False, folder=None):
    if keep_log==True:
        logdir = folder+"/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
        callback=[tensorboard_callback]
    else: callback=[]

    start_time=datetime.now()
    history=model.fit(np.array(inputs).T,np.array(outputs).T,
                      epochs=epoch_no,
                      batch_size=batch_size,
                      validation_split=0.10,
                      verbose=0,
                      callbacks=callback)
    print('training done! now='+str(datetime.now())+' | Time lapsed='+str(datetime.now()-start_time))
    model.save(save_name)
    return history

def plotHist(history, trial_no, folder, plot_MSE=True):
    epoch = history.epoch
    hist=history.history
    MAE,valMAE=hist['MAE'],hist['val_MAE']
    fig, ax = plt.subplots(1, 1)
    ax.plot(epoch,MAE,'b',label='MAE')
    ax.plot(epoch,valMAE,'r',label='valMAE')
    if plot_MSE==True:
        MSE,valMSE = hist['MSE'],hist['val_MSE']
        ax.plot(epoch,MSE,'b:',label='MSE')
        ax.plot(epoch,valMSE,'r:',label='valMSE')
    ax.set_yscale('log')
    ax.set_xlabel('epoch')
    ax.set_ylabel('metric')
    ax.legend()
    fig.savefig(folder+'/history'+str(trial_no)+'.png')

def plotHR(Teffm, Lm, Mm, Teffg, Lg, Mg, trial_no, folder):
    fig, ax=plt.subplots(1,2,figsize=[18,10])
    s1=ax[0].scatter(Teffm,Lm,s=5,c=Mm, cmap='viridis')
    ax[0].set_xlim(ax[0].get_xlim()[::-1])
    ax[0].set_ylabel(r'$log(L/L_{\odot})$')
    ax[0].set_xlabel(r'$log T_{eff}$')
    s2=ax[1].scatter(Teffg,Lg,s=5,c=Mg, cmap='viridis')
    ax[1].set_xlim(ax[1].get_xlim()[::-1])
    ax[1].set_ylabel(r'$log(L/L_{\odot})$')
    ax[1].set_xlabel(r'$log T_{eff}$')
    fig.colorbar(s2)
    fig.savefig(folder+'/HR'+str(trial_no)+'.png')

data=np.genfromtxt('grid_0_0.csv', delimiter=',', skip_header=1)
#print(open('grid_0_0.csv', 'r').read().split('\n')[0].split(','))

evo_tracks=[]
last_number=-1
last_index=0
for i,entry in enumerate(data):
    if entry[0]<last_number:
        evo_tracks.append(data[last_index:i])
        last_index=i
    if i==len(data)-1:
        evo_tracks.append(data[last_index:])
    last_number=entry[0]

#main code bit
time_now=datetime.now().strftime("%Y-%m-%d_%H%M%S")
folder_name='Hin_gridNN_outputs_'+time_now
os.mkdir(folder_name)
x_in=fetchData(evo_tracks,['mass','age'])
y_out=fetchData(evo_tracks,['L','Teff'])
#m1=buildModel(True,inout_shape=[len(x_in),len(y_out)],no_layers=4,no_nodes=32, reg=['l2',0.01])
m1=buildModel(False, call_name='small_grid_model_reg.h5')
compileModel(m1, 0.0001,'MAE',metrics=['MAE','MSE'])
hist=fitModel(m1, x_in, y_out, 200000, len(x_in[0]),folder_name+'/small_grid_model_temp.h5', keep_log=False)

saving_dict=hist.history
saving_dict.update({'epoch':hist.epoch})
with open(folder_name+'/trainHistoryDict', 'wb') as file_pi:
    pickle.dump(saving_dict, file_pi)

plotHist(hist, 1, folder_name, plot_MSE=False)
NN_tracks=m1.predict(np.array(x_in).T,verbose=2).T
NN_m=x_in[0]
plot_tracks,plot_m=datatoplot(evo_tracks)
plotHR(NN_tracks[1], NN_tracks[0], NN_m, np.log10(plot_tracks[0]), np.log10(plot_tracks[1]), plot_m, 1, folder_name)