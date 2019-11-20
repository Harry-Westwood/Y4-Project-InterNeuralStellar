# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:11:41 2019

@author: User
"""

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
#from tensorflow.keras import backend
from matplotlib import rc
rc("font", family="serif", size=14)
from datetime import datetime
import pickle

class neuralStellar:
    def __init__(self, filename):
        self.file = filename
        self.data = None
        self.indices = None
        self.evo_tracks = None
        self.ranged_tracks = None
        
    def buildIndex(self):
        headers=open(self.file, 'r').read().split('\n')[0].split(',')
        dictionary={}
        for i,h in enumerate(headers):
            dictionary[h]=i
        self.indices = dictionary
    
    def popIndex(self, names):
        self.indices['step'] = self.indices.pop(names[0])
        self.indices['mass'] = self.indices.pop(names[1])
        self.indices['age'] = self.indices.pop(names[2])
        self.indices['Teff'] = self.indices.pop(names[3])
        self.indices['L'] = self.indices.pop(names[4])

    def initialData(self, age_range=None):
        #indices=[step, mass, age, Teff, L]
        self.data=np.genfromtxt(self.file, delimiter=',', skip_header=1)
        print('Data headers = ')
        print(open(self.file, 'r').read().split('\n')[0].split(','))
    
        evo_tracks=[]
        last_number=-1
        last_index=0
        for i,entry in enumerate(self.data):
            if entry[self.indices['step']]<last_number:
                evo_tracks.append(self.data[last_index:i])
                last_index=i
            if i==len(self.data)-1:
                evo_tracks.append(self.data[last_index:])
            last_number=entry[self.indices['step']]
        self.evo_tracks = np.array(evo_tracks)
        
        if age_range != None:
            range_tracks=[]
            for track in self.evo_tracks:
                range_track=[]
                for entry in track:
                    if entry[self.indices['age']]>=age_range[0] and entry[self.indices['age']]<=age_range[1]:
                        range_track.append(entry)
                if len(range_track)>0:
                    range_tracks.append(np.array(range_track))
            self.ranged_tracks=np.array(range_tracks)

    def datatoplot(self, track_choice, track_no=None, track_index=None):
        if track_choice == 'evo':
            tracks = self.evo_tracks
        elif track_choice == 'ranged':
            tracks = self.ranged_tracks
        else: raise NameError('Wrong track name!')
        if track_no != None:
            if len(tracks) > track_no:
                if type(track_index)==type(None):
                    track_index=np.random.choice(np.arange(len(tracks)),track_no)
                tracks=[tracks[i] for i in track_index]
        plot_tracks=[tracks[0][:,self.indices['Teff']],tracks[0][:,self.indices['L']]]
        plot_m=[tracks[0][:,self.indices['mass']]]
        for track in tracks:
            plot_tracks[0]=np.append(plot_tracks[0],track[:,self.indices['Teff']])
            plot_tracks[1]=np.append(plot_tracks[1],track[:,self.indices['L']])
            plot_m=np.append(plot_m,track[:,self.indices['mass']])
        return plot_tracks, plot_m
    
    def plotHR1(self, track_choice, track_no=None):
        plot_tracks,plot_m=self.datatoplot(track_choice, track_no=track_no)
        fig, ax=plt.subplots(1,1,figsize=[10,10])
        #print(np.ones(10)*np.array([1,2]))
        s1=ax.scatter(np.log(plot_tracks[0]),np.log(plot_tracks[1]),s=5,c=plot_m,cmap='viridis')
        ax.set_xlim(ax.get_xlim()[::-1])
        ax.set_ylabel(r'$log(L/L_{\odot})$')
        ax.set_xlabel(r'$log T_{eff}$')
        fig.colorbar(s1)
        plt.show()
        
    def fetchData(self, track_choice, parameters):
        if track_choice == 'evo':
            return fetchData(self.evo_tracks, parameters, self.indices)
        elif track_choice == 'ranged':
            return fetchData(self.evo_tracks, parameters, self.indices)
        else: raise NameError('Wrong track name!')
    
def fetchData(data, parameters, indices):
    fIndex=[]
    for p in parameters:
        fIndex.append(indices[p])
    return_array=[]
    if np.shape(data[0])==(len(indices),):
        for i,ind in enumerate(fIndex):
            return_array.append(np.log10(data[:,ind]).astype('float32'))
    else:
        for i,ind in enumerate(fIndex):
            dummy_array=[]
            for d in data:
                dummy_array=np.append(dummy_array,np.log10(d[:,ind]).astype('float32'))
            return_array.append(dummy_array)
    return return_array

class NNModel:
    def __init__(self, track_choice):
        self.model = None
        self.history = None
        if track_choice=='evo' or track_choice=='ranged':
            self.track_choice = track_choice
        else: raise NameError('Wrong track name!')
    
    def buildModel(self, New_model,inout_shape=[0,0],no_layers=0,no_nodes=0,reg=None, call_name=None):
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
        self.model = model

    def compileModel(self, lr, loss, metrics=None, beta_1=0.9, beta_2=0.999):
        optimizer=keras.optimizers.Nadam(lr=lr, beta_1=beta_1, beta_2=beta_2)
        if metrics!=None:
            self.model.compile(optimizer=optimizer,loss=loss, metrics=metrics)
        else: self.model.compile(optimizer=optimizer,loss=loss)

    def fitModel(self, inputs, outputs, epoch_no, batch_size, save_name, keep_log=True):
        if keep_log==True:
            logdir = "/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
            callback=[tensorboard_callback]
        else: callback=[]
    
        start_time=datetime.now()
        self.history=self.model.fit(np.array(inputs).T,np.array(outputs).T,
                          epochs=epoch_no,
                          batch_size=batch_size,
                          validation_split=0.10,
                          verbose=0,
                          callbacks=callback)
        print('training done! now='+str(datetime.now())+' | Time lapsed='+str(datetime.now()-start_time))
        self.model.save(save_name)

    def loadHist(self, filename):
        self.history = filename
    
    def evalData(self, grid, nth_track):
        if self.track_choice == 'evo':
            tracks = grid.evo_tracks
        elif self.track_choice == 'ranged':
            tracks = grid.ranged_tracks
        eva_in=fetchData(tracks[nth_track], ['mass','age'],grid.indices)
        eva_out=fetchData(tracks[nth_track],['L','Teff'],grid.indices)
        self.model.evaluate(np.array(eva_in).T,np.array(eva_out).T,verbose=2)
    
    def plotHist(self, trial_no=None, savefile=None, plot_MSE=True):
        if type(self.history)==str:
            hist=pickle.load(open( self.history, "rb" ))
            epoch=hist['epoch']
        else:
            epoch = self.history.epoch
            hist=self.history.history
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
        plt.plot()
        if savefile != None:
            fig.savefig(savefile+'/history'+str(trial_no)+'.png')
    
    def plotHR(self, grid, track_no, trial_no=None, savefile=None):
        if self.track_choice == 'evo':
            tracks = grid.evo_tracks
        elif self.track_choice == 'ranged':
            tracks = grid.ranged_tracks
        if track_no != None:
            if len(tracks) > track_no:
                track_index=np.random.choice(np.arange(len(tracks)),track_no)
                tracks=[tracks[i] for i in track_index]
            else: track_index=None
        else: track_index=None
        if len(tracks)>200:
            raise ValueError('Too many tracks, are you sure you want to plot '+str(len(tracks))+' tracks??')
        x_in=fetchData(tracks,['mass','age'],grid.indices)
        NN_tracks=self.model.predict(np.array(x_in).T,verbose=2).T
        NN_m=x_in[0]
        plot_tracks,plot_m=grid.datatoplot(self.track_choice, track_no=track_no, track_index=track_index)
        [Teffm, Lm, Mm, Teffg, Lg, Mg] = [NN_tracks[1], NN_tracks[0], NN_m, np.log10(plot_tracks[0]), np.log10(plot_tracks[1]), plot_m]
        
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
        plt.plot()
        if savefile != None:
            fig.savefig(savefile+'/HR'+str(trial_no)+'.png')
        
    def lastLoss(self, key):
        if type(self.history)==str:
            hist=pickle.load(open( self.history, "rb" ))
        else:
            hist=self.history.history
        return hist[key][-1]