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
import pymc3 as pm
import theano.tensor as T

class stellarGrid:
    """
    Class object that stores and process relevent information about a stellar grid.
    """
    proper_index = ['step', 'mass', 'age', 'feh', 'Y', 'MLT', 'Teff', 'L', 'delnu']
    def __init__(self, filename):
        """
        Parameters: 
        ----------
        filename: str
            path and filename of the grid, in csv form.
            
        Other class attributes:
        ----------
        data: numpy array
            stores the entire data block read from the grid
        indices: dictionary
            stores the name-to-column-index of the grid data, used to index the correct
            stellar parameters in the code
        evo_tracks: numpy array/list
            stores the stellar evolution tracks (that have been properly seperated from 
            the grid data) that are of the full age range
        ranged_tracks: numpy array/list
            stores the version of evo_tracks but under an age constraint
        """
        self.file = filename
        self.data = None
        self.indices = None
        self.evo_tracks = None
        self.ranged_tracks = None
        
    def buildIndex(self):
        """Reads out the headers on the grid csv and saves it as self.indices dictonary"""
        headers=open(self.file, 'r').read().split('\n')[0].split(',')
        dictionary={}
        for i,h in enumerate(headers):
            dictionary[h]=i
        self.indices = dictionary
    
    def popIndex(self, names, proper=None):
        """
        Interchanges out parts of the index dictionary into names that the code uses.
        This is seperated from buildIndex method so users can print the index dictionary
        before inputting the correct key names into this method.
        
        Parameters:
        ----------
        names: str
            list of the names of keys in the original index dictionary to be replaced
            with, in the order [step, mass, age, feh, Y, MLT, Teff, luminosity, delnu]
        """
        if proper==None:
            if len(names) != len(self.proper_index):
                raise ValueError('Expecting '+str(len(self.proper_index))+' keys but '
                                 +str(len(names))+' given.')
            for i,key in enumerate(self.proper_index):
                if names[i] != None:
                    self.indices[key] = self.indices.pop(names[i])
        else:
            if len(names) != len(proper):
                raise ValueError('Expecting '+str(len(proper))+' keys but '
                                 +str(len(names))+' given.')
            for i,key in enumerate(proper):
                self.indices[key] = self.indices.pop(names[i])

    def initialData(self, age_range=None):
        """
        Identifies and cuts out the grid data into the individual stellar evolution
        tracks (to be saved into self.evo_tracks) and, if age_range is provided,
        does the age pruning as well for the tracks (to be saved into self.ranged_tracks)
        
        Parameters:
        ----------
        age_range: list, optional
            age_range[0] = lower boundary of age pruning
            age_range[1] = upper boundary of age pruning
        """
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
        """
        Returns the needed data in the correct format for platting HR diagrams from
        the grid data.
        
        Parameters:
        ----------
        track_choice: str
            'evo' = to plot self.evo_tracks, 'ranged' = to plot self.ranged_tracks
            Raises NameError 'Wrong track name!' if the input does not match either 
            of them.
        track_no: int, optional
            number of tracks to be plotted. If None is given, will plot all tracks 
            in either evo or ranged. Raises ValueError 'Too many tracks, are you 
            sure you want to plot x tracks?' if number of tracks > 200.
        track_index: numpy array/list, optional
            index of the explicit tracks to be plotted. A parameter only used by 
            the NNmodel.plotHR method, where the tracks plotted there for the NN
            predicted results has to match the ones from the grid itself.
        
        Returns
        -------
        plot_tracks: 2D array
            [0] = Teff list, [1] = luminosity list (both are not log-ed)
        plot_m: 1D array
            mass list
        """
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
        if len(tracks)>200:
            raise ValueError('Too many tracks, are you sure you want to plot '+str(len(tracks))+' tracks??')
        plot_tracks=[tracks[0][:,self.indices['Teff']],tracks[0][:,self.indices['L']]]
        plot_m=[tracks[0][:,self.indices['mass']]]
        for track in tracks:
            plot_tracks[0]=np.append(plot_tracks[0],track[:,self.indices['Teff']])
            plot_tracks[1]=np.append(plot_tracks[1],track[:,self.indices['L']])
            plot_m=np.append(plot_m,track[:,self.indices['mass']])
        return plot_tracks, plot_m
    
    def plotHR1(self, track_choice, track_no=None):
        """
        Plots a HR diagram of the grid data given. Calls datatoplot method.
        
        Parameters:
        ----------
        track_choice: str
            'evo' or 'ranged', passed onto datatoplot
        track_no: int, optional
            number of tracks to be plotted, passed onto datatoplot. Plots all tracks
            avaliable if given None.
        """
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
    """
    Gathers up and returns an array of specific stellar parameters from a grid data.
    Return array is of size (p, n) where p is the number of parameters requested 
    to be extracted from the data, and n is the total number of grid points
    involved. All data are log10ed upon returning except feh, Y and MLT.
    
    Parameters:
    ----------
    data: 2D data array or 3D stellar tracks
    parameters: list
        list of names of the stellar parameters the user wants to extract from data
    indices: dictionary
        the index dictionary of the data given
    """
    fIndex=[]
    for p in parameters:
        fIndex.append(indices[p])
    return_array=[]
    if np.shape(data[0])==(len(indices),):
        for p in parameters:
            if p == 'feh' or p == 'Y' or p == 'MLT':
                return_array.append(data[:,indices[p]].astype('float32'))
            else:
                return_array.append(np.log10(data[:,indices[p]]).astype('float32'))
    else:
        for p in parameters:
            if p == 'feh' or p == 'Y' or p == 'MLT':
                dummy_array=[]
                for d in data:
                    dummy_array=np.append(dummy_array,d[:,indices[p]].astype('float32'))
            else:
                dummy_array=[]
                for d in data:
                    dummy_array=np.append(dummy_array,np.log10(d[:,indices[p]]).astype('float32'))
            return_array.append(dummy_array)
    return return_array

def shuffleInputs(x_in, y_out):
    """
    Unisonly shuffles the inputs and outputs to a NN.
    
    Parameters:
    ----------
    x_in: 1D or 2D numpy array (different stars in different columns)
    y_out: 1D or 2D numpy array (different stars in different columns)
    
    Returns:
    ----------
    suffled arrays in the same shape and size of x_in and y_out
    """
    if np.array(x_in).ndim == 2:
        a = x_in[0]
        #b = [*x_in[1:], *y_out]
        p = np.random.permutation(len(a))
        in_return = [a[p]]
        for i in range(len(x_in)-1):
            in_return.append(x_in[i+1][p])
        if np.array(y_out).ndim == 2:
            out_return=[yi[p] for yi in y_out]
        else: out_return = y_out[p]
        return in_return, out_return
        
    else:
        a = x_in
        b = y_out
        p = np.random.permutation(len(a))
        if np.array(b).ndim == 2:
            return a[p], [bi[p] for bi in b]
        else: return a[p], b[p]

class NNmodel:
    """
    Class object that stores a keras model trained/to be trained on stellar
    grids, and helps plotting its training results.
    """
    def __init__(self, track_choice, input_index, output_index):
        """
        Parameters: 
        ----------
        track_choice: str
            'evo' = to plot grid.evo_tracks, 'ranged' = to plot grid.ranged_tracks
            Raises NameError 'Wrong track name!' if the input does not match either 
            of them.
        input_index: list of strings
            The list of keys that corresponds to the inputs to the NN
        output_index: list of strings
            The list of keys that corresponds to the outputs to the NN
            
        Other class attributes:
        ----------
        model: keras model object
            Given by self.buildModel method
        history: keras history object or str
            If it is a keras history object, then it is directly written from a
            self.model.fit function. If it is a string, then it is the file name
            of a saved history dictionary given by self.loadHist
        """
        self.model = None
        self.history = None
        self.weights = None
        self.no_hidden_layers = None
        if track_choice=='evo' or track_choice=='ranged':
            self.track_choice = track_choice
        else: raise NameError('Wrong track name!')
        self.input_index = input_index
        self.output_index = output_index
    
    def buildModel(self, no_layers, no_nodes, reg=None):
        """
        Builds a new NN to self.model, Prints the summary.
        
        Parameters: 
        ----------
        inout_shape: 2-long list
            [0] = number of inputs, [1] = number of outputs, of NN
        no_layers: int
            number of HIDDEN layers of the NN, all layers are densely connected
        no_nodes: int
            number of neurons in each hidden layer
        reg: 2-long list, optional
            regularization is enforced if reg is not None.
            [0] = str, name of regularizer, l1 or l2, raises Name Error 'Wrong 
            regularizer name!' if input does not match either of the two.
            [1] = regularizer value
        """
        if reg!=None:
            if reg[0]=='l1':
                regu=keras.regularizers.l1(reg[1])
            elif reg[0]=='l2':
                regu=keras.regularizers.l2(reg[1])
            else: raise NameError('Wrong regularizer name!')
        inputs=keras.Input(shape=(len(self.input_index),))
        if reg==None:
            xx=keras.layers.Dense(no_nodes,activation='relu')(inputs)
        else: xx=keras.layers.Dense(no_nodes,activation='relu',kernel_regularizer=regu)(inputs)
        for i in range(no_layers-1):
            if reg==None:
                xx=keras.layers.Dense(no_nodes,activation='relu')(xx)
            else: xx=keras.layers.Dense(no_nodes,activation='relu',kernel_regularizer=regu)(xx)
        outputs=keras.layers.Dense(len(self.output_index),activation='linear')(xx)
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.summary()

    def loadModel(self, filename):
        """
        Loads in a pre-trained/pre-built NN to self.model, prints the summary.
        """
        self.model = keras.models.load_model(filename)
        self.model.summary()
        
    def compileModel(self, lr, loss, metrics=None, beta_1=0.9, beta_2=0.999):
        """
        Compiles self.model with Nadam optimizer.
        
        Parameters: 
        ----------
        lr: float
            learning rate
        loss: str
            metric to measure the loss
        metrics: list, optional
            list of metrics to be calculated
        beta_1 and beta_2: float, optional, as defined in keras nadam optimizer
        """
        optimizer=keras.optimizers.Nadam(lr=lr, beta_1=beta_1, beta_2=beta_2)
        if metrics!=None:
            self.model.compile(optimizer=optimizer,loss=loss, metrics=metrics)
        else: self.model.compile(optimizer=optimizer,loss=loss)

    def fitModel(self, inputs, outputs, epoch_no, batch_size, save_name, vsplit=0.1, keep_log=True):
        """
        Trains self.model, saves history to self.history.
        
        Parameters: 
        ----------
        inputs: 2D array
            NN inputs
        outputs: 2D array
            NN outputs
        epoch_no: int
            number of epochs to train the NN for
        batch_size: int
            batch_size of NN training, for the gird, ideally set this to the length
            of the training data
        save_name: str
            file name to save the trained NN to
        vsplit: float, optional
            validation_split during training, default to 0.1
        keep_log: bool, optional
            if True, saves tensorboard logs under the folder /logs/datetime.now()
            defaults to True
        """
        if keep_log==True:
            logdir = "/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
            callback=[tensorboard_callback]
        else: callback=[]
    
        start_time=datetime.now()
        self.history=self.model.fit(np.array(inputs).T,np.array(outputs).T,
                          epochs=epoch_no,
                          batch_size=batch_size,
                          validation_split=vsplit,
                          verbose=0,
                          callbacks=callback)
        print('training done! now='+str(datetime.now())+' | Time lapsed='+str(datetime.now()-start_time))
        self.model.save(save_name)

    def saveHist(self, filename):
        """Saves the history dictionary into a txt file with pickle"""
        saving_dict=self.history.history
        saving_dict.update({'epoch':self.history.epoch})
        with open(filename, 'wb') as file_pi:
            pickle.dump(saving_dict, file_pi)
    
    def loadHist(self, filename):
        """Passes the history file name to self.history, does basically nothing"""
        self.history = filename
    
    def evalData(self, grid, nth_track):
        """
        Evaluates the NN on a given grid data, prints the result
        
        Parameters:
        ----------
        grid: stellarGrid object
            grid object with track data stored
        nth_track: int
            the nth track to be used for evaluation, in the grid data
        """
        if self.track_choice == 'evo':
            tracks = grid.evo_tracks
        elif self.track_choice == 'ranged':
            tracks = grid.ranged_tracks
        eva_in=fetchData(tracks[nth_track], self.input_index, grid.indices)
        eva_out=fetchData(tracks[nth_track], self.output_index, grid.indices)
        print('evaluation results:')
        self.model.evaluate(np.array(eva_in).T,np.array(eva_out).T,verbose=2)
    
    def plotHist(self, plot_MSE=True, epochs=None, savefile=None, trial_no=None):
        """
        Plots both training and validation loss vs epochs form training history. Can save
        plot.
        
        Parameters:
        ----------
        plot_MSE: bool, optional
            if True, plots MSE in the same plot as MAE,
            if False, only plots MAE
        epochs: list, optional
            if None, all epochs is plotted
            if given a list of [lower limit, upper limit] of epoch numbers, it will
            plot the history in epochs lower limit to upper limit-1.
        savefile: str, optional
            path and filename for saving the plot. Plot is only saved if not None
        trial_no: int, optional
            only used if savefile is not None. The trial number to be tagged after
            the diagram savename, matches the excel notes.
        """
        if type(self.history)==str:
            hist=pickle.load(open( self.history, "rb" ))
            epoch=hist['epoch']
        else:
            epoch = self.history.epoch
            hist=self.history.history
        MAE,valMAE=hist['MAE'],hist['val_MAE']
        if type(epochs)!=type(None):
            epoch = epoch[epochs[0]:epochs[1]]
            MAE = MAE[epochs[0]:epochs[1]]
            valMAE = valMAE[epochs[0]:epochs[1]]
        fig, ax = plt.subplots(1, 1)
        ax.plot(epoch,MAE,'b',label='MAE')
        ax.plot(epoch,valMAE,'r',label='valMAE')
        if plot_MSE==True:
            MSE,valMSE = hist['MSE'],hist['val_MSE']
            if type(epochs)!=type(None):
                MSE = MSE[epochs[0]:epochs[1]]
                valMSE = valMSE[epochs[0]:epochs[1]]
            ax.plot(epoch,MSE,'b:',label='MSE')
            ax.plot(epoch,valMSE,'r:',label='valMSE')
        ax.set_yscale('log')
        ax.set_xlabel('epoch')
        ax.set_ylabel('metric')
        ax.legend()
        plt.show()
        if savefile != None:
            fig.savefig(savefile+'/history'+str(trial_no)+'.png')
            print('history plot saved as "'+savefile+'/history'+str(trial_no)+'.png"')
    
    def prepPlot(self, grid, track_no):
        """
        Fetches out the correct number of tracks from a grid as inputs to NN prediction
        for the comparison plots.
        
        Parameters:
        ----------
        grid: stellarGrid object
            grid object with track data stored
        track_no: int, optional
            number of tracks to be plotted. If None is given, will plot all tracks.
            Raises ValueError 'Too many tracks, are you sure you want to plot x
            tracks?' if number of tracks > 200.
            
        Returns:
        ----------
        x_in: numpy array
            the input array that goes into a NN prediction, in order specified in
            self.input_index
        track_index: 1D list/numpy array
            list of indices to be passed to stellarGrid.datatoplot method to index
            the same randomly chosen tracks as here
        """
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
        return fetchData(tracks, self.input_index, grid.indices), track_index
    
    def plotHR(self, grid, track_no, savefile=None, trial_no=None):
        """
        Plots both grid(data) and NN predicted HR diagrams. Can save
        plot.
        
        Parameters:
        ----------
        grid: stellarGrid object
            grid object with track data stored
        track_no: int, optional
            number of tracks to be plotted. Passed to self.prepPlot
        savefile: str, optional
            path and filename for saving the plot. Plot is only saved if not None
        trial_no: int, optional
            only used if savefile is not None. The trial number to be tagged after
            the diagram savename, matches the excel notes.
        """
        x_in, track_index = self.prepPlot(grid, track_no)
        NN_tracks=self.model.predict(np.array(x_in).T,verbose=2).T
        NN_m=10**x_in[self.input_index.index('mass')]
        plot_tracks,plot_m=grid.datatoplot(self.track_choice, track_no=track_no, track_index=track_index)
        [Teffm, Lm, Mm, Teffg, Lg, Mg] = [NN_tracks[1], NN_tracks[0], NN_m, np.log10(plot_tracks[0]), np.log10(plot_tracks[1]), plot_m]
        
        fig, ax=plt.subplots(1,2,figsize=[16,8])
        ax[0].scatter(Teffm,Lm,s=5,c=Mm, cmap='viridis')
        ax[0].set_xlim(ax[0].get_xlim()[::-1])
        ax[0].set_ylabel(r'$\log10(L/L_{\odot})$')
        ax[0].set_xlabel(r'$\log10 T_{eff}$')
        ax[0].set_title('NN predicted')
        s2=ax[1].scatter(Teffg,Lg,s=5,c=Mg, cmap='viridis')
        ax[1].set_xlim(ax[1].get_xlim()[::-1])
        ax[1].set_ylabel(r'$\log10(L/L_{\odot})$')
        ax[1].set_xlabel(r'$\log10 T_{eff}$')
        ax[1].set_title('Real data')
        fig.subplots_adjust(right=0.83)
        cbar_ax = fig.add_axes([0.85, ax[1].get_position().y0, 0.02, ax[1].get_position().height])
        cbar_ax.text(0.5,1.015,r'$M/M_{\odot}$',fontsize=13,horizontalalignment='center',transform=cbar_ax.transAxes)
        fig.colorbar(s2, cax=cbar_ax)
        plt.show()
        if savefile != None:
            fig.savefig(savefile+'/HR'+str(trial_no)+'.png')
            print('HR diagram saved as "'+savefile+'/HR'+str(trial_no)+'.png"')
    
    def plotIsochrone(self, grid, iso_ages, indices, isos, widths, N=5000, savefile=None, trial_no=None):
        """
        Plots both grid(data) and NN predicted isochrones of specified ages in HR
        diagrams, with the colour bar showing variation in age. Can save plot.
        
        Parameters:
        ----------
        grid: stellarGrid object
            grid object with track data stored
        iso_ages: list/array
            the isochronic ages to be plotted
        indices: list of strings, length = len(self.input_index)-1 (has age)
            the order of NN input parameters (stellar fundamentals) of which to
            constraint, in the form(spelling) of stellarGrid.proper_index.
            'age' must be at the first index.
        isos: list/array of, length = len(self.input_index)-2 (no age)
            the 'mean values', mu of the parameters other than age to be constrainted
        widths: 2D list, length = len(self.input_index)-1 (has age)
            the 'boundary width', delta to be added onto the relevent isos number, making
            the selector select data falling between mu-delta and mu+delta.
            For each (ith) input parameter, if widths[i][0]=='r', boundary width is 
            calculated in ratio mode: ie. delta = isos[i-1]*widths[i][1]
            if widths[i][0]=='a', boundary width is absolute: ie. delta = widths[i][1].
            If the iso parameter (mu) is zero and widths[i][0] is set to 'r',
            will force an arbituary width of 0.05 to this instance of the parameter.
        
        Note: indices, isos and widths must be in the same order in terms of stellar
        parameters, and age must always come first, for example, if 4 parameters go
        into my NN, ['mass', 'age', 'feh', 'MLT'], I need to apply constraint on
        age, feh and MLT, so my inputs will be:
            indices = ['age','feh','MLT']
            isos = [feh_isovalue, MLT_isovalue]
            widths = [['r',age_width_ratio],['a',feh_width_value],['a',MLT_width_value]]
        Orders of feh and MLT can be interchanged, but if it is interchanged in one of
        the inputs, the other two must be changed in accordance.
        """
        #checking for correct input lengths
        if len(indices)!=len(self.input_index)-1 or len(isos)!=len(self.input_index)-2 or len(widths)!=len(self.input_index)-1:
            raise ValueError('Two (or more) of the lengths of the inputs do not match!')
        #cropping the correct grid data with given boundaries
        fetched_data = []
        for j,iso_age in enumerate(iso_ages):
            if widths[0][0]=='r':
                if iso_age!=0:
                    age_width = iso_age*widths[0][1]
                else: age_width = 0.05
            else: age_width=widths[0][1]
            data = grid.data
            data = data[np.where(np.logical_and(
                    data.T[grid.indices['age']] >= iso_age-age_width,
                    data.T[grid.indices['age']] <= iso_age+age_width))]
            for i,param in enumerate(indices[1:]):
                if widths[i+1][0]=='r':
                    if isos[i]!=0:
                        this_width = isos[i]*widths[i+1][1]
                    else: this_width = 0.05
                else: this_width = widths[i+1][1]
                data = data[np.where(np.logical_and(
                        data.T[grid.indices[param]] >= isos[i]-this_width,
                        data.T[grid.indices[param]] <= isos[i]+this_width))]
            if j==0:
                fetched_data = data
            else: 
                fetched_data = np.concatenate((fetched_data,data))
        
        #creating new input lists for NN to predict
        masses = np.log10(np.linspace(min(fetched_data[:,grid.indices['mass']]),max(fetched_data[:,grid.indices['mass']]),N))
        fixed_inputs=[]
        for i,param in enumerate(indices[1:]):
            fixed_inputs.append(np.ones(N)*isos[i])
        x_in = []
        for i in self.input_index:
            x_in.append(np.array([]))
        for i,iso_age in enumerate(iso_ages):
            for param in self.input_index:
                if param=='mass':
                    this_ind=self.input_index.index('mass')
                    x_in[this_ind] = np.append(x_in[this_ind],masses)
                elif param=='age':
                    this_ind=self.input_index.index('age')
                    x_in[this_ind] = np.append(x_in[this_ind],np.log10(np.ones(N)*iso_age))
                else:
                    this_ind=self.input_index.index(param)
                    x_in[this_ind] = np.append(x_in[this_ind],fixed_inputs[indices.index(param)-1])
        #NN prediction
        NN_tracks=self.model.predict(np.array(x_in).T,verbose=2).T
        NN_age=10**x_in[self.input_index.index('age')]
        
        #plotting the isochrone
        plot_data = fetchData(fetched_data, ['Teff','L','age'], grid.indices)
        [Teffm, Lm, Am, Teffg, Lg, Ag] = [NN_tracks[self.output_index.index('Teff')], NN_tracks[self.output_index.index('L')], NN_age, plot_data[0], plot_data[1], 10**plot_data[2]]    
        fig, ax=plt.subplots(1,2,figsize=[16,8])
        ax[0].scatter(Teffm,Lm,s=5,c=Am, cmap='viridis')
        ax[0].set_xlim(ax[0].get_xlim()[::-1])
        ax[0].set_ylabel(r'$\log10(L/L_{\odot})$')
        ax[0].set_xlabel(r'$\log10 T_{eff}$')
        ax[0].set_title('NN predicted')
        s2=ax[1].scatter(Teffg,Lg,s=5,c=Ag, cmap='viridis')
        ax[1].set_xlim(ax[1].get_xlim()[::-1])
        ax[1].set_ylabel(r'$\log10(L/L_{\odot})$')
        ax[1].set_xlabel(r'$\log10 T_{eff}$')
        ax[1].set_title('Real data')
        ax[0].set_xlim(ax[1].get_xlim())
        ax[0].set_ylim(ax[1].get_ylim())
        fig.subplots_adjust(right=0.83)
        cbar_ax = fig.add_axes([0.85, ax[1].get_position().y0, 0.020, ax[1].get_position().height])
        cbar_ax.text(0.5,1.015,'Age(Gyr)',fontsize=13,horizontalalignment='center',transform=cbar_ax.transAxes)
        fig.colorbar(s2, cax=cbar_ax)
        plt.show()
        if savefile != None:
            fig.savefig(savefile+'/Iso'+str(trial_no)+'.png')
            print('HR diagram saved as "'+savefile+'/Iso'+str(trial_no)+'.png"')
    
    def plotSR(self, grid, track_no, savefile=None, trial_no=None):
        """
        Plots star mass vs [log10 (delta_nu^(-4)*Teff^(3/2)] for both grid(data)
        and NN predicted (factors in the mass scaling relation). Can save plot.
        
        Parameters:
        ----------
        grid: stellarGrid object
            grid object with track data stored
        track_no: int, optional
            number of tracks to be plotted. Passed to self.prepPlot
        savefile: str, optional
            path and filename for saving the plot. Plot is only saved if not None
        trial_no: int, optional
            only used if savefile is not None. The trial number to be tagged after
            the diagram savename, matches the excel notes.
        """
        x_in = self.prepPlot(grid, track_no)[0]
        NNtracks=self.model.predict(np.array(x_in).T,verbose=2).T
        NNmass=10**x_in[self.input_index.index('mass')]
        NNx=np.log10((10**NNtracks[self.output_index.index('delnu')])**-4*
                     (10**NNtracks[self.output_index.index('Teff')])**(3/2))
        
        fig, ax=plt.subplots(1,2,figsize=[16,8])
        ax[0].scatter(NNx, NNmass, s=5, c=x_in[1], cmap='viridis')
        ax[0].set_xlabel(r'$\log10\;( \Delta \nu^{-4}{T_{eff}}^{3/2})$')
        ax[0].set_ylabel(r'$M/M_{\odot}$')
        ax[0].set_title('NN predicted')
        
        plot_data=grid.fetchData('evo',['mass','delnu','Teff', 'age'])
        mass=10**plot_data[0]
        x=np.log10((10**plot_data[1])**-4*(10**plot_data[2])**(3/2))
        s2=ax[1].scatter(x, mass, s=5, c=plot_data[3], cmap='viridis')
        ax[1].set_xlabel(r'$\log10\;( \Delta \nu^{-4}{T_{eff}}^{3/2})$')
        ax[1].set_ylabel(r'$M/M_{\odot}$')
        ax[1].set_title('Real data')
        fig.subplots_adjust(right=0.83)
        cbar_ax = fig.add_axes([0.85, ax[1].get_position().y0, 0.02, ax[1].get_position().height])
        cbar_ax.text(0.5,1.015,'log10 Age\n(Gyr)',fontsize=13,horizontalalignment='center',transform=cbar_ax.transAxes)
        fig.colorbar(s2, cax=cbar_ax)
        plt.show()
        if savefile != None:
            fig.savefig(savefile+'/SR'+str(trial_no)+'.png')
            print('SR plot saved as "'+savefile+'/SR'+str(trial_no)+'.png"')
    
    def plotDelnuAge(self, grid, track_no, savefile=None, trial_no=None):
        """
        Plots star age vs log10 delta_nu for both grid(data) and NN predicted.
        Can save plot.
        
        Parameters:
        ----------
        grid: stellarGrid object
            grid object with track data stored
        track_no: int, optional
            number of tracks to be plotted. Passed to self.prepPlot
        savefile: str, optional
            path and filename for saving the plot. Plot is only saved if not None
        trial_no: int, optional
            only used if savefile is not None. The trial number to be tagged after
            the diagram savename, matches the excel notes.
        """
        x_in = self.prepPlot(grid, track_no)[0]
        NNtracks=self.model.predict(np.array(x_in).T,verbose=2).T
        
        fig, ax=plt.subplots(1,2,figsize=[16,8])
        ax[0].scatter(NNtracks[self.output_index.index('delnu')], x_in[self.input_index.index('age')],
          s=5, c=10**x_in[self.input_index.index('mass')], cmap='viridis')
        ax[0].set_xlabel(r'$\log10\; \Delta \nu$')
        ax[0].set_ylabel(r'$\log10 Age (Gyr)$')
        ax[0].set_title('NN predicted')
        
        plot_data=grid.fetchData('evo',['age','delnu', 'mass'])
        s2=ax[1].scatter(plot_data[1], plot_data[0], s=5, c=10**plot_data[2], cmap='viridis')
        ax[1].set_xlabel(r'$\log10\; \Delta \nu$')
        ax[1].set_ylabel(r'$\log10\;Age\;(Gyr)$')
        ax[1].set_title('Real data')
        fig.subplots_adjust(right=0.83)
        cbar_ax = fig.add_axes([0.85, ax[1].get_position().y0, 0.02, ax[1].get_position().height])
        cbar_ax.text(0.5,1.015,r'$M/M_{\odot}$',fontsize=13,horizontalalignment='center',transform=cbar_ax.transAxes)
        fig.colorbar(s2, cax=cbar_ax)
        plt.show()
        if savefile != None:
            fig.savefig(savefile+'/DelnuAge'+str(trial_no)+'.png')
            print('delnu vs age plot saved as "'+savefile+'/DelnuAge'+str(trial_no)+'.png"')
    
    def lastLoss(self, key):
        """
        Returns the final training loss during training from history.
        
        Parameters:
        ----------
        key: str
            key to the history dictionary that holds the loss function,
            e.g. 'MAE'
        """
        if type(self.history)==str:
            hist=pickle.load(open( self.history, "rb" ))
        else:
            hist=self.history.history
        return hist[key][-1]
    
    def getDex(self, grid):
        """
        Calculates the accuracy of each of the NN outputs (without logs) in dex.
        
        Parameters:
        ----------
        grid: stellarGrid object
            grid object with track data stored
        
        Returns:
        ----------
        dex_values: dictionary
            accuracy of each NN output in dex, has the same length as self.output_index
            element example: {'mass': 0.005}
        """
        x_in = fetchData(grid.data, self.input_index, grid.indices)
        y_out = fetchData(grid.data, self.output_index, grid.indices)
        NN_tracks=self.model.predict(np.array(x_in).T,verbose=2).T
        dex_values = {}
        for i,Dout in enumerate(y_out):
            Mout = 10**NN_tracks[i]
            Dout = 10**Dout
            dex_values[self.output_index[i]] = np.mean(abs((Mout-Dout)/Dout))
        return dex_values
    
    def getWeights(self):
        self.weights = self.model.get_weights()
        self.no_hidden_layers = len(self.weights)/2-1
    
    def manualPredict(self, inputs):
        #input shape = 2D array with N rows of parameters, each of M columns of stars
        xx=inputs
        for i in np.arange(0,self.no_hidden_layers)*2:
            i=int(i)
            xx=T.nnet.relu(pm.math.dot(self.weights[i].T,xx).T+self.weights[i+1]).T
        xx=(T.dot(self.weights[-2].T,xx).T+self.weights[-1]).T
        return xx
