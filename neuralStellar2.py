# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 12:58:48 2020

@author: User
"""

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc("font", family="serif", size=14)
from datetime import datetime,timedelta
import pandas as pd
import pickle
import pymc3 as pm
import theano.tensor as T
import tensorflow as tf
from keras.backend import sigmoid
import seaborn as sns

class stellarGrid:
    """
    Class object that stores and process relevent information about a stellar grid.
    Proper spelling of names: step, mass, age, feh, Y, MLT, L, radius, Teff, delnu
    """
    proper_index=['step', 'mass', 'age', 'feh', 'Y', 'MLT', 'L', 'radius', 'Teff', 'delnu']
    def __init__(self, filename):
        """
        Parameters: 
        ----------
        filename: str
            path and filename of the grid, in csv form.
            
        Other class attributes:
        ----------
        data: numpy array
            stores the entire dataframe read from the grid
        indices: dictionary
            stores the name-to-column-index of the grid data, used to index the correct
            stellar parameters in the code
        ranged_tracks: numpy array/list
            stores the dataframe that is of a given age range of stars
        """
        self.file = filename
    
    def loadData(self):
        """Reads out the csv grid data"""
        self.data = pd.read_csv(self.file)
    
    def popIndex(self, names, proper=None):
        """
        Interchanges out parts of the index dictionary into names that the code uses.
        This is seperated from buildIndex method so users can print the index dictionary
        before inputting the correct key names into this method.
    
        Parameters:
        ----------
        names: str
            list of the names of keys in the original index dictionary to be replaced
            with, in the order [step, mass, age, feh, Y, MLT, luminosity, Teff, delnu]
        """
        if proper==None:
            if len(names) != len(self.proper_index):
                raise ValueError('Expecting '+str(len(self.proper_index))+' keys but '
                                 +str(len(names))+' given.')
            indexDict = {}
            for i,key in enumerate(self.proper_index):
                if names[i] != None:
                    indexDict[names[i]] = self.proper_index[i]
    
        else:
            if len(names) != len(proper):
                raise ValueError('Expecting '+str(len(proper))+' keys but '
                                 +str(len(names))+' given.')
            indexDict = {}
            for i,key in enumerate(proper):
                indexDict[names[i]] = proper[i]
        
        self.data=self.data.rename(columns=indexDict)
    
    def initialData(self, initial_columns=None):
        """
        Sorts data in increasing step number, dentifies individual stellar tracks, and 
        assign a unique track_no to each data in the dataframe
        To call for a specific track from the dataframe, use
        df.loc[df.track_no==0]      for single value, or
        df.loc[df['track_no'].isin([0,1])] for multiple values
        """
        print('Data headers = ')
        print(self.data.keys())
        
        self.data.sort_values(by='step', axis=0, inplace=True)
        self.data.set_index(keys=['step'], drop=False,inplace=True)
        
        if initial_columns is None:
            initials = np.array(self.data[['initial_mass',self.proper_index[4], 'initial_feh', self.proper_index[5]]])
        else: initials = np.array(self.data[initial_columns])
        initial_sum = np.sum(initials, axis=1)
        diff = initial_sum[1:]-initial_sum[:-1]
        boundaries = np.where(abs(diff)>0)[0]
    
        lastb = 0
        tracks = np.array([])
        for i,bi in enumerate(boundaries):
            tracks = np.concatenate((tracks,np.ones(bi+1-lastb)*i))
            lastb = bi+1
        tracks = np.concatenate((tracks, np.ones(len(self.data.index)-len(tracks))*(tracks[-1]+1)))
        self.data['track_no'] = tracks
    
    def getAgeRange(self, age_lb, age_ub):
        """
        Prunes the dataframe to get points that fall within range
        age_lb<star_age<age_ub, saves to self.ranged_tracks
        
        Parameters:
        ----------
        age_lb: float, lower boundary
        age_ub: float, upper boundary
        """
        self.ranged_tracks = self.data[self.data[self.proper_index[2]].between(age_lb, age_ub, inclusive=True)]
    
    def datatoplot(self, track_choice, track_no=None, track_index=None):
        """
        Returns the needed data in the correct format for plotting HR diagrams from
        the grid data.
    
        Parameters:
        ----------
        track_choice: str
            'evo' = to plot self.evo_tracks, 'ranged' = to plot self.ranged_tracks
            Raises NameError 'Wrong track name!' if the input does not match either 
            of them. Raises NameError 'No ranged tracks!' if the input asks for
            ranged tracks when there is no ranged tracks
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
            
        Called in: self.plotHR
        """
        if track_choice == 'evo':
            tracks = self.data
        elif track_choice == 'ranged':
            if self.ranged_tracks is None:
                raise NameError('No ranged tracks!')
            else: tracks = self.ranged_tracks
        else: raise NameError('Wrong track name!')
        
        max_tracks = tracks['track_no'].nunique()
        uniques = tracks['track_no'].unique()
        if track_no != None:
            if max_tracks > track_no:
                if type(track_index)==type(None):
                    track_index=np.random.choice(uniques, track_no)
            else: track_index = uniques
        else: track_index = uniques
        if len(track_index)>200:
            raise ValueError('Too many tracks, are you sure you want to plot '+str(len(track_index))+' tracks??')
        
        selected = tracks.loc[tracks['track_no'].isin(track_index)]
        plot_tracks = [selected['Teff'], selected['L']]
        plot_m = selected['mass']
        return plot_tracks, plot_m
    
    def plotHR(self, track_choice, track_no=None):
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
        s1=ax.scatter(np.log10(plot_tracks[0]),np.log10(plot_tracks[1]),s=5,c=plot_m,cmap='viridis')
        ax.set_xlim(ax.get_xlim()[::-1])
        ax.set_ylabel(r'$log10(L/L_{\odot})$')
        ax.set_xlabel(r'$log10\;T_{eff}$')
        fig.colorbar(s1)
        plt.show()

class NNmodel:
    """
    Class object that stores a keras model trained/to be trained on stellar
    grids, and helps plotting its training results.
    """
    def __init__(self, track_choice, input_index, output_index, 
                 non_log_columns=['feh'], Teff_scaling=1, seed=53, precision='float32'):
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
        non_log_columns: list of strings, optional
            The list of input keys that is not log10ed when passed to NN
        Teff_scaling: float, optional
            scaling factor from NN output Teff to actual Teff
            (grid Teff = NN output Teff * Teff_scaling)
        seed: int, optional
            tensorflow randomization seed
    
        Other class attributes:
        ----------
        model: keras model object
            Given by self.buildModel method
        history: keras history object or str
            If it is a keras history object, then it is directly written from a
            self.model.fit function. If it is a string, then it is the file name
            of a saved history dictionary given by self.loadHist
        """
        if track_choice=='evo' or track_choice=='ranged':
            self.track_choice = track_choice
        else: raise NameError('Wrong track name!')
        self.input_index = input_index
        self.output_index = output_index
        self.non_log_columns = non_log_columns
        self.Teff_scaling = Teff_scaling
        self.set_seed(seed)
        self.precision = precision
        self.history = None
        self.leg = {}
    
    def set_seed(self, seed):
        ''' Set the seed '''
        np.random.seed(seed)
        tf.random.set_seed(seed)
    
    def swish(self, x, beta = 1):
        return (x * sigmoid(beta * x))
    
    def normTrainInputs(self, df, input_cols):
        """
        Normalizes training inputs, log10s the data in the process
        
        Parameters:
        ----------
        df: pandas dataframe, holds all training data
        input_cols: list, holds the names of the NN input cols in the df
        
        Returns:
        ----------
        return_df: pandas dataframe, logged and normalized inputs for training
        norm_dict: dictionary, means and stds of the inputs
        
        Called in: self.fitModel
        """
        norm_dict = {}
        return_df = pd.DataFrame()
        for parameter in input_cols:
            data = np.log10(df[parameter])
            mean = np.mean(data)
            std = np.std(data)
            norm_dict[parameter] = {'mean':mean,'std':std}
            return_df = return_df.append((data-mean)/std, sort=True)
        return return_df.transpose(), norm_dict
    
    def normPredictInputs(self, array):
        """
        Normalizes prediction inputs, using information in self.history['norm'],
        requires history file to be loaded in to work, assumes input array is
        already log10ed
        Nothing will be done to the data if self.history['norm'] does not exist
        (indicating NN was not trained on normalized data)
        
        Parameters:
        ----------
        array: numpy array, already log10ed inputs
        
        Returns:
        ----------
        return_array: numpy array, normalized inputs
        
        Called in: self.plotHR, plotIsochrone, plotSR, plotDelnuAge, getDex, plotError
        """
        if 'norm' not in self.history.keys():
            return array
        else:
            return_array = []
            for i,parameter in enumerate(self.input_index):
                norm_key = list(self.history['norm'].keys())[i]
                mean,std = self.history['norm'][norm_key]['mean'],self.history['norm'][norm_key]['std']
                return_array.append((array[i]-mean)/std)
            return np.array(return_array)
    
    def buildModel(self, arch, activation, reg=None, dropout=None, summary=True):
        """
        Builds a new NN to self.model, Prints the summary.
        
        Parameters: 
        ----------
        arch: list
            The layer by layer number of nodes, includes input and output layers
            if layer is specified with 'bn', then layer is a batch normalization layer
        activation: string
            The chosen activation function
        reg: 2-long list, optional
            regularization is enforced if reg is not None.
            [0] = str, name of regularizer, l1 or l2, raises Name Error 'Wrong 
            regularizer name!' if input does not match either of the two.
            [1] = regularizer value
        dropout: int, optional
            dropout is enforced after each hidden layer if not none, dropout fraction
            given by this value
        """
        self.leg['reg'] = reg
        self.leg['dropout'] = dropout
        if reg!=None:
            if reg[0]=='l1':
                regu = keras.regularizers.l1(reg[1])
            elif reg[0]=='l2':
                regu = keras.regularizers.l2(reg[1])
            else: raise NameError('Wrong regularizer name!')
        if activation=='swish':
            activation=self.swish
        inputs = keras.Input(shape=(arch[0],))
        if arch[1]=='bn':
            xx = keras.layers.BatchNormalization()(inputs)
        else:
            if reg==None:
                xx = keras.layers.Dense(arch[1],activation=activation)(inputs)
            else: 
                xx = keras.layers.Dense(arch[1],activation=activation,kernel_regularizer=regu)(inputs)
            if dropout is not None:
                xx = keras.layers.Dropout(dropout)(xx)
        for i in range(2, len(arch)-1):
            if arch[i]=='bn':
                xx = keras.layers.BatchNormalization()(xx)
            else:
                if reg==None:
                    xx = keras.layers.Dense(arch[i],activation=activation)(xx)
                else: 
                    xx = keras.layers.Dense(arch[i],activation=activation,kernel_regularizer=regu)(xx)
                if dropout is not None:
                    xx = keras.layers.Dropout(dropout)(xx)
        outputs = keras.layers.Dense(arch[-1],activation='linear')(xx)
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='neuralstellar')
        if summary==True:
            self.model.summary()
    
    def loadModel(self, filename, summary=True):
        """
        Loads in a pre-trained/pre-built NN to self.model, prints the summary.
        """
        self.model = keras.models.load_model(filename, custom_objects={'swish':self.swish})
        if summary==True:
            self.model.summary()
        
    def compileModel(self, opt, lr, loss, metrics=None, beta_1=0.9, beta_2=0.999, 
                     decay=0, momentum=0):
        """
        Compiles self.model with Nadam optimizer.
        
        Parameters: 
        ----------
        opt: string
            name of optimizer used, Nadam or SDG
        lr: float
            learning rate
        loss: str
            metric to measure the loss
        metrics: list, optional
            list of metrics to be calculated
        beta_1 and beta_2: float, optional, as defined in keras nadam optimizer
        """
        self.leg['recompile'] = True
        self.leg['optimizer'] = opt
        self.leg['lr'] = lr
        self.leg['loss_func'] = loss
        if opt=='Nadam':
            self.leg['decay'] = 'N/A'
            self.leg['momentum'] = 'N/A'
            optimizer=keras.optimizers.Nadam(lr=lr, beta_1=beta_1, beta_2=beta_2)
        elif opt=='SGD':
            self.leg['decay'] = decay
            self.leg['momentum'] = momentum
            optimizer=keras.optimizers.SGD(learning_rate=lr, decay=decay, momentum=momentum)
        else: raise NameError('No such optimizer!!')
        if metrics!=None:
            self.model.compile(optimizer=optimizer,loss=loss, metrics=metrics)
        else: self.model.compile(optimizer=optimizer,loss=loss)
    
    def setWeights(self, model):
        """
        set the model weights to be identical to the model given
        """
        self.model.set_weights(model.get_weights())
    
    def fitModel(self, df, cols, epoch_no, batch_size, save_name, norm=False, 
                 vsplit=0.3, callback=[], baseline=0.0005, fractional_patience=0.1):
        """
        Trains self.model, saves history to self.history.
    
        Parameters: 
        ----------
        df: pandas dataframe, the track data
        cols: list
        cols[0] = input columns, cols[1] = output columns, of df
        epoch_no: int
            number of epochs to train the NN for
        batch_size: int
            batch_size of NN training, for the gird, ideally set this to the length
            of the training data
        save_name: str
            file name to save the trained NN to
        norm: bool
            whether to manually normalize input training data
        vsplit: float, optional
            validation_split during training, default to 0.1
        callback: list of strings, optional
            callback options to choose from a combination of TensorBoard,
            EarlyStopping and ModelCheckpoint
        baseline and fractional_patience: float, optional
            parameters for EarlyStopping
        """
        if self.history is None:
            self.leg['leg_no']=1
        else:
            last_leg = self.history['legs'][-1]
            self.leg['leg_no'] = last_leg['leg_no']+1
            if 'recompile' not in self.leg:
                for item in ['reg','dropout','optimizer','lr','loss_func','decay','momentum']:
                    self.leg[item] = last_leg[item]
                self.leg['recompile'] = False
        self.leg['batch_size'] = batch_size
        self.leg['epoch_no'] = epoch_no
    
        if norm==True:
            if self.history is not None:
                if 'norm' not in self.history.keys():
                    print('You assigned for input normalization but this NN was'+
                          'training on data without normalization!'+
                          '\nContinuing without normalization')
                    norm=False
        else:
            if self.history is not None:
                if 'norm' in self.history.keys():
                    print('You assigned for no input normalization but this NN was'+
                          'training on data with normalization!'+
                          '\nContinuing with normalization')
                    norm=True
    
        if norm==True:
            input_df,norm_dict = self.normTrainInputs(df,cols[0])
            x = input_df.values.astype(self.precision)
        else:
            x = np.log10(df[cols[0]].values).astype(self.precision)
        y = np.log10(df[cols[1]].values).astype(self.precision)
    
        cb=[]
        if 'tb' in callback:
            logdir = "/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
            tb = keras.callbacks.TensorBoard(log_dir=logdir)
            cb.append(tb)
        if 'es' in callback:
            patience = int(fractional_patience * epoch_no)
            es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                            baseline=baseline, patience=patience)
            cb.append(es)
        if 'mc' in callback:
            mc = keras.callbacks.ModelCheckpoint('{}_best_model.h5'.format(save_name),
                                              monitor='val_loss',
                                              mode='min', save_best_only=True)
            cb.append(mc)
    
        start_time=datetime.now()
        history = self.model.fit(x,y,
                          epochs=epoch_no,
                          batch_size=batch_size,
                          validation_split=vsplit,
                          verbose=0,
                          callbacks=cb)
        runtime = datetime.now()-start_time
        self.leg['runtime'] = runtime
        try: last_loss = history.history['mean_absolute_error'][-1]
        except KeyError: last_loss = history.history['MAE'][-1]
        self.leg['final_loss'] = last_loss
        print('training done! now='+str(datetime.now())+' | Time elapsed='+str(runtime))
        self.model.save('{}.h5'.format(save_name))
        hist = history.history
        if self.history is not None:
            for key in hist.keys():
                joined_key = self.history[key]+hist[key]
                self.history[key] = joined_key
            max_epoch = -1
            for leg in self.history['legs']:
                max_epoch+=leg['epoch_no']
            self.train_epochs=[max_epoch,max_epoch+epoch_no]
            self.leg['epoch_range'] = [max_epoch,max_epoch+epoch_no]
            self.leg['cumulative_epochs'] = max_epoch+epoch_no+1
            self.history['legs'] = self.history['legs']+[self.leg]
        else:
            self.train_epochs=[0,epoch_no-1]
            self.leg['epoch_range'] = [0,epoch_no-1]
            self.leg['cumulative_epochs'] = epoch_no
            hist['legs'] = [self.leg]
            if norm==True:
                hist['norm'] = norm_dict
            self.history = hist
        self.leg = {}
    
    def saveHist(self, filename):
        """Saves the history dictionary into a txt file with pickle"""
        with open(filename, 'wb') as file_pi:
            pickle.dump(self.history, file_pi)
    
    def loadHist(self, filename, filetype, append=False):
        """
        Passes the history file name to self.history, does basically nothing
        Note: filetype = 'pickle' or 'dill', depends on how the history file was saved
        """
        history = pickle.load(open( filename, "rb" ))
        if append==True:
            if self.history is not None:
                for key in history.keys():
                    joined_key = self.history[key]+history[key]
                    self.history[key] = joined_key
            else:
                self.history = history
        else: self.history = history
    
    def runtime(self):
        """
        calculates the total runtime spent on this model from self.history['legs']
        """
        runtime=timedelta()
        for leg in self.history['legs']:
            try:
                runtime+=leg['runtime']
            except TypeError:
                t=datetime.strptime(leg['runtime'],'%H:%M:%S.%f')
                runtime+=timedelta(hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond)
        return runtime
    
    def saveLegs(self, savefile):
        """
        saves the legs into a csv
        """
        df = pd.DataFrame()
        legs = self.history['legs'].copy()
        for leg in legs:
            leg['runtime']=str(leg['runtime'])
            for item in ['reg','epoch_range']:
                if item in leg.keys():
                    leg[item] = str(leg[item])
            df2=pd.DataFrame(leg, index=[0])
            df=df.append(df2, ignore_index=True, sort=True)
        cols = list(df.columns)
        for item in ['leg_no','final_loss','runtime']:
            cols.remove(item)
        df = df[['leg_no']+cols+['final_loss','runtime']]
        df.to_csv(path_or_buf=savefile, index=False)
    
    def fetchData(self, tracks, parameters):
        """
        Grabs the requested columns(parameters) from provided dataframe and convert
        them to be in NN-ready form
        
        Parameters: 
        ----------
        tracks: pandas dataframe
        parameters: list of strings
            columns in the dataframe to be fetch
            
        Called in: many places
        """
        if 'feh' in parameters:
            return_array=[]
            for i in parameters:
                if i in self.non_log_columns:
                    return_array.append(10**tracks[i].values)
                else: return_array.append(tracks[i].values)
            return np.log10(return_array)
        else: return np.log10(tracks[parameters].values).T
    
    def evalData(self, grid, nth_track, track_no=None):
        """
        Evaluates the NN on a given grid data, prints the result
    
        Parameters:
        ----------
        grid: stellarGrid object
        nth_track: list
            a list of the nth track to be used for evaluation, in the grid data
        """
        if track_no==None:
            if self.track_choice == 'evo':
                tracks = grid.data
            elif self.track_choice == 'ranged':
                if grid.ranged_tracks is None:
                    raise NameError('Grid has no ranged tracks!')
                else: tracks = grid.ranged_tracks
            nth_track = tracks['track_no'].unique()[nth_track]
            selected = tracks.loc[tracks['track_no'].isin(nth_track)]
        else:
            selected = self.prepPlot(grid, track_no=track_no)
        eva_in = self.fetchData(selected, self.input_index)
        eva_in = self.normPredictInputs(eva_in)
        eva_out = self.fetchData(selected, self.output_index)
        if 'Teff' in self.output_index:
            eva_out[self.output_index.index('Teff')] = eva_out[self.output_index.index('Teff')]-np.log10(self.Teff_scaling)
        print('evaluation results:')
        self.model.evaluate(eva_in.T,eva_out.T,verbose=2)
    
    def plotHist(self, plot_MSE=True, epochs=None, this_train=False, savefile=None, trial_no=None):
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
        hist = self.history
        if 'epoch' not in self.history.keys():
            max_epoch = -1
            for leg in self.history['legs']:
                max_epoch+=leg['epoch_no']
            if epochs is not None:
                if epochs[1]=='max':
                    epochs[1]=max_epoch
            epoch = np.arange(max_epoch+1)
        else:
            epoch = hist['epoch']
            if epochs is not None:
                if epochs[1]=='max':
                    epochs[1]=max(epoch)
        if this_train==True:
            epochs=self.train_epochs
        keys = hist.keys()
        if 'MAE' in keys:
            MAE,valMAE=hist['MAE'],hist['val_MAE']
        elif 'mae' in keys:
            MAE,valMAE=hist['mae'],hist['val_mae']
        elif 'mean_absolute_error' in keys:
            MAE,valMAE=hist['mean_absolute_error'],hist['val_mean_absolute_error']
        if type(epochs)!=type(None):
            epoch = epoch[epochs[0]:epochs[1]+1]
            MAE = MAE[epochs[0]:epochs[1]+1]
            valMAE = valMAE[epochs[0]:epochs[1]+1]
        fig, ax = plt.subplots(1, 1)
        ax.plot(epoch,MAE,'b',label='MAE')
        ax.plot(epoch,valMAE,'r',label='valMAE')
        if plot_MSE==True:
            if 'MSE' in keys:
                MSE,valMSE=hist['MSE'],hist['val_MSE']
            elif 'mae' in keys:
                MSE,valMSE=hist['mse'],hist['val_mse']
            elif 'mean_absolute_error' in keys:
                MSE,valMSE=hist['mean_squared_error'],hist['val_mean_squared_error']
            if type(epochs)!=type(None):
                MSE = MSE[epochs[0]:epochs[1]+1]
                valMSE = valMSE[epochs[0]:epochs[1]+1]
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
        track_no: int, optional
            number of tracks to be plotted. If None is given, will plot all tracks.
            Raises ValueError 'Too many tracks, are you sure you want to plot x
            tracks?' if number of tracks > 200.
    
        Returns:
        ----------
        selected: pandas dataframe
            the fetched tracks
        
        Called in: self.plotHR
        """
        if self.track_choice == 'evo':
            tracks = grid.data
        elif self.track_choice == 'ranged':
            if grid.ranged_tracks is None:
                raise NameError('Grid has no ranged tracks!')
            else: tracks = grid.ranged_tracks
        
        max_tracks = tracks['track_no'].nunique()
        uniques = tracks['track_no'].unique()
        if track_no != None:
            if max_tracks > track_no:
                track_index=np.random.choice(uniques, track_no)
            else: track_index = uniques
        else: track_index = uniques
        if len(track_index)>200:
            raise ValueError('Too many tracks, are you sure you want to plot '+str(len(track_index))+' tracks??')
        
        selected = tracks.loc[tracks['track_no'].isin(track_index)]
        return selected
    
    def calOutputs(self, y_out, check_L=True):
        """
        Scales Teff correctly, and if check_L==True and there is no luminosity among
        NN outputs, calculates luminosity from radius and Teff by:
        L = 4*pi*R**2*boltzmann_constant*Teff**4
        """
        output_index = self.output_index.copy()
        if 'Teff' in self.output_index:
            y_out[self.output_index.index('Teff')] = y_out[self.output_index.index('Teff')]+np.log10(self.Teff_scaling)
        if check_L:
            if 'L' not in self.output_index:
                if 'radius' in self.output_index and 'Teff' in self.output_index:
                    radius = 10**y_out[self.output_index.index('radius')]
                    Teff = 10**y_out[self.output_index.index('Teff')]
                    y_out[self.output_index.index('radius')] = np.log10(radius**2*(Teff/5776.02970722)**4)
                    output_index[output_index.index('radius')] = 'L'
                else: raise NameError('Missing means to calculate luminosity!\nOutput options = '+str(self.output_index))
        return y_out, output_index
    
    def plotHR(self, grid, track_no=20, savefile=None, trial_no=None):
        """
        Plots both grid(data) and NN predicted HR diagrams. Can save
        plot.
    
        Parameters:
        ----------
        grid: stellarGrid object
        track_no: int, optional
            number of tracks to be plotted. Passed to self.prepPlot
        savefile: str, optional
            path and filename for saving the plot. Plot is only saved if not None
        trial_no: int, optional
            only used if savefile is not None. The trial number to be tagged after
            the diagram savename, matches the excel notes.
        """
        tracks = self.prepPlot(grid, track_no)
        plot_tracks = np.log10([tracks['Teff'], tracks['L']])
        plot_m = tracks['mass']
        x_in = self.fetchData(tracks, self.input_index)
        x_in = self.normPredictInputs(x_in)
        y_out = self.model.predict(x_in.T,verbose=2, batch_size=len(x_in.T)).T
        y_out, output_index = self.calOutputs(y_out)
        [Teffm, Lm, Teffg, Lg, M] = [y_out[output_index.index('Teff')], y_out[output_index.index('L')], 
                                          plot_tracks[0], plot_tracks[1], plot_m]
    
        fig, ax=plt.subplots(1,2,figsize=[16,8])
        ax[0].scatter(Teffm,Lm,s=5,c=M, cmap='viridis')
        ax[0].set_xlim(ax[0].get_xlim()[::-1])
        ax[0].set_ylabel(r'$\log10(L/L_{\odot})$')
        ax[0].set_xlabel(r'$\log10 T_{eff}$')
        ax[0].set_title('NN predicted')
        s2=ax[1].scatter(Teffg,Lg,s=5,c=M, cmap='viridis')
        ax[1].set_xlim(ax[1].get_xlim()[::-1])
        ax[1].set_ylabel(r'$\log10(L/L_{\odot})$')
        ax[1].set_xlabel(r'$\log10 T_{eff}$')
        ax[1].set_title('MESA data')
        fig.subplots_adjust(right=0.83)
        cbar_ax = fig.add_axes([0.85, ax[1].get_position().y0, 0.02, ax[1].get_position().height])
        cbar_ax.text(0.5,1.015,r'$M/M_{\odot}$',fontsize=13,horizontalalignment='center',transform=cbar_ax.transAxes)
        fig.colorbar(s2, cax=cbar_ax)
        plt.show()
        if savefile != None:
            fig.savefig(savefile+'/HR'+str(trial_no)+'.png')
            print('HR diagram saved as "'+savefile+'/HR'+str(trial_no)+'.png"')
    
    def plotIsochrone(self, grid, iso_ages, indices, isos, widths=None, N=5000, 
                      one_per_track=True, extended=True, savefile=None, trial_no=None):
        """
        Plots both grid(data) and NN predicted isochrones of specified ages in HR
        diagrams, with the colour bar showing variation in age. Can save plot.
    
        Parameters:
        ----------
        grid: stellarGrid object
        iso_ages: list/array
            the isochronic ages to be plotted
        indices: list of strings, length = len(self.input_index)-1 (has age)
            the order of NN input parameters (stellar fundamentals) of which to
            constraint, in the form(spelling) of stellarGrid.proper_index.
            'age' must be at the first index.
        isos: list/array of, length = len(self.input_index)-2 (no age)
            the 'mean values', mu of the parameters other than age to be constrainted
        widths: 2D list, length = len(self.input_index)-1 (has age), optional
            the 'boundary width', delta to be added onto the relevent isos number, making
            the selector select data falling between mu-delta and mu+delta.
            For each (ith) input parameter, if widths[i][0]=='r', boundary width is 
            calculated in ratio mode: ie. delta = isos[i-1]*widths[i][1]
            if widths[i][0]=='a', boundary width is absolute: ie. delta = widths[i][1].
            If the iso parameter (mu) is zero and widths[i][0] is set to 'r',
            will force an arbituary width of 0.05 to this instance of the parameter.
        one_per_track: bool, optional
            if True: each track only supply one point in the grid plot
            if False: multiple points can be supplied per track
        extended: bool, optional
            if True: NN side plots the entire mass range of datapoints picked from the grid,
                for all individual isochrones.
            If False: NN side's mass input range depends on the individual groups of
                datapoints picked from the grid for the corresponding iso-age
    
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
        if widths is None:
            widths = []
            for i in indices:
                if i == 'feh' or i == 'MLT':
                    widths.append(['a',0.1])
                elif i == 'Y':
                    widths.append(['a',0.015])
                else: widths.append(['r',0.05])
        #checking for correct input lengths
        if len(indices)!=len(self.input_index)-1 or len(isos)!=len(self.input_index)-2 or len(widths)!=len(self.input_index)-1:
            raise ValueError('Two (or more) of the lengths of the inputs do not match!')
        
        data = grid.data
        for i,param in enumerate(indices[1:]):
            if widths[i+1][0]=='r':
                if isos[i]!=0:
                    this_width = isos[i]*widths[i+1][1]
                else: this_width = 0.05
            else: this_width = widths[i+1][1]
            data = data[(data[param] >= isos[i]-this_width) & (data[param] <= isos[i]+this_width)]
        fetched_data = pd.DataFrame()
        mass_ranges = []
        for iso_age in iso_ages:
            if widths[0][0]=='r':
                if iso_age!=0:
                    age_width = iso_age*widths[0][1]
                else: age_width = 0.05
            else: age_width=widths[0][1]
            all_tracks = data[(data['age'] >= iso_age-age_width) & (data['age'] <= iso_age+age_width)]
            mass_ranges.append([min(all_tracks['mass'].unique()),max(all_tracks['mass'].unique())])
            if one_per_track==True:
                for i,track_no in enumerate(all_tracks['track_no'].unique()):
                    track_data = all_tracks.loc[all_tracks.track_no==track_no]
                    if len(track_data.index)>1:
                        age_dist = abs(track_data['age'].values-iso_age)
                        fetched_data = fetched_data.append(track_data.iloc[np.argmin(age_dist)],sort=True)
                    else: fetched_data = fetched_data.append(track_data,sort=True)
            else:
                fetched_data = fetched_data.append(all_tracks, sort=True)
        print('found '+str(len(fetched_data.index))+' stars.')
        
        #creating new input lists for NN to predict
        fixed_inputs=[]
        for i,param in enumerate(indices[1:]):
            if param in self.non_log_columns:
                fixed_inputs.append(np.ones(N)*isos[i])
            else: fixed_inputs.append(np.log10(np.ones(N)*isos[i]))
        x_in = []
        for i in self.input_index:
            x_in.append(np.array([]))
        for i,iso_age in enumerate(iso_ages):
            for param in self.input_index:
                if param=='mass':
                    this_ind=self.input_index.index('mass')
                    if extended==False:
                        masses = np.log10(np.linspace(*mass_ranges[i],N))
                    else: masses = np.log10(np.linspace(min(fetched_data['mass']),max(fetched_data['mass']),N))
                    x_in[this_ind] = np.append(x_in[this_ind],masses)
                elif param=='age':
                    this_ind=self.input_index.index('age')
                    x_in[this_ind] = np.append(x_in[this_ind],np.log10(np.ones(N)*iso_age))
                else:
                    this_ind=self.input_index.index(param)
                    x_in[this_ind] = np.append(x_in[this_ind],fixed_inputs[indices.index(param)-1])
        
        #NN prediction
        x_in = np.array(x_in)
        NN_age=10**x_in[self.input_index.index('age')]
        x_in = self.normPredictInputs(x_in)
        NN_tracks=np.log10(10**self.model.predict(x_in.T, batch_size=len(x_in.T),verbose=2).T)
        NN_tracks, output_index = self.calOutputs(NN_tracks)
        
        #plotting the isochrone
        plot_data = self.fetchData(fetched_data, ['Teff','L','age'])
        [Teffm, Lm, Am, Teffg, Lg, Ag] = [NN_tracks[output_index.index('Teff')],
                                          NN_tracks[output_index.index('L')],
                                          NN_age, plot_data[0], plot_data[1], 10**plot_data[2]]
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
        ax[1].set_title('MESA data')
        ax[0].set_xlim(ax[1].get_xlim())
        ax[0].set_ylim(ax[1].get_ylim())
        fig.subplots_adjust(right=0.83)
        cbar_ax = fig.add_axes([0.85, ax[1].get_position().y0, 0.020, ax[1].get_position().height])
        cbar_ax.text(0.5,1.015,'Age(Gyr)',fontsize=13,horizontalalignment='center',transform=cbar_ax.transAxes)
        fig.colorbar(s2, cax=cbar_ax)
        plt.show()
        if savefile != None:
            fig.savefig(savefile+'/Iso'+str(trial_no)+'.png')
            print('Isochrone saved as "'+savefile+'/Iso'+str(trial_no)+'.png"')
    
    def plotSR(self, grid, track_no=20, savefile=None, trial_no=None):
        """
        Plots star mass vs [log10 (delta_nu^(-4)*Teff^(3/2)] for both grid(data)
        and NN predicted (factors in the mass scaling relation). Can save plot.
    
        Parameters:
        ----------
        grid: stellarGrid object
        track_no: int, optional
            number of tracks to be plotted. Passed to self.prepPlot
        savefile: str, optional
            path and filename for saving the plot. Plot is only saved if not None
        trial_no: int, optional
            only used if savefile is not None. The trial number to be tagged after
            the diagram savename, matches the excel notes.
        """
        tracks = self.prepPlot(grid, track_no)
        plot_data = self.fetchData(tracks, ['mass','delnu','Teff', 'age'])
        mass=10**plot_data[0]
        SRx=np.log10((10**plot_data[1])**-4*(10**plot_data[2])**(3/2))
        
        x_in = self.fetchData(tracks, self.input_index)
        NNmass=10**x_in[self.input_index.index('mass')]
        NNage=x_in[1]
        x_in = self.normPredictInputs(x_in)
        NNtracks = self.model.predict(x_in.T,batch_size=len(x_in.T),verbose=2).T
        NNtracks, output_index = self.calOutputs(NNtracks)
        NNx=np.log10((10**NNtracks[output_index.index('delnu')])**-4*
                     (10**NNtracks[output_index.index('Teff')])**(3/2))
    
        fig, ax=plt.subplots(1,2,figsize=[16,8])
        ax[0].scatter(NNx, NNmass, s=5, c=NNage, cmap='viridis')
        ax[0].set_xlabel(r'$\log10\;( \Delta \nu^{-4}{T_{eff}}^{3/2})$')
        ax[0].set_ylabel(r'$M/M_{\odot}$')
        ax[0].set_title('NN predicted')
        s2=ax[1].scatter(SRx, mass, s=5, c=plot_data[3], cmap='viridis')
        ax[1].set_xlabel(r'$\log10\;( \Delta \nu^{-4}{T_{eff}}^{3/2})$')
        ax[1].set_ylabel(r'$M/M_{\odot}$')
        ax[1].set_title('MESA data')
        fig.subplots_adjust(right=0.83)
        cbar_ax = fig.add_axes([0.85, ax[1].get_position().y0, 0.02, ax[1].get_position().height])
        cbar_ax.text(0.5,1.015,'log10 Age\n(Gyr)',fontsize=13,horizontalalignment='center',transform=cbar_ax.transAxes)
        fig.colorbar(s2, cax=cbar_ax)
        plt.show()
        if savefile != None:
            fig.savefig(savefile+'/SR'+str(trial_no)+'.png')
            print('SR plot saved as "'+savefile+'/SR'+str(trial_no)+'.png"')
    
    def plotDelnuAge(self, grid, track_no=20, savefile=None, trial_no=None):
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
        tracks = self.prepPlot(grid, track_no)
        plot_data = self.fetchData(tracks, ['age','delnu', 'mass'])
        x_in = self.fetchData(tracks, self.input_index)
        x_in_norm = self.normPredictInputs(x_in)
        NNtracks = self.model.predict(x_in_norm.T,batch_size=len(x_in_norm.T),verbose=2).T
    
        fig, ax=plt.subplots(1,2,figsize=[16,8])
        ax[0].scatter(NNtracks[self.output_index.index('delnu')], x_in[self.input_index.index('age')],
          s=5, c=10**x_in[self.input_index.index('mass')], cmap='viridis')
        ax[0].set_xlabel(r'$\log10\; \Delta \nu$')
        ax[0].set_ylabel(r'$\log10 Age (Gyr)$')
        ax[0].set_title('NN predicted')
        s2=ax[1].scatter(plot_data[1], plot_data[0], s=5, c=10**plot_data[2], cmap='viridis')
        ax[1].set_xlabel(r'$\log10\; \Delta \nu$')
        ax[1].set_ylabel(r'$\log10\;Age\;(Gyr)$')
        ax[1].set_title('MESA data')
        fig.subplots_adjust(right=0.83)
        cbar_ax = fig.add_axes([0.85, ax[1].get_position().y0, 0.02, ax[1].get_position().height])
        cbar_ax.text(0.5,1.015,r'$M/M_{\odot}$',fontsize=13,horizontalalignment='center',transform=cbar_ax.transAxes)
        fig.colorbar(s2, cax=cbar_ax)
        plt.show()
        if savefile != None:
            fig.savefig(savefile+'/DelnuAge'+str(trial_no)+'.png')
            print('delnu vs age plot saved as "'+savefile+'/DelnuAge'+str(trial_no)+'.png"')
        
    def plotAll(self, grid, track_no=100, savefile=None, trial_no=None):
        """
        Plotting the three plots HR, SR and DelnuAge in one go
        """
        self.plotHR(grid, track_no=track_no, savefile=savefile, trial_no=trial_no)
        self.plotSR(grid, track_no=track_no, savefile=savefile, trial_no=trial_no)
        self.plotDelnuAge(grid, track_no=track_no, savefile=savefile, trial_no=trial_no)
    
    def lastLoss(self, key):
        """
        Returns the final training loss during training from history.
    
        Parameters:
        ----------
        key: str
            key to the history dictionary that holds the loss function,
            e.g. 'MAE'
        """
        return self.history[key][-1]
    
    def getDex_old(self, grid):
        """
        Note: old and wrong dex function, calculates the absolute error ratio instead
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
        tracks = self.prepPlot(grid, track_no=200)
        x_in = self.fetchData(tracks, self.input_index)
        x_in = self.normPredictInputs(x_in)
        y_out = self.fetchData(tracks, self.output_index)
        NN_tracks=self.model.predict(x_in.T,len(x_in.T),verbose=2).T
        dex_values = {}
        for i,Dout in enumerate(y_out):
            Mout = 10**NN_tracks[i]
            Dout = 10**Dout
            dex_values[self.output_index[i]] = np.mean(abs((Mout-Dout)/Dout))
        return dex_values
    
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
        tracks = self.prepPlot(grid, track_no=200)
        x_in = self.fetchData(tracks, self.input_index)
        x_in = self.normPredictInputs(x_in)
        y_out = self.fetchData(tracks, self.output_index)
        NN_tracks = self.model.predict(x_in.T,batch_size=len(x_in.T),verbose=2).T
        NN_tracks = self.calOutputs(NN_tracks,check_L=False)[0]
        dex_values = {}
        for i,Dout in enumerate(y_out):
            Mout = 10**NN_tracks[i]
            Dout = 10**Dout
            dex_values[self.output_index[i]] = np.mean(abs((Mout-Dout)/Dout)*np.log10(Dout))/np.log(10)
        return dex_values
    
    def plotError(self, grid, std_limit=True):
        x_in = self.fetchData(grid.data, self.input_index)
        x_in = self.normPredictInputs(x_in)
        y_out = self.fetchData(grid.data, ['L','Teff','delnu'])
        NN_tracks = self.model.predict(x_in.T,batch_size=len(x_in.T),verbose=2).T
        NN_tracks, output_index = self.calOutputs(NN_tracks)
        fig, ax = plt.subplots(1,3,figsize=[15,4])
        x_labels = [r'luminosity ($L_\odot$)',r'$T_{eff}$ (K)',r'$\Delta \nu$ ($\mu$Hz)']
        for i,Dout in enumerate(y_out):
            Mout = 10**NN_tracks[i]
            Dout = 10**Dout
            errors = Mout-Dout
            std = np.std(errors)
            median = np.round(np.median(abs(errors)),2)
            sns.distplot(errors, bins=200, ax=ax[i], label='median absolute ='+str(median))
            ax[i].set_xlabel(x_labels[i])
            ax[i].legend()
            if std_limit==True:
                ax[i].set_xlim([-10*std,10*std])
        plt.show()
    
    def manualPredict(self, inputs):
        """
        "Manual calculation" of a NN done with theano tensors, for pymc3 to use
        """
        xx = T.transpose(inputs)
        for i,layer in enumerate(self.model.layers):
            if i == len(self.model.layers)-1:
                weights = layer.get_weights()
                xx=T.dot(xx,weights[0])+weights[1]
            elif 'batch_normalization' in layer.get_config()['name']:
                weights = layer.get_weights()
                xx=T.nnet.bn.batch_normalization_test(xx,*weights,epsilon=0.001)
            elif 'dense' in layer.get_config()['name']:
                weights = layer.get_weights()
                xx=T.nnet.elu(pm.math.dot(xx,weights[0])+weights[1])
        return xx.T
