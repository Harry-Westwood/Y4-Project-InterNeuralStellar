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
import pandas as pd
import pickle
import dill
import pymc3 as pm
import theano.tensor as T

class stellarGrid:
    """
    Class object that stores and process relevent information about a stellar grid.
    """
    proper_index = ['step', 'mass', 'age', 'feh', 'Y', 'MLT', 'L', 'Teff', 'delnu']
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
        
    def buildIndex(self):
        """Reads out the headers on the grid csv and saves it as self.indices dictonary"""
        self.data = pd.read_csv(self.file)
        headers = self.data.keys()
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
            with, in the order [step, mass, age, feh, Y, MLT, luminosity, Teff, delnu]
        """
        if proper==None:
            if len(names) != len(self.proper_index):
                raise ValueError('Expecting '+str(len(self.proper_index))+' keys but '
                                 +str(len(names))+' given.')
            indexDict = {}
            for i,key in enumerate(self.proper_index):
                if names[i] != None:
                    self.indices[key] = self.indices.pop(names[i])
                    indexDict[names[i]] = self.proper_index[i]
    
        else:
            if len(names) != len(proper):
                raise ValueError('Expecting '+str(len(proper))+' keys but '
                                 +str(len(names))+' given.')
            indexDict = {}
            for i,key in enumerate(proper):
                self.indices[key] = self.indices.pop(names[i])
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
    def __init__(self, track_choice, input_index, output_index, non_log_columns=['feh']):
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
    
    def loadModel(self, filename):
        """
        Loads in a pre-trained/pre-built NN to self.model, prints the summary.
        """
        self.model = keras.models.load_model(filename)
        self.model.summary()
    
    def loadHist(self, filename, filetype):
        """
        Passes the history file name to self.history, does basically nothing
        Note: filetype = 'pickle' or 'dill', depends on how the history file was saved
        """
        self.history = filename
        self.hist_type = filetype
    
    def fetchData(self, tracks, parameters):
        """
        Grabs the requested columns(parameters) from provided dataframe and convert
        them to be in NN-ready form
        
        Parameters: 
        ----------
        tracks: pandas dataframe
        parameters: list of strings
            columns in the dataframe to be fetch
        """
        if 'feh' in parameters:
            return_array=[]
            for i in parameters:
                if i in self.non_log_columns:
                    return_array.append(10**tracks[i].values)
                else: return_array.append(tracks[i].values)
            return np.log10(return_array)
        else: return np.log10(tracks[parameters].values).T
    
    def evalData(self, grid, nth_track):
        """
        Evaluates the NN on a given grid data, prints the result
    
        Parameters:
        ----------
        grid: stellarGrid object
        nth_track: list
            a list of the nth track to be used for evaluation, in the grid data
        """
        if self.track_choice == 'evo':
            tracks = grid.data
        elif self.track_choice == 'ranged':
            if grid.ranged_tracks is None:
                raise NameError('Grid has no ranged tracks!')
            else: tracks = grid.ranged_tracks
        nth_track = tracks['track_no'].unique()[nth_track]
        eva_in = self.fetchData(tracks.loc[tracks['track_no'].isin(nth_track)], self.input_index)
        eva_out = self.fetchData(tracks.loc[tracks['track_no'].isin(nth_track)], self.output_index)
        print('evaluation results:')
        self.model.evaluate(eva_in.T,eva_out.T,verbose=2)
    
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
        if self.hist_type=='pickle':
            hist=pickle.load(open( self.history, "rb" ))
            epoch=hist['epoch']
        elif self.hist_type=='dill':
            hist=dill.load(open( self.history, "rb" ))
            epoch=np.arange(hist['epoch'])
        else:
            raise NameError('Incorrect history type '+str(self.hist_type)+'!')
        keys = hist.keys()
        if 'MAE' in keys:
            MAE,valMAE=hist['MAE'],hist['val_MAE']
        elif 'mae' in keys:
            MAE,valMAE=hist['mae'],hist['val_mae']
        elif 'mean_absolute_error' in keys:
            MAE,valMAE=hist['mean_absolute_error'],hist['val_mean_absolute_error']
        if type(epochs)!=type(None):
            epoch = epoch[epochs[0]:epochs[1]]
            MAE = MAE[epochs[0]:epochs[1]]
            valMAE = valMAE[epochs[0]:epochs[1]]
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
        y_out = self.model.predict(x_in.T,verbose=2).T
        [Teffm, Lm, Teffg, Lg, M] = [y_out[self.output_index.index('Teff')], y_out[self.output_index.index('L')], 
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
        ax[1].set_title('Real data')
        fig.subplots_adjust(right=0.83)
        cbar_ax = fig.add_axes([0.85, ax[1].get_position().y0, 0.02, ax[1].get_position().height])
        cbar_ax.text(0.5,1.015,r'$M/M_{\odot}$',fontsize=13,horizontalalignment='center',transform=cbar_ax.transAxes)
        fig.colorbar(s2, cax=cbar_ax)
        plt.show()
        if savefile != None:
            fig.savefig(savefile+'/HR'+str(trial_no)+'.png')
            print('HR diagram saved as "'+savefile+'/HR'+str(trial_no)+'.png"')
    
    def plotIsochrone(self, grid, iso_ages, indices, isos, widths=None, N=5000, savefile=None, trial_no=None):
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
        for iso_age in iso_ages:
            if widths[0][0]=='r':
                if iso_age!=0:
                    age_width = iso_age*widths[0][1]
                else: age_width = 0.05
            else: age_width=widths[0][1]
            all_tracks = data[(data['age'] >= iso_age-age_width) & (data['age'] <= iso_age+age_width)]
            for i,track_no in enumerate(all_tracks['track_no'].unique()):
                track_data = all_tracks.loc[all_tracks.track_no==track_no]
                if len(track_data.index)>1:
                    age_dist = abs(track_data['age'].values-iso_age)
                    fetched_data = fetched_data.append(track_data.iloc[np.argmin(age_dist)],sort=True)
                else: fetched_data = fetched_data.append(track_data,sort=True)
        print('found '+str(len(fetched_data.index))+' stars.')
        
        #creating new input lists for NN to predict
        masses = np.log10(np.linspace(min(fetched_data['mass']),max(fetched_data['mass']),N))
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
                    x_in[this_ind] = np.append(x_in[this_ind],masses)
                elif param=='age':
                    this_ind=self.input_index.index('age')
                    x_in[this_ind] = np.append(x_in[this_ind],np.log10(np.ones(N)*iso_age))
                else:
                    this_ind=self.input_index.index(param)
                    x_in[this_ind] = np.append(x_in[this_ind],fixed_inputs[indices.index(param)-1])
        
        #NN prediction
        NN_tracks=np.log10(10**self.model.predict(np.array(x_in).T,verbose=2).T)
        NN_age=10**x_in[self.input_index.index('age')]
        
        #plotting the isochrone
        plot_data = self.fetchData(fetched_data, ['Teff','L','age'])
        [Teffm, Lm, Am, Teffg, Lg, Ag] = [NN_tracks[self.output_index.index('Teff')],
                                          NN_tracks[self.output_index.index('L')],
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
        NNtracks = self.model.predict(x_in.T,verbose=2).T
        NNmass=10**x_in[self.input_index.index('mass')]
        NNx=np.log10((10**NNtracks[self.output_index.index('delnu')])**-4*
                     (10**NNtracks[self.output_index.index('Teff')])**(3/2))
    
        fig, ax=plt.subplots(1,2,figsize=[16,8])
        ax[0].scatter(NNx, NNmass, s=5, c=x_in[1], cmap='viridis')
        ax[0].set_xlabel(r'$\log10\;( \Delta \nu^{-4}{T_{eff}}^{3/2})$')
        ax[0].set_ylabel(r'$M/M_{\odot}$')
        ax[0].set_title('NN predicted')
        s2=ax[1].scatter(SRx, mass, s=5, c=plot_data[3], cmap='viridis')
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
        NNtracks = self.model.predict(x_in.T,verbose=2).T
    
        fig, ax=plt.subplots(1,2,figsize=[16,8])
        ax[0].scatter(NNtracks[self.output_index.index('delnu')], x_in[self.input_index.index('age')],
          s=5, c=10**x_in[self.input_index.index('mass')], cmap='viridis')
        ax[0].set_xlabel(r'$\log10\; \Delta \nu$')
        ax[0].set_ylabel(r'$\log10 Age (Gyr)$')
        ax[0].set_title('NN predicted')
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
        if self.hist_type=='pickle':
            hist=pickle.load(open( self.history, "rb" ))
        elif self.hist_type=='dill':
            hist=dill.load(open( self.history, "rb" ))
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
        tracks = self.prepPlot(grid, track_no=200)
        x_in = self.fetchData(tracks, self.input_index)
        y_out = self.fetchData(tracks, self.output_index)
        NN_tracks=self.model.predict(x_in.T,verbose=2).T
        dex_values = {}
        for i,Dout in enumerate(y_out):
            Mout = 10**NN_tracks[i]
            Dout = 10**Dout
            dex_values[self.output_index[i]] = np.mean(abs((Mout-Dout)/Dout))
        return dex_values
    
    def getWeights(self):
        """
        pass self.weights the NN's weights and calculate number of hidden layers
        """
        self.weights = self.model.get_weights()
        self.no_hidden_layers = len(self.weights)/2-1
    
    def manualPredict(self, inputs):
        """
        "Manual calculation" of a NN done with theano tensors, for pymc3 to use
        """
        xx=T.nnet.elu(pm.math.dot(self.weights[0].T,inputs).T+self.weights[1])
        for i in np.arange(1,self.no_hidden_layers)*2:
            i=int(i)
            xx=T.nnet.elu(pm.math.dot(xx,self.weights[i])+self.weights[i+1])
        xx=(T.dot(xx,self.weights[-2])+self.weights[-1])
        return xx.T
    