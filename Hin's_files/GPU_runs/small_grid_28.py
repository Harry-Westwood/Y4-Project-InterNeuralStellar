# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 01:49:00 2019

@author: User
"""

from neuralStellar import *
from datetime import datetime
import os

time_now=datetime.now().strftime("%Y-%m-%d_%H%M%S")
folder_name='Hin_gridNN_outputs_'+time_now
os.mkdir(folder_name)

file='grid_0_0.csv'
small_grid=stellarGrid(file)
small_grid.buildIndex()
small_grid.popIndex(['','star_mass','star_age','star_feh','star_MLT','effective_T','luminosity','delta_nu'],
                    proper=['step','mass','age','feh','MLT','Teff','L','delnu'])
small_grid.initialData()

in_dex=['mass','age','feh','MLT']
out_dex=['L','Teff','delnu']
x_in=small_grid.fetchData('evo',in_dex)
y_out=small_grid.fetchData('evo',out_dex)
x_in, y_out=shuffleInputs(x_in,y_out)
m1=NNmodel('evo',in_dex, out_dex)
m1.buildModel([len(x_in),len(y_out)], 8, 128, reg=['l2',0.0001])
m1.compileModel(0.001,'MAE',metrics=['MAE','MSE'], beta_1=0.9999, beta_2=0.999)
m1.fitModel(x_in, y_out, 500000, len(x_in[0]),folder_name+'/small_grid_model.h5', keep_log=False)
m1.saveHist(folder_name+'/trainHistoryDict')