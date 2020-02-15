#add that it prints which leg is currently being done and use the history to print that leg's properties
#add the extended isochrone to the large end plot
#...unlikely you'll need to change it so that you can freely changed between MSE and MAE whilst training. 

from datetime import datetime
import numpy as np

from matplotlib import rc
rc("font", family="serif", size=14)
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l1, l2
import pickle
from shutil import copyfile


import smtplib, ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

from neuralStellar2 import stellarGrid,NNmodel

def set_seed(self, seed):
    ''' Set the seed '''
    np.random.seed(seed)
    tf.random.set_random_seed(seed)

def Leg_Setup(folder_name,reg,lr,epoch_no,batch_size,opt,activation,dropout,momentum,decay,mean_error_type,recompile=False):
    leg_num = Get_Num_Legs(reg=reg,lr=lr,epoch_no=epoch_no,batch_size=batch_size,opt=opt,activation=activation,dropout=dropout, momentum=momentum, decay=decay, mean_error_type=mean_error_type)
    legs = []
    for leg in range(leg_num):
        leg_dict = {'contNN': False,'changeingNNparam':False}
        
        if leg > 0:
            leg_dict['contNN'] = True

        #if always_recompile == True:
        #    leg_dict['changeingNNparam'] = True
        if type(recompile) is list:
            try:
                 leg_dict['changeingNNparam'] = recompile[leg]    
            except IndexError:
                #leg_dict['changeingNNparam'] = recompile[-1]
                pass
        else:
            leg_dict['changeingNNparam'] = recompile

        
        if reg != None:
            if type(reg) is list:
                    try:
                         leg_dict['reg'] = reg[leg]
                         if leg > 0:
                             if reg[leg] != reg[leg-1]:
                                 leg_dict['changeingNNparam'] = True
                    except IndexError:
                        leg_dict['reg'] = reg[-1]
        else:
                leg_dict['reg'] = None
                
        if type(lr) is list:
            try:
                 leg_dict['lr'] = lr[leg]
                 if leg > 0:
                     if lr[leg] != lr[leg-1]:
                         leg_dict['changeingNNparam'] = True
            except IndexError:
                leg_dict['lr'] = lr[-1]
        else:
            leg_dict['lr'] = lr

        if type(epoch_no) is list:
            try:
                 leg_dict['epoch_no'] = epoch_no[leg]
            except IndexError:
                leg_dict['epoch_no'] = epoch_no[-1]
        else:
            leg_dict['epoch_no'] = epoch_no

        if type(batch_size) is list:
            try:
                 leg_dict['batch_size'] = batch_size[leg]
            except IndexError:
                leg_dict['batch_size'] = batch_size[-1]   
        else:
            leg_dict['batch_size'] = batch_size

        if type(opt) is list:
            try:
                 leg_dict['opt'] = opt[leg]
                 if leg > 0:
                     if opt[leg] != opt[leg-1]:
                         leg_dict['changeingNNparam'] = True
            except IndexError:
                leg_dict['opt'] = opt[-1]   
        else:
            leg_dict['opt'] = opt

        if type(activation) is list:
            try:
                 leg_dict['activation'] = activation[leg]
                 if leg > 0:
                     if activation[leg] != activation[leg-1]:
                         leg_dict['changeingNNparam'] = True
            except IndexError:
                leg_dict['activation'] = activation[-1]   
        else:
            leg_dict['activation'] = activation

        if type(dropout) is list:
            try:
                 leg_dict['dropout'] = dropout[leg]
                 if leg > 0:
                     if dropout[leg] != dropout[leg-1]:
                         leg_dict['changeingNNparam'] = True
            except IndexError:
                leg_dict['dropout'] = dropout[-1]    
        else:
            leg_dict['dropout'] = dropout

        if type(momentum) is list:
            try:
                 leg_dict['momentum'] = momentum[leg]
                 if leg > 0:
                     if momentum[leg] != momentum[leg-1]:
                         leg_dict['changeingNNparam'] = True
            except IndexError:
                leg_dict['momentum'] = momentum[-1]    
        else:
            leg_dict['momentum'] = momentum

        if type(decay) is list:
            try:
                 leg_dict['decay'] = decay[leg]
                 if leg > 0:
                     if decay[leg] != decay[leg-1]:
                         leg_dict['changeingNNparam'] = True
            except IndexError:
                leg_dict['decay'] = decay[-1]    
        else:
            leg_dict['decay'] = decay

        if type(mean_error_type) is list:
            try:
                 leg_dict['mean_error_type'] = mean_error_type[leg]
                 if leg > 0:
                     if mean_error_type[leg] != mean_error_type[leg-1]:
                         leg_dict['changeingNNparam'] = True
            except IndexError:
                leg_dict['mean_error_type'] = mean_error_type[-1]   
        else:
                leg_dict['mean_error_type'] = mean_error_type

        legs.append(leg_dict)
    return legs
        
def NN_run(folder_name, df, hidden_layers, legs, leg):
    contNN = legs[leg]['contNN']
    changeingNNparam = legs[leg]['changeingNNparam']
    activation = legs[leg]['activation']
    reg = legs[leg]['reg']
    dropout = legs[leg]['dropout']
    opt = legs[leg]['opt']
    lr = legs[leg]['lr']
    epoch_no = legs[leg]['epoch_no']
    batch_size = legs[leg]['batch_size']
    decay = legs[leg]['decay']
    momentum = legs[leg]['momentum']
    mean_error_type = legs[leg]['mean_error_type']

    if contNN == True and changeingNNparam == True:
        m0=NNmodel('evo',['mass', 'age', 'feh', 'Y', 'MLT'], ['L', 'Teff', 'delnu']) #not compiling NN, but different...who knows???
        m0.loadModel('drive/My Drive/4th Year Project/{}/{}_best_model.h5'.format(folder_name,folder_name),summary = False)
    if contNN == False:
        start_epoch = 0
        #from datetime import datetime
    x_cols = ['star_mass', 'star_age', 'tenstarfeh', 'initial_Yinit', 'initial_MLT'] #inputs cols
    y_cols = ['radius', 'scale_T', 'delta_nu'] # output cols
    arch = [len(x_cols)]+hidden_layers+[len(y_cols)]
    m1=NNmodel('evo',['mass', 'age', 'feh', 'Y', 'MLT'], ['L', 'Teff', 'delnu'])

    if contNN == True:
        m1.loadHist('drive/My Drive/4th Year Project/{}/Hist{}'.format(folder_name,folder_name),'pickle')
        start_epoch = m1.history['legs'][-1]['epoch_no']

    if (contNN == True and changeingNNparam == True) or contNN == False:
        #m1.buildModel(arch=arch, activation='swish',reg=None,dropout=None)
        if leg == 0:
            m1.buildModel(arch=arch, activation=activation,reg=reg,dropout=dropout, summary = True)
        else:
            m1.buildModel(arch=arch, activation=activation,reg=reg,dropout=dropout, summary = False)
        m1.compileModel(opt=opt, lr=lr, loss=mean_error_type, metrics=['MAE','MSE'], beta_1=0.9995, beta_2=0.999,decay=decay,momentum=momentum) #
        #m1.compileModel(opt='SGD', lr=0.0000001, loss='MAE', metrics=['MAE','MSE'], beta_1=0.9995, beta_2=0.999)

    if contNN == True and changeingNNparam == False:
        m1.loadModel('drive/My Drive/4th Year Project/{}/{}_best_model.h5'.format(folder_name,folder_name),summary = False)

    if contNN == True and changeingNNparam == True: 
        m1.setWeights(m0.model)

    #m1.fitModel(df=df, cols=[x_cols,y_cols], epoch_no=1000, batch_size=int(len(df.index)), save_name='drive/My Drive/4th Year Project/{}/{}'.format(folder_name,new_file_version), callback=['mc'])
    m1.fitModel(df=df, cols=[x_cols,y_cols], epoch_no=epoch_no, batch_size=batch_size, save_name='drive/My Drive/4th Year Project/{}/{}'.format(folder_name,folder_name), callback=['mc'])
    m1.saveHist('drive/My Drive/4th Year Project/{}/Hist{}'.format(folder_name,folder_name))
    copyfile('drive/My Drive/4th Year Project/{}/{}_best_model.h5'.format(folder_name,folder_name), 'drive/My Drive/4th Year Project/{}/{}_prev_leg_best_model.h5'.format(folder_name,folder_name)) #saves best model from last completed leg in case of crashes
    return start_epoch, m1

def NN_results(grid_file, folder_name,starttime,start_epoch,last_leg=False):
    finishtime = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    m1=NNmodel('evo',['mass', 'age', 'feh', 'Y', 'MLT'], ['radius','Teff', 'delnu'], Teff_scaling=5000)
    m1.loadModel('drive/My Drive/4th Year Project/{}/{}_best_model.h5'.format(folder_name,folder_name), summary = False)
    m1.loadHist('drive/My Drive/4th Year Project/{}/Hist{}'.format(folder_name,folder_name),'pickle')
    final_epoch = m1.history['legs'][-1]['epoch_range'][1]
    #
    m1.plotHist(plot_MSE=False, this_train=False, savefile='drive/My Drive/4th Year Project/{}'.format(folder_name), trial_no=folder_name)
    m1.plotHist(plot_MSE=False, epochs=m1.history['legs'][-1]['epoch_range'],this_train=False) 
    
    last_loss = m1.lastLoss('mean_absolute_error')

    grid1=stellarGrid(grid_file)
    grid1.loadData()
    grid1.popIndex(['Unnamed: 0','star_mass','star_age','star_feh','initial_Yinit','initial_MLT','luminosity','effective_T','delta_nu'])
    grid1.initialData()
    if last_leg == False:
        m1.plotIsochrone(grid1, np.linspace(1,11,12), ['age','feh','Y','MLT'], [0, 0.28,1.9], widths=[['r',0.05],['a',0.1],['a',0.01],['a',0.1]],savefile='drive/My Drive/4th Year Project/{}'.format(folder_name), trial_no=folder_name,extended=False)
        print(last_loss)
        
    #m1.plotAll(grid1, savefile=folder_no, trial_no=trial_no)

    if last_leg == True:
        dexvals = m1.getDex(grid1)
        evaluation = m1.evalData(grid1, grid1.data['track_no'].unique().astype('int'))
        evaluation_loss = evaluation[1]
        runresults = [str(last_loss),str(evaluation_loss),str(dexvals['radius']),str(dexvals['Teff']),str(dexvals['delnu'])]

        sheets_results = ' '.join(runresults)
        m1.plot_Loss_Error_HR_Iso_in_one(grid=grid1, iso_ages=np.linspace(1,11,12), indices=['age','feh','Y','MLT'], isos=[0, 0.28,1.9], plot_MSE=False, epochs=None, widths=[['r',0.05],['a',0.1],['a',0.01],['a',0.1]], N=5000, one_per_track=False, track_no=100, savefile='drive/My Drive/4th Year Project/{}'.format(folder_name), trial_no=folder_name)
        print(sheets_results)
        strFrom = 'astroneuralnetworks@gmail.com'
        strTo = 'accelerationduetogravity9.81@gmail.com'
        password = "Elu2u2_!"

        #subject = "Neural Network run completed"
        subject = "Neural Network run completed"
        body = "Trial: {}<br><br>Started at: {}<br>Finished at: {}<br>Epoch range: {}-{}<br><br>Final learning loss: {}<br>Evaluation loss: {}<br>Radius dex: {}<br>Teff dex: {}<br>Delnu dex: {}".format(folder_name,starttime,finishtime,start_epoch,final_epoch,round(last_loss,5),round(evaluation_loss,5), round(dexvals['radius'],5),round(dexvals['Teff'],5),round(dexvals['delnu'],5))

        # Create the root message and fill in the from, to, and subject headers
        msgRoot = MIMEMultipart('related')
        msgRoot['Subject'] = subject
        msgRoot['From'] = strFrom
        msgRoot['To'] = strTo
        msgRoot.preamble = ''

        msgAlternative = MIMEMultipart('alternative')
        msgRoot.attach(msgAlternative)

        msgText = MIMEText('Plain Text message')
        msgAlternative.attach(msgText)

        msgText = MIMEText('<b>{}<br>Graphical Results</b><br><img src="cid:image1"><br>'.format(body), 'html')
        msgAlternative.attach(msgText)
        fp = open('drive/My Drive/4th Year Project/{}/Combo{}.png'.format(folder_name,folder_name), 'rb')
        msgImage = MIMEImage(fp.read())
        fp.close()
        msgImage.add_header('Content-ID', '<image1>')
        msgRoot.attach(msgImage)

        port = 465  # For SSL
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
            server.login(strFrom, password)
            server.sendmail(strFrom, strTo, msgRoot.as_string())
        return m1

def Get_Num_Legs(reg,lr,epoch_no,batch_size,opt,activation,dropout, momentum, decay, mean_error_type):
    leg_num = -1
    if reg != None:
        if type(reg) is list:
            if type(reg[0]) is list:
                if len(reg) > leg_num:
                    leg_num = len(reg)

    if type(lr) is list:
        if len(lr) > leg_num:
            leg_num = len(lr)
            
    if type(epoch_no) is list:
        if len(epoch_no) > leg_num:
            leg_num = len(epoch_no)
            
    if type(batch_size) is list:
        if len(batch_size) > leg_num:
            leg_num = len(batch_size)
            
    if type(opt) is list:
        if len(opt) > leg_num:
            leg_num = len(opt)
            
    if type(activation) is list:
        if len(activation) > leg_num:
            leg_num = len(activation)


    if type(dropout) is list:
        if len(dropout) > leg_num:
            leg_num = len(dropout)

    if type(momentum) is list:
        if len(momentum) > leg_num:
            leg_num = len(momentum)

    if type(decay) is list:
        if len(decay) > leg_num:
            leg_num = len(decay)
    
    if type(mean_error_type) is list:
        if len(mean_error_type) > leg_num:
            leg_num = len(mean_error_type)
          
    return leg_num

def Get_Num_Legs_Trained(folder_name,load_partially_trained_model):
    if load_partially_trained_model == True:
        m1=NNmodel('evo',['mass', 'age', 'feh', 'Y', 'MLT'], ['radius','Teff', 'delnu'], Teff_scaling=5000)
        m1.loadHist('drive/My Drive/4th Year Project/{}/Hist{}'.format(folder_name,folder_name),'pickle')
        return len(m1.history['legs'])
    else:
        return 0

        



    
    
