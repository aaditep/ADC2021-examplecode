import numpy as np
from keras_flops import get_flops
import time
import argparse
import pandas as pd
from datetime import datetime
import h5py
import math
import os
import pathlib
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, Layer, ReLU, LeakyReLU
from tensorflow.keras import backend as K
from func import mse_loss
from sklearn.metrics import roc_curve, auc


def autoencoder(to_save_m,m_save_path,save_plots,s_plots_path):
    

    signal_in='./signals/oh_trig/'
    bkg_filename = './signals/oh_trig/BKG_OH_TRIG_OH_dataset.h5'
    X_train,X_test,X_val,signal_data,signal_labels=read_bkg_and_signals(bkg_filename,signal_in)
    
    # add correct signal labels
    data_labels  = ['Ato4l_lepFilter_13TeV_output',
                     'hChToTauNu_13TeV_PU20_output',
                     'hToTauTau_13TeV_PU20_output',
                     'leptoquark_LOWMASS_lepFilter_13TeV_output',
                     'background_for_training_output']
    
    EPOCHS = 10
    BATCH_SIZE = 1024
    print("Training model")
    autoencoder,FLOPs=AE_setup_training(EPOCHS,BATCH_SIZE,X_train,X_test,X_val,signal_data,signal_labels)
    print("Model trained")
    if to_save_m=='y':
        #model_name = 'DENSE_AE_OH4_model_flops_test'
        #model_directory = './models/'
        print("Saving model..")
        autoencoder.save(m_save_path+"/model")
        #save_model(model_directory+model_name, autoencoder)
        print("Model save complete")
    
    #print('TF Profile: Total number of FLOPs =  {}'.format(get_flops(model_directory+model_name))
    print("Inference in progress")
    AE_OH_results,inference_t=predict_signal_bkg(signal_data,autoencoder,X_test)
    print("Inference complete")
    #reshape the results and reverse OH vectors with argmax to single type of object
    ground_truth,resh_type_results=OH_reverse_convert(AE_OH_results,signal_data,X_test)
    print("OH vectors reverse conversion complete")
    #find attribute multiplicities for signal  and background
    truth_attributes,results_attributes=find_attribute_multiplicities(ground_truth,resh_type_results)
    print("Attribute multipliciti sorting complete")
    #Dataframe results
    truth_dfs,results_dfs=dataframing(ground_truth,resh_type_results,truth_attributes,results_attributes)
    print("Dataframes created")
    #Plot results: mse loss, Roc curve, MET_pt attribute for signal and background
    print("Plotting results")
    plotting(truth_dfs,results_dfs,signal_labels,s_plots_path,save_plots,inference_t,FLOPs)
    
    
def read_bkg_and_signals(bkg_filename,signal_in):
    #Outputs training,test,validation and signal_data data
    #From assigned source/path
    # add correct path to signal files
    signals_file = [signal_in+'Ato4l_lepFilter_13TeV_input_OH_trig_OH_dataset.h5',
                signal_in+'hChToTauNu_13TeV_PU20_input_OH_trig_OH_dataset.h5',
                signal_in+'hToTauTau_13TeV_PU20_input_OH_trig_OH_dataset.h5',
                signal_in+'leptoquark_LOWMASS_lepFilter_13TeV_input_OH_trig_OH_dataset.h5'] 
    # add correct signal labels
    signal_labels = ['Ato4l_lepFilter_13TeV_dataset',
                     'hChToTauNu_13TeV_PU20_dataset',
                     'hToTauTau_13TeV_PU20_dataset',
                     'leptoquark_LOWMASS_lepFilter_13TeV_dataset']    
    with h5py.File(bkg_filename, 'r') as file:
        X_train = np.array(file['X_train'])
        X_test = np.array(file['X_test'])
        X_val = np.array(file['X_val'])
    signal_data = []
    for i, label in enumerate(signal_labels):
        with h5py.File(signals_file[i], 'r') as file:
            test_data = np.array(file['Data'])
        signal_data.append(test_data)
    
    return X_train,X_test,X_val,signal_data,signal_labels

def AE_setup_training(EPOCHS,BATCH_SIZE,X_train,X_test,X_val,signal_data,signal_labels):
    #Autoencoder(AE) model setup
    input_shape = 171#before 152 and before 57
    latent_dimension = 3
    num_nodes=[16,8]
    
    #encoder
    inputArray = Input(shape=(input_shape))
    x = Dense(num_nodes[0], use_bias=False)(inputArray)
    x = Activation('relu')(x)
    x = Dense(latent_dimension, use_bias=False)(x)
    encoder = Activation('relu')(x)
    
    #decoder
    x = Dense(num_nodes[0], use_bias=False)(encoder)
    x = Activation('relu')(x)
    
    decoder = Dense(input_shape)(x)
    #Create AE
    autoencoder = Model(inputs = inputArray, outputs=decoder)
    autoencoder.summary()
    
    #Compile AE
    autoencoder.compile(optimizer = keras.optimizers.Adam(), loss='mse')
    print("flops:")
    # Compute FLOPs
    FLOPs = get_flops(autoencoder, batch_size=1)
    print("keras-flops: Total number of FLOPs = {} ".format(FLOPs))
   
    
    #Treenimine
    #EPOCHS = 10
    #BATCH_SIZE = 1024
    history = autoencoder.fit(X_train, X_train, epochs = EPOCHS, batch_size = BATCH_SIZE,
                  validation_data=(X_val, X_val))
    return autoencoder,FLOPs
    
def predict_signal_bkg(signal_data,autoencoder,X_test):
    #Siia vist vaja panna inference aja võtmine ka
    AE_OH_results=[]
    for i in range(4):
        #All 4 signals predicted
        signal_prediction = autoencoder.predict(signal_data[i])
        AE_OH_results.append(np.array(signal_prediction))
    #Background prediction
    dummy=autoencoder.predict(X_test[:100])
    start = time.time()
    bkg_prediction = autoencoder.predict(X_test,batch_size=15000)
    end = time.time()
    inference_t=((end-start)/len(X_test))*10**6
    print(f'inference time per event: {inference_t} μs')
    AE_OH_results.append(np.array(bkg_prediction))
    
    return AE_OH_results,inference_t


def OH_reverse_convert(AE_OH_results,signal_data,X_test):
    resh_type_results=[]
    ground_truth=[]
    #Reshaping for autoencoder results
    for i in range(5):
        data=AE_OH_results[i]#take list  with bkg and signals  flattened output
        data=np.reshape(data,(len(data), 19,9))#reshape events by 19 objects
        id_oh=data[:,:,-5:]#take OH vector ids
        id_idmax=np.argmax(id_oh, axis=-1)#find the type
        id_idmax=np.reshape(id_idmax,(len(data),19,1))#reshape for concat
        event_wo_type=data[:,:,:4]#take events wo. type
        reshaped_data=np.concatenate([event_wo_type,id_idmax],axis=-1)#concat.
        resh_type_results.append(reshaped_data)#add to list
    #reshaping for ground truth in order to dataframe it correctly
    AE_OH_truth=signal_data
    AE_OH_truth.append(X_test)
    for i in range(5):
        data=AE_OH_truth[i]#take list  with bkg and signals  flattened output
        data=np.reshape(data,(len(data), 19,9))#reshape events by 19 objects
        id_oh=data[:,:,-5:]#take OH vector ids
        id_idmax=np.argmax(id_oh, axis=-1)#find the type
        id_idmax=np.reshape(id_idmax,(len(data),19,1))#reshape for concat
        event_wo_type=data[:,:,:4]#take events wo. type
        reshaped_data=np.concatenate([event_wo_type,id_idmax],axis=-1)#concat.
        ground_truth.append(reshaped_data)#add to list
    
    return ground_truth,resh_type_results



def find_attribute_multiplicities(ground_truth,resh_type_results):
    #function find multiplicities (JETS, LEPTONS, etc..) per event
    truth_attributes=[]
    results_attributes=[]
    attributes=[truth_attributes,results_attributes]
    
    # add correct signal labels
    data_labels  = ['Ato4l_lepFilter_13TeV_input',
                     'hChToTauNu_13TeV_PU20.h5_input',
                     'hToTauTau_13TeV_PU20.h5_input',
                     'leptoquark_LOWMASS_lepFilter_13TeV_input',
                     'background_for_training.h5_input']
    counter=0
    for n in range(2):
        if n==0:
            data=ground_truth
        else:
            data=resh_type_results
        for k, label in enumerate(data_labels):
            print("Finding lepton,jets etc.. multiplicities per event")
            counter += 1
            print(f'{counter}/10')
            MET=[]
            ELEKT=[]
            MUON=[]
            JETID=[]
            packed=[]
            for i in range(len(data[k])):
                ith_event=data[k][i][:,4]
                MET.append(np.count_nonzero(ith_event == 1))
                ELEKT.append(np.count_nonzero(ith_event == 2))
                MUON.append(np.count_nonzero(ith_event == 3))
                JETID.append(np.count_nonzero(ith_event == 4))

            packed.append(MET)
            packed.append(ELEKT)
            packed.append(MUON)
            packed.append(JETID)
            packed=np.asarray(packed)
            #print(packed)
            #print(packed.shape)
            attributes[n].append(packed)

    return attributes[0],attributes[1]
    
    
def dataframing(ground_truth,resh_type_results,truth_attributes,results_attributes):
    #Set up framing column names
    column_names=[]
    columns=['P_t','η','cos(φ)','sin(φ)','type']
    objects=['MET','e/γ_1','e/γ_2','e/γ_3','e/γ_4','μ_1','μ_2','μ_3','μ_4','jet_1','jet_2','jet_3',
                       'jet_4','jet_5','jet_6','jet_7','jet_8','jet_9','jet_10']
    #MET
    for i in range(5):#before 4
        column_names.append(objects[0]+"_"+columns[i])
    #electron/photon
    for i in range(4):
        [column_names.append(objects[i+1]+"_"+columns[j]) for j in range(5)]
    #muons      
    for i in range(4):
        [column_names.append(objects[i+5]+"_"+columns[j]) for j in range(5)]
    #jets
    for i in range(10):
        [column_names.append(objects[i+9]+"_"+columns[j]) for j in range(5)]
    #Multiplicity of MET,ELEKt etc
    [column_names.append(["MET","ELEKT","MUON","JETS"][i]) for i in range(4)]
    #concatenate results and their attributes
    conced_truth=[]
    conced_results=[]
    for i in range(5):
        #shape(N-events,cardinality(19),attributes(4))
        reshaped=ground_truth[i].reshape(len(ground_truth[i]),95)#19*5=95 before76=19*4
        conced=np.concatenate([reshaped,truth_attributes[i].T],axis=1)
        conced_truth.append(conced)
    for i in range(5):
        reshaped=resh_type_results[i].reshape(len(resh_type_results[i]),95)#
        conced=np.concatenate([reshaped,results_attributes[i].T],axis=1)
        conced_results.append(conced)
    #Turn data into dataframes    
    truth_dfs=[]
    d={}
    for i in range(5):
        d["df_"+str(i)]=pd.DataFrame(conced_truth[i])#76+4 attributes
        d["df_"+str(i)].columns=column_names
        truth_dfs.append(d["df_"+str(i)])
    results_dfs=[]
    for i in range(5):
        d["df_"+str(i)]=pd.DataFrame(conced_results[i])
        d["df_"+str(i)].columns=column_names
        results_dfs.append(d["df_"+str(i)]) 
    
    return truth_dfs,results_dfs   
    
def plotting(truth_dfs,results_dfs,signal_labels,s_plots_path,save_plots,inference_t,FLOPs):
    
    if save_plots == "y":
        date= datetime.now().strftime("_%d_%m")
        plot_name = input("Enter plots name(date is added automatically): ")
        plot_name = plot_name+date
        
        #save FLOPS and latency to txt file
        f= open(s_plots_path+plot_name+"_lat_FLOPs.txt","w+")
        f.write(f'Latency: {inference_t} microseconds \n')
        f.write(f'FLOPs: {FLOPs}')
        f.close()
    data_labels  = ['Ato4l_lepFilter_13TeV_input',
                 'hChToTauNu_13TeV_PU20.h5_input',
                 'hToTauTau_13TeV_PU20.h5_input',
                 'leptoquark_LOWMASS_lepFilter_13TeV_input',
                 'background_for_training.h5_input']
    
    
    #MSE loss plotting
    #compute loss value (true, predicted)##1 väljund ja plot roc nr 3.
    total_loss = []
    #BKG loss
    total_loss.append(mse_loss(truth_dfs[4], results_dfs[4]).numpy())
    #signals loss
    for i, signal_X in enumerate(truth_dfs[:4]):
        total_loss.append(mse_loss(signal_X, results_dfs[i]).numpy())
            
    #bin_size=np.linspace(0, 10000, 100)
    bin_size=100
    
    plt.figure(figsize=(10,8))
    for i, label in enumerate(signal_labels):
        plt.hist(total_loss[i], bins=bin_size, label=label, density = True, histtype='step', fill=False, linewidth=1.5)
    plt.yscale('log')
    plt.xlabel("Autoencoder Loss",fontsize=15)
    plt.ylabel("Probability (a.u.)",fontsize=15)
    plt.title('MSE loss',fontsize=20)
    plt.legend(loc='best',fontsize=15)
    if save_plots == "n":
        pass
    else:
        plt.savefig(s_plots_path+plot_name+"_MSE.png")
        
    #Roc Curve plotting
    labels = np.concatenate([['Background'], np.array(signal_labels)])
    target_background = np.zeros(total_loss[0].shape[0])
    plt.figure(figsize=(10,8))
    for i, label in enumerate(labels):
        if i == 0: continue # background events
        trueVal = np.concatenate((np.ones(total_loss[i].shape[0]), target_background)) # anomaly=1, bkg=0
        predVal_loss = np.concatenate((total_loss[i], total_loss[0]))
        fpr_loss, tpr_loss, threshold_loss = roc_curve(trueVal, predVal_loss)
        auc_loss = auc(fpr_loss, tpr_loss)
        plt.plot(fpr_loss, tpr_loss, "-", label='%s (auc = %.1f%%)'%(label,auc_loss*100.), linewidth=1.5)
        plt.semilogx()
        plt.semilogy()
        plt.ylabel("True Positive Rate",fontsize=15)
        plt.xlabel("False Positive Rate",fontsize=15)
        plt.legend(loc=4,fontsize=15)
        plt.grid(True)
        plt.tight_layout()
    plt.plot(np.linspace(0, 1),np.linspace(0, 1), '--', color='0.75')
    plt.axvline(0.00001, color='red', linestyle='dashed', linewidth=1) # threshold value for measuring anomaly detection efficiency
    plt.title("ROC AE",fontsize=20)
    if save_plots == "n":
        pass
    else:
        plt.savefig(s_plots_path+plot_name+"_ROC.png",bbox_inches="tight")
        
        
    
    #2d histogramm plotting MET P_t
    
    bins=np.linspace(0,250,num=150)
    fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(15,15))
    #signal plotting
    for ax,i in zip(axes.flat[0:],[0,1,2,3]):
        ax
        ax.set_title(signal_labels[i]+'. Events:{0}'.format(len(truth_dfs[i])),fontsize=14)
        ax.hist2d(truth_dfs[i]["MET_P_t"],results_dfs[i]["MET_P_t"],bins=bins,label=label,cmin=1)
        if i == 0 or i==2:
            ax.set_ylabel('Prediction',fontsize=20)
        if i == 2 or i ==3:   
            ax.set_xlabel('Ground truth',fontsize=20)
    if save_plots == "n":
        pass
    else:
        plt.savefig(s_plots_path+plot_name+"_signal_MET.png")
    #background plotting
    fig, ax = plt.subplots(figsize =(10, 7))
    matplotlib.pyplot.hist2d(truth_dfs[4]["MET_P_t"],results_dfs[4]["MET_P_t"],bins=bins,cmin=1)
    ax.set_title('Background dataset'+' Events:{0}'.format(len(truth_dfs[4])),fontsize=20)
    ax.set_ylabel('Prediction',fontsize=15)
    ax.set_xlabel('Ground truth',fontsize=15)
    if save_plots == "n":
        pass
    else:
        plt.savefig(s_plots_path+plot_name+"_bkg_MET.png")
     
    #2d histogramm first JET P_t
    bins=np.linspace(0,250,num=150)
    fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(15,15))
    #signal plotting
    for ax,i in zip(axes.flat[0:],[0,1,2,3]):
        ax
        ax.set_title(signal_labels[i]+'. Events:{0}'.format(len(truth_dfs[i])),fontsize=14)
        ax.hist2d(truth_dfs[i]['jet_1_P_t'],results_dfs[i]['jet_1_P_t'],bins=bins,label=label,cmin=1)
        if i == 0 or i==2:
            ax.set_ylabel('Prediction',fontsize=20)
        if i == 2 or i ==3:   
            ax.set_xlabel('Ground truth',fontsize=20)
    if save_plots == "n":
        pass
    else:
        plt.savefig(s_plots_path+plot_name+"_signal_1stjet.png")
    #background plotting
    fig, ax = plt.subplots(figsize =(10, 7))
    matplotlib.pyplot.hist2d(truth_dfs[4]['jet_1_P_t'],results_dfs[4]['jet_1_P_t'],bins=bins,cmin=1)
    ax.set_title('Background dataset 1st jet Pt'+' Events:{0}'.format(len(truth_dfs[4])),fontsize=20)
    ax.set_ylabel('Prediction',fontsize=15)
    ax.set_xlabel('Ground truth',fontsize=15)
    if save_plots == "n":
        pass
    else:
        plt.savefig(s_plots_path+plot_name+"_bkg_1stjet.png")
        
    #Plotting MET per event for ground truth and AE results
    bins = np.linspace(0, 10, 11)
    plt.figure(figsize=(10,5))
    plt.suptitle('MET per event for ground truth and results',fontsize=20)
    for n in range(2):
        if n == 0:
            data=truth_dfs
            string="Ground truth "
        else:
            data=results_dfs 
            string="AE results "
            
        plt.subplot(1, 2, n+1)
        for i, label in enumerate(data_labels):
            plt.hist(data[i]["MET"], bins=bins, label=label, density = True, histtype='step', fill=False, linewidth=1.5)
        plt.yscale('log')
        plt.xlabel("MET multiplicity",fontsize=15)
        if n == 0:
            plt.ylabel("Probability",fontsize=15)
            
        plt.title(string+'METs per event',fontsize=15)
        if n == 0:
            plt.legend(bbox_to_anchor=(0,-0.48), loc="lower left")
    if save_plots == "n":
        pass
    else:
        plt.savefig(s_plots_path+plot_name+"_MET_mult.png")
        
    #Plotting N leptons per event for ground truth and AE results    
    bins = np.linspace(0, 14, 15)
    plt.figure(figsize=(10,5))
    plt.suptitle('Leptons per event for ground truth and results',fontsize=20)
    for n in range(2):
        if n == 0:
            data=truth_dfs
            string="Ground truth "
        else:
            data=results_dfs 
            string="AE results "
            
        plt.subplot(1, 2, n+1)
        for i, label in enumerate(data_labels):
            plt.hist(data[i]["ELEKT"]+data[i]["MUON"], bins=bins, label=label, density = True, histtype='step', fill=False, linewidth=1.5)
        plt.yscale('log')
        plt.xlabel("Lepton multiplicity",fontsize=15)
        if n== 0:
            plt.ylabel("Probability",fontsize=15)
        plt.title(string+'Leptons per event',fontsize=15)
        if n == 0:
            plt.legend(bbox_to_anchor=(0,-0.48), loc="lower left")
    if save_plots == "n":
        pass
    else:
        plt.savefig(s_plots_path+plot_name+"_LEP_mult.png")       
    
    #Jets per event for gorund truth and AE results
    bins = np.linspace(0,14, 15)
    plt.figure(figsize=(10,5))
    plt.suptitle('Jets per event for ground truth and results',fontsize=20)
    for n in range(2):
        if n == 0:
            data=truth_dfs
            string="Ground truth "
        else:
            data=results_dfs 
            string="AE results "
            
        plt.subplot(1, 2, n+1)
        for i, label in enumerate(data_labels):
            plt.hist(data[i]["JETS"], bins=bins, label=label, density = True, histtype='step', fill=False, linewidth=1.5)
        plt.yscale('log')
        plt.xlabel("Jet multiplicity",fontsize=15)
        if n == 0:
            plt.ylabel("Probability",fontsize=15)
        plt.title(string+'Jets per event',fontsize=15)
        if n == 0:
            plt.legend(bbox_to_anchor=(0,-0.48), loc="lower left")
    if save_plots == "n":
        pass
    else:
        plt.savefig(s_plots_path+plot_name+"_JETS_mult.png")  
    
    #Plot MET vs cos(phi) per event for ground truth and AE results
    bin_size=100
    #bins = np.linspace(0,10, 11)
    plt.figure(figsize=(10,6))
    plt.suptitle('MET cos(phi) per event for ground truth and results',fontsize=20)
    for n in range(2):
        if n == 0:
            data=truth_dfs
            string="Ground truth "
        else:
            data=results_dfs 
            string="AE results "   
        plt.subplot(1, 2, n+1)
        for i, label in enumerate(data_labels):
            plt.hist(data[i]["MET_cos(φ)"],bins=bin_size, label=label, histtype='step', fill=False, linewidth=1.5,log=True)
        plt.yscale('log')
        plt.xlabel("Met cos(phi)",fontsize=15)
        if n== 0:
            plt.ylabel("EVENTS",fontsize=15)
        plt.title(string+'MET cos(phi) vs events',fontsize=15)
        if n == 0:
            plt.legend(bbox_to_anchor=(0,-0.4), loc="lower left")
    if save_plots == "n":
        pass
    else:
        plt.savefig(s_plots_path+plot_name+"_MET_cos(phi).png")  

    #Plot MET vs sin(phi) per event for ground truth and AE results
    bin_size=100
    #bins = np.linspace(0,10, 11)
    plt.figure(figsize=(10,6))
    plt.suptitle('MET sin(phi) per event for ground truth and results',fontsize=20)
    for n in range(2):
        if n == 0:
            data=truth_dfs
            string="Ground truth "
        else:
            data=results_dfs 
            string="AE results "   
        plt.subplot(1, 2, n+1)
        for i, label in enumerate(data_labels):
            plt.hist(data[i]["MET_sin(φ)"],bins=bin_size, label=label, histtype='step', fill=False, linewidth=1.5,log=True)
        plt.yscale('log')
        plt.xlabel("Met sin(phi)",fontsize=15)
        if n== 0:
            plt.ylabel("EVENTS",fontsize=15)
        plt.title(string+'MET sin(phi) vs events',fontsize=15)
        if n == 0:
            plt.legend(bbox_to_anchor=(0,-0.4), loc="lower left")
    if save_plots == "n":
        pass
    else:
        plt.savefig(s_plots_path+plot_name+"_MET_sin(phi).png")  

    #Plot MET vs PT per event for ground truth and AE results
    bin_size=100
    plt.figure(figsize=(10,6))
    plt.suptitle('MET Pt per event for ground truth and results',fontsize=20)
    for n in range(2):
        if n == 0:
            data=truth_dfs
            string="Ground truth "
        else:
            data=results_dfs 
            string="AE results "
        plt.subplot(1, 2, n+1)
        for i, label in enumerate(data_labels):
            plt.hist(data[i]["MET_P_t"],bins=bin_size, label=label, histtype='step', fill=False, linewidth=1.5,log=True)
        plt.yscale('log')
        plt.xlabel("Met Pt",fontsize=15)
        if n == 0:
            plt.ylabel("EVENTS",fontsize=15)
        plt.title(string+"MET Pt vs events",fontsize=15)
        if n == 0:
            plt.legend(bbox_to_anchor=(0,-0.4), loc="lower left")
    if save_plots == "n":
        pass
    else:
        plt.savefig(s_plots_path+plot_name+"_MET_Pt.png")
    
    
    
    
    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_model", type=str, default="n", help="Save model or not", choices=["y","n"])
    parser.add_argument("--m_save_path", type=str, default=None, help="Save model to path(with intended model name)")
    parser.add_argument("--save_plots", type=str, default="n", help="Save plots or not", choices=["y", "n"] )
    parser.add_argument("--s_plots_path", type=str, default=None, help="Save plots to path..")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = vars(parse_args())
    autoencoder(args['save_model'],args['m_save_path'],args['save_plots'],args['s_plots_path'])
