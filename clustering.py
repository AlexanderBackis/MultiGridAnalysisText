import os
import pandas as pd
import numpy as np


# =============================================================================
# Main Functions
# =============================================================================



def import_data():
    """ Imports raw data from sister folder 'Data/', stores it in a pandas 
        DataFrame and then returns it. If several files are present in the 
        'Data/'-folder, these are concatenated into one large DataFrame.
         
    Returns:
        df_tot (pd.DataFrame): DataFrame were each row is a datapoint, and each 
                               column is a different measured property 
                               (Trigger, HighTime, Time, Bus, Channel and ADC).
            
    """
    dirname = os.path.dirname(__file__)
    print(dirname)
    folder = os.path.join(dirname, '../Data/')
    files_in_folder = os.listdir(folder)
    df_tot = pd.DataFrame()
    for file in files_in_folder:
        if file[-4:] == '.csv':
            file_path = folder + file
            df = pd.read_csv(file_path, header=None, sep=', ', 
                             names=['Trigger', 'HighTime', 'Time', 'Bus', 
                                    'Channel', 'ADC'], engine='python')
            df = df.drop(df.index[0])
            df_tot = pd.concat([df_tot, df])
    
    df_tot['Trigger'] = df_tot['Trigger'].astype(int)
    df_tot['HighTime'] = df['HighTime'].astype(int)
    df_tot['Time'] = df_tot['Time'].astype(int)
    df_tot['Bus'] = df_tot['Bus'].astype(int)
    df_tot['Channel'] = df_tot['Channel'].astype(int)
    df_tot['ADC'] = df_tot['ADC'].astype(int)
    
    df_tot.reset_index(drop=True, inplace=True)
    print(df_tot)
    return df_tot

def cluster_data(df, bus):
    """ Clusters the data contained in the imported pd.DataFrame. Data points
        with the same timestamp are grouped together, the wire channel and
        grid channel with most collected charge are then used as 2D hit 
        position.
        
    Args:
        df (pd.DataFrame): DataFrame containing the imported data
        bus (int): The current bus analyzed (each bus acts as an independent
                   detector)
         
    Returns:
        df_clu (pd.DataFrame): DataFrame were each row is a candidate neutron
                               event, and each column is a different property 
                               of the candidate event (Time, ToF, wCh, gCh,
                               wADC, gADC, wM, gM)
            
    """
    rowNbr = 1
    df = df[(df.Bus == bus) | (df.Bus == -1)]
    size = df.shape[0]
    itr = df.iterrows()
    event_index = 0
    dict_clu = create_dict(size)
    row = next(itr)[1]
    ExternalTrigger = 0
    while rowNbr < size:
        timestamp = row.Time
        if row.Bus == -1:
            ExternalTrigger = row.Time
            row = next(itr)[1]
            rowNbr = rowNbr + 1
        else:
            wChTemp = [-1, 0]
            gChTemp = [-1, 0]
            wADC = 0
            wM = 0
            gADC = 0
            gM = 0
            while timestamp == row.Time and rowNbr < size:
                Channel = row.Channel
                if Channel < 80:
                    wADC = wADC + row.ADC
                    wM = wM + 1
                    if row.ADC > wChTemp[1]:
                        wChTemp[0] = Channel
                else:
                    gADC = gADC + row.ADC
                    gM = gM + 1
                    if row.ADC > gChTemp[1]:
                        gChTemp[0] = Channel
                
                row = next(itr)[1]
                rowNbr = rowNbr + 1
            
            wCh = wChTemp[0] 
            gCh = gChTemp[0]
                        
            dict_clu['ToF'][event_index] = timestamp - ExternalTrigger
            dict_clu['Time'][event_index] = timestamp
            dict_clu['wCh'][event_index] = wCh
            dict_clu['wADC'][event_index] = wADC
            dict_clu['wM'][event_index] = wM
            dict_clu['gCh'][event_index] = gCh
            dict_clu['gADC'][event_index] = gADC
            dict_clu['gM'][event_index] = gM
            
            event_index = event_index + 1
    
        if rowNbr % 100000 == 0:
            print('Progress: ' + str(round(((rowNbr)/size),2)*100) + ' %')
            print('Number of events: ' + str(event_index) + '\n')
    
    df_clu = pd.DataFrame(dict_clu)
    df_clu = df_clu.drop(range(event_index, size, 1))
    return df_clu

def save_clusters(df_clu, bus):
    """ Saves clustered data into a '.csv'-file in sister folder 'Clusters/'.
    
    Args:
        df_clu (pd.DataFrame): DataFrame containing the clustered data
        bus (int): The current bus analyzed (each bus acts as an independent
                   detector)
         
    Returns:
        df_tot (pd.DataFrame): DataFrame were each row is a datapoint, and each 
                               column is a different measured property 
                               (Trigger, HighTime, Time, Bus, Channel and ADC).
            
    """
    dirname = os.path.dirname(__file__)
    folder = os.path.join(dirname, '../Clusters/')
    file_path = folder + 'Bus_' + str(bus) + '.csv'
    df_clu.to_csv(file_path, sep=',', encoding='utf-8', index=False)
    

def import_and_save():
    """ Imports data, clusters it, and saves the clusters to file.
         
    Yields:
        Saves the clusters to file
            
    """
    df = import_data()
    bus_vec = np.array(range(0,3))
    for bus in bus_vec:
        df_clu = cluster_data(df, bus)  
        save_clusters(df_clu, bus)
        




# =============================================================================
# Helper Functions
# =============================================================================

def get_path():
    dirname = os.path.dirname(__file__)
    folder = os.path.join(dirname, '../Plot/')
    return folder
                                       
def create_dict(size):
    ToF = np.empty([size],dtype=int)
    Time = np.empty([size],dtype=int)
    wCh = np.empty([size],dtype=int)
    wADC = np.empty([size],dtype=int)
    wM = np.empty([size],dtype=int)
    gCh = np.empty([size],dtype=int)
    gADC = np.empty([size],dtype=int)
    gM = np.empty([size],dtype=int)
    return {'Time': Time, 'ToF': ToF, 'wCh': wCh, 'gCh': gCh, 'wADC': wADC, 
            'gADC': gADC, 'wM': wM, 'gM': gM}