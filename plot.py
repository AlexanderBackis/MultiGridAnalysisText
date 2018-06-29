import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm




# =============================================================================
# Plot 2D Pulse Height Spectrum, Channel VS Charge
# =============================================================================

def plot_PHS(df, bus, fig):
    df_red = df[df.Bus == bus]
    plt.subplot(1,3,bus+1)
    print(df_red.Channel)
    print(df_red.ADC)
    plt.hist2d(df_red.Channel, df_red.ADC, bins=[10, 120], norm=LogNorm(), 
               range=[[100, 110], [0, 4400]], vmin=1, vmax=10000)
    plt.ylabel("Charge [ADC channels]")
    plt.xlabel("Channel [a.u.]")
    plt.colorbar()
    name = 'Bus ' + str(bus)
    plt.title(name)
    
def plot_PHS_buses(df):
    bus_vec = np.array(range(0,3))
    fig = plt.figure()
    fig.suptitle('2D-Histogram of Channel vs Charge',x=0.5,
                 y=1)
    fig.set_figheight(4)
    fig.set_figwidth(14)
    for bus in bus_vec:
        plot_PHS(df, bus, fig)
    name = '2D-Histogram of Channel vs Charge all buses'
    plt.tight_layout()
    plt.show()
    plot_path = get_path() + name  + '.pdf'
    fig.savefig(plot_path)




# =============================================================================
# Plot 2D Histogram of Hit Position
# =============================================================================

def plot_2D_hit(bus, fig):
    df_clu = load_clusters(bus)
    df_clu_red = df_clu[(df_clu.wCh != -1) & (df_clu.gCh != -1)]
    plt.subplot(1,3,bus+1)
    plt.hist2d(df_clu_red.wCh, df_clu_red.gCh, bins=[80, 40], 
               range=[[0,80],[80,120]], norm=LogNorm(), vmin=1, vmax=10000)
    plt.xlabel("Wire [Channel number]")
    plt.ylabel("Grid [Channel number]")
    plt.colorbar()
    name = 'Bus ' + str(bus)
    plt.title(name)
    
def plot_2D_hit_buses():
    bus_vec = np.array(range(0,3))
    fig = plt.figure()
    fig.suptitle('2D-Histogram of hit position',x=0.5, y=1)
    fig.set_figheight(4)
    fig.set_figwidth(14)
    for bus in bus_vec:
        plot_2D_hit(bus, fig)
    name = '2D-Histogram of hit position, individual buses'
    plt.tight_layout()
    plt.show()
    plot_path = get_path() + name  + '.pdf'
    fig.savefig(plot_path)
    
    
    
    
# =============================================================================
# Plot Histogram of Charge fraction
# =============================================================================

def plot_charge_frac(bus, fig):
    df_clu = load_clusters(bus)
    df_clu_red = df_clu[(df_clu.wCh != -1) & (df_clu.gCh != -1)]
    plt.subplot(1,3,bus+1)
    print(df_clu_red[df_clu_red['gADC'] > df_clu_red['wADC']])
    plt.hist(np.divide(df_clu_red.gADC, df_clu_red.wADC), bins=150, log=True, 
             range=[0,6])
    plt.xlabel("gADC / wADC")
    plt.ylabel("Counts")
    plt.ylim([1,1000000])
    name = 'Bus ' + str(bus)
    plt.title(name)
    
def plot_charge_frac_buses():
    bus_vec = np.array(range(0,3))
    fig = plt.figure()
    fig.suptitle('Histogram of charge fraction',x=0.5, y=1)
    fig.set_figheight(4)
    fig.set_figwidth(14)
    for bus in bus_vec:
        plot_charge_frac(bus, fig)
    name = 'Histogram of charge fraction, individual buses'
    plt.tight_layout()
    plt.show()
    plot_path = get_path() + name  + '.pdf'
    fig.savefig(plot_path)
   
    
    
    
# =============================================================================
# Plot Histogram of Charge
# =============================================================================
    
def plot_charge(bus, fig):
    df_clu = load_clusters(bus)
    df_clu_red = df_clu[(df_clu.wCh != -1) & (df_clu.gCh != -1)]
    plt.subplot(1,3,bus+1)
    plt.hist(df_clu_red.wADC, bins=50, log = True, range=[0,15000], 
             alpha = 0.8, label = 'Wires')
    plt.hist(df_clu_red.gADC, bins=50, log = True, range=[0,15000], 
             alpha = 0.8, label = 'Grids')
    plt.legend(loc='upper right')
    plt.xlabel("Charge [ADC channels]")
    plt.ylabel("Counts")
    plt.ylim([1,1000000])
    name = 'Bus ' + str(bus)
    plt.title(name)
    
def plot_charge_buses():
    bus_vec = np.array(range(0,3))
    fig = plt.figure()
    fig.suptitle('Histogram of charge',x=0.5, y=1)
    fig.set_figheight(4)
    fig.set_figwidth(14)
    for bus in bus_vec:
        plot_charge(bus, fig)
    name = 'Histogram of charge, individual buses'
    plt.tight_layout()
    plt.show()
    plot_path = get_path() + name  + '.pdf'
    fig.savefig(plot_path)
    
 
    
    
# =============================================================================
# Plot time difference of consequtive external triggers
# =============================================================================      
    
def plot_trigger_difference(df):
    fig = plt.figure()
    df_red = df[(df.Bus == -1)]
    print(df_red)
    print(df_red.index[0])
    size = df_red.shape[0]
    print(size)
    print(df_red.index[size-1])
    df_red_1 = df_red.drop(df_red.index[size-1])
    df_red_2 = df_red.drop(df_red.index[0])
    df_red_1.reset_index(drop=True, inplace=True)
    df_red_2.reset_index(drop=True, inplace=True)

    plt.plot(range(0,df_red_1.shape[0]),df_red_2['Time'] - df_red_1['Time'], 
             '.-')
    plt.xlabel('External trigger')
    plt.xlim([0,500])
    plt.ylabel('$\Delta$T [TDC channels]')
    plt.yscale('log')
    name = 'Time difference $\Delta$T between external triggers'
    plt.title(name)
    plt.show()
    plot_path = get_path() + name  + '.pdf'
    fig.savefig(plot_path)
    
    
    


# =============================================================================
# Plot Histogram of Multiplicity
# =============================================================================    
    
def plot_multiplicity(bus, fig):
    df_clu = load_clusters(bus)
    plt.subplot(1,3,bus+1)
    plt.hist(df_clu.wM, bins=25, log = True, range=[0,25],
             alpha = 0.8, label = 'Wires')
    plt.hist(df_clu.gM, bins=25, log = True, range=[0,25],
             alpha = 0.8, label = 'Grids')
    plt.legend(loc='upper right')
    plt.xlabel("Multiplicity")
    plt.ylabel("Counts")
    plt.ylim([1,2000000])
    name = 'Bus ' + str(bus)
    plt.title(name)
    
def plot_multiplicity_buses():
    bus_vec = np.array(range(0,3))
    fig = plt.figure()
    fig.suptitle('Histogram of multiplicity',x=0.5, y=1)
    fig.set_figheight(4)
    fig.set_figwidth(14)
    for bus in bus_vec:
        plot_multiplicity(bus, fig)
    name = 'Histogram of multiplicity, individual buses'
    plt.tight_layout()
    plt.show()
    plot_path = get_path() + name  + '.pdf'
    fig.savefig(plot_path)



    
# =============================================================================
# Plot 2D Histogram of Multiplicity
# =============================================================================       
    
#def plot_2D_multiplicity(bus, fig):
#    df_clu = load_clusters(bus)
#    plt.subplot(1,3,bus+1)
#    plt.hist2d(df_clu.wM, df_clu.gM, bins=[8, 8], range=[[0,8],[0,8]],
#               norm=LogNorm(), vmin=1, vmax=1000000)
#    plt.xlabel("Wire Multiplicity")
#    plt.ylabel("Grid Multiplicity")
#    plt.colorbar()
#    name = 'Bus ' + str(bus)
#    plt.title(name)
    
def plot_2D_multiplicity(bus, fig):
    df_clu = load_clusters(bus)
    plt.subplot(1,3,bus+1)
    hist, xbins, ybins, im = plt.hist2d(df_clu.wM, df_clu.gM, bins=[8, 8], 
                                        range=[[0,8],[0,8]],
                                       norm=LogNorm(), vmin=1, vmax=1000000)
    tot = df_clu.shape[0]
    for i in range(len(ybins)-1):
        for j in range(len(xbins)-1):
            if hist[j,i] > 0:
                plt.text(xbins[j]+0.5,ybins[i]+0.5, 
                         str(format(100*(round((hist[j,i]/tot),3)),'.1f')) + 
                         "%", color="r", ha="center", va="center", 
                         fontweight="bold", fontsize=8.5)
    
    plt.xticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5],['0','1','2','3','4','5','6',
               '7'])
    plt.yticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5],['0','1','2','3','4','5','6',
               '7'])
    plt.xlabel("Wire Multiplicity")
    plt.ylabel("Grid Multiplicity")
    plt.colorbar()
    plt.tight_layout()
    name = 'Bus ' + str(bus)
    plt.title(name)

def plot_2D_multiplicity_buses():
    bus_vec = np.array(range(0,3))
    fig = plt.figure()
    fig.suptitle('2D Histogram of multiplicity within a time cluster', x=0.5, 
                 y=1)
    fig.set_figheight(4)
    fig.set_figwidth(14)
    for bus in bus_vec:
        plot_2D_multiplicity(bus, fig)
    name = '2D Histogram of multiplicity, individual buses'
    
    plt.tight_layout()
    plt.show()
    plot_path = get_path() + name  + '.pdf'
    fig.savefig(plot_path)
   
# =============================================================================
# Plot time difference between events
# =============================================================================

def plot_DeltaT_events(bus, fig):
    df_clu = load_clusters(bus)
    size = df_clu.shape[0]
    df_red_1 = df_clu.drop(df_clu.index[size-1])
    df_red_2 = df_clu.drop(df_clu.index[0])
    df_red_1.reset_index(drop=True, inplace=True)
    df_red_2.reset_index(drop=True, inplace=True)
    print(df_red_1)
    print(df_red_2)
    plt.subplot(1,3,bus+1)
    plt.hist((df_red_2['Time'] - df_red_1['Time']), bins=200, 
             range=[0, 4000])
    plt.xlabel("$\Delta$T [TDC channels]")
    plt.ylabel("Counts")
    plt.ylim([1,21000])
    name = 'Bus ' + str(bus)
    plt.title(name)

def plot_DeltaT_events_buses():
    bus_vec = np.array(range(0,3))
    fig = plt.figure()
    fig.suptitle('Histogram of time $\Delta$T between events', x=0.5, y=1.0)
    fig.set_figheight(4)
    fig.set_figwidth(14)
    for bus in bus_vec:
        plot_DeltaT_events(bus, fig)
    name = 'Histogram of time $\Delta$T  difference between events'
    plt.tight_layout()
    plt.show()
    plot_path = get_path() + name  + '.pdf'
    fig.savefig(plot_path)


# =============================================================================
# Plot time difference between events compare 44kHz and 75kHz
# =============================================================================

def plot_DeltaT_events_Compare44and75(bus, fig):
    print('Bus: ' + str(bus))
    dirname = os.path.dirname(__file__)
    folder = os.path.join(dirname, '../Clusters44kHz/')
    file_path = folder + 'Bus_' + str(bus) + '.csv'
    df_clu = load_clusters_from_file_path(bus, file_path)
    size = df_clu.shape[0]
    df_red_1 = df_clu.drop(df_clu.index[size-1])
    df_red_2 = df_clu.drop(df_clu.index[0])
    df_red_1.reset_index(drop=True, inplace=True)
    df_red_2.reset_index(drop=True, inplace=True)
    ax = plt.subplot(1,3,bus+1)
    y, x, _ = plt.hist((df_red_2['Time'] - df_red_1['Time']), bins=200, 
             range=[0, 4000], alpha = 0.6, label = '44 kHz')
    
    
    
    
    max_idx = np.where(y == np.amax(y))[0][0]
    print('44kHz')
    print('Maximum at ' + str(x[max_idx]))    
    print('Mean at ' + str(np.sum(x[:-1]*y)/np.sum(y)))
    text = ('44kHz \n' + 'Max at: ' + str(x[max_idx]) + ' TDC ch.' + '\n' + 
            'Mean: ' + str(round(np.sum(x[:-1]*y)/np.sum(y)))  + ' TDC ch.')
    plt.text(0.55, 0.7, text, ha='left', va='center', transform=ax.transAxes)
    
    
    
    folder = os.path.join(dirname, '../Clusters75kHz/')
    file_path = folder + 'Bus_' + str(bus) + '.csv'
    df_clu = load_clusters_from_file_path(bus, file_path)
    size = df_clu.shape[0]
    df_red_1 = df_clu.drop(df_clu.index[size-1])
    df_red_2 = df_clu.drop(df_clu.index[0])
    df_red_1.reset_index(drop=True, inplace=True)
    df_red_2.reset_index(drop=True, inplace=True)
    y, x, _ = plt.hist((df_red_2['Time'] - df_red_1['Time']), bins=200, 
                 range=[0, 4000], alpha = 0.6, label = '75 kHz')
    
    max_idx = np.where(y == np.amax(y))[0][0]
    print('75kHz')
    print('Maximum at ' + str(x[max_idx]))    
    print('Mean at ' + str(np.sum(x[:-1]*y)/np.sum(y)))
    text = ('75kHz \n' + 'Max at: ' + str(x[max_idx]) + ' TDC ch.' + '\n' + 
            'Mean: ' + str(round(np.sum(x[:-1]*y)/np.sum(y)))  + ' TDC ch.')
    plt.text(0.55, 0.5, text, ha='left', va='center', transform=ax.transAxes)
    

    plt.legend(loc='upper right')
    plt.xlabel("$\Delta$T [TDC channels]")
    plt.ylabel("Counts")
    plt.ylim([1,22000])
    name = 'Bus ' + str(bus)
    plt.title(name)

def plot_DeltaT_events_Compare44and75_buses():
    bus_vec = np.array(range(0,3))
    fig = plt.figure()
    fig.suptitle('Histogram of time $\Delta$T between events', x=0.5, y=1.0)
    fig.set_figheight(4)
    fig.set_figwidth(14)
    for bus in bus_vec:
        plot_DeltaT_events_Compare44and75(bus, fig)
    name = 'Histogram of time difference between events, 44kHz and 75kHz'
    plt.tight_layout()
    plt.show()
    plot_path = get_path() + name  + '.pdf'
    fig.savefig(plot_path)
    
    
# =============================================================================
# Plot time difference between events from several different data sets
# =============================================================================

def plot_DeltaT_and_compare_bus(df_clu, bus, fig, name):
    plt.subplot(1,3,bus+1)
    size = df_clu.shape[0]
    df_red_1 = df_clu.drop(df_clu.index[size-1])
    df_red_2 = df_clu.drop(df_clu.index[0])
    df_red_1.reset_index(drop=True, inplace=True)
    df_red_2.reset_index(drop=True, inplace=True)
    y, x, _ = plt.hist((df_red_2['Time'] - df_red_1['Time']), 
                       bins=300, range=[0, 6000], alpha = 0.6, label = name)
    calculate_frequency(x,y)
    plt.legend(loc='upper right')
    plt.xlabel("$\Delta$T [$\mu$s]")
    plt.ylabel("Counts")
    plt.ylim([1,35000])
    #plt.xlim([0,5000])
    loc = np.arange(0, 7000, step=1000)
    plt.xticks(loc, loc*0.0625) 

    
    name = 'Bus ' + str(bus)
    plt.title(name)
    
def plot_DeltaT_and_compare(name_vec):
    bus_vec = np.array(range(0,3))
    fig = plt.figure()
    fig.suptitle('Histogram of time $\Delta$T between events', x=0.5, y=1.0)
    fig.set_figheight(4)
    fig.set_figwidth(14)
    for name in name_vec:
        folder = get_clusters_folder_path(name)
        print(name)
        for bus in bus_vec:
            print(bus)
            file_path = folder + 'Bus_' + str(bus) + '.csv'
            df_clu = load_clusters_from_file_path(bus, file_path)
            plot_DeltaT_and_compare_bus(df_clu, bus, fig, name)
    name = 'Histogram of time difference between events, ILL data 2018_06_28'
    plt.tight_layout()
    plt.show()
    plot_path = get_path() + name  + '.pdf'
    fig.savefig(plot_path)       
        

def calculate_frequency(x,y):
    mean = np.sum(x[:-1]*y)/np.sum(y)
    print(mean * 0.0625)
    freq = 1 / (mean * 62.5 * 10 ** (-9))
    sd = np.sqrt((np.sum(y*np.power((x[:-1]-mean),2)))/(np.sum(y)))
    print('SD: ' + str(sd*0.0625))
    print('Frequency: ' + str(freq) + 'Hz')
    freqUpper = 1 / ((mean-sd) * 62.5 * 10 ** (-9))
    freqLower = 1 / ((mean+sd) * 62.5 * 10 ** (-9))
    print('Plus: ' + str(freqUpper - freq))
    print('Minus: ' + str(freq- freqLower))


  
    
    
    
    
    
# =============================================================================
# Helper Functions
# =============================================================================    

def get_clusters_folder_path(name):
    dirname = os.path.dirname(__file__)
    folder = os.path.join(dirname, '../Clusters/' + name + '/')
    return folder
    
    
def get_path():
    dirname = os.path.dirname(__file__)
    folder = os.path.join(dirname, '../Plot/')
    return folder

def load_clusters(bus):
    dirname = os.path.dirname(__file__)
    folder = os.path.join(dirname, '../Clusters/')
    file_path = folder + 'Bus_' + str(bus) + '.csv'
    df_clu = pd.read_csv(file_path, header=None, sep=',', 
                     names=['Time', 'ToF', 'wCh', 'gCh', 'wADC', 
                            'gADC', 'wM', 'gM'], engine='python')
    df_clu = df_clu.drop(df_clu.index[0])  
    df_clu.reset_index(drop=True, inplace=True)
    
    df_clu['Time'] = df_clu['Time'].astype(int)
    df_clu['ToF'] = df_clu['ToF'].astype(int)
    df_clu['wCh'] = df_clu['wCh'].astype(int)
    df_clu['gCh'] = df_clu['gCh'].astype(int)
    df_clu['wADC'] = df_clu['wADC'].astype(int)
    df_clu['gADC'] = df_clu['gADC'].astype(int)
    df_clu['wM'] = df_clu['wM'].astype(int)
    df_clu['gM'] = df_clu['gM'].astype(int)
    
    return df_clu

def load_clusters_from_file_path(bus, file_path):
    df_clu = pd.read_csv(file_path, header=None, sep=',', 
                     names=['Time', 'ToF', 'wCh', 'gCh', 'wADC', 
                            'gADC', 'wM', 'gM'], engine='python')
    df_clu = df_clu.drop(df_clu.index[0])  
    df_clu.reset_index(drop=True, inplace=True)
    
    df_clu['Time'] = df_clu['Time'].astype(int)
    df_clu['ToF'] = df_clu['ToF'].astype(int)
    df_clu['wCh'] = df_clu['wCh'].astype(int)
    df_clu['gCh'] = df_clu['gCh'].astype(int)
    df_clu['wADC'] = df_clu['wADC'].astype(int)
    df_clu['gADC'] = df_clu['gADC'].astype(int)
    df_clu['wM'] = df_clu['wM'].astype(int)
    df_clu['gM'] = df_clu['gM'].astype(int)
    
    return df_clu





