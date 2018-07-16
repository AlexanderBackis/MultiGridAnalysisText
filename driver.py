import clustering as clu
import plot as pl

ADCthres = 0
countThres = 0
switch=True
#
#clu.import_and_save(ADCthres, s)
#pl.plot_2D_hit_buses(ADCthres)
#pl.plot_2D_multiplicity_buses(ADCthres) 
##
#bus_vec = [0,1,2]
#pl.plot_all_sides_3D(bus_vec, 300)
#pl.plot_all_sides(bus_vec)
#
df = clu.import_data(0, switch)
#pl.plot_PHS_3D_surface(df)
#pl.plot_PHS_buses(df)

#pl.plot_wADC_vs_gADC_buses()



bus = 1

pl.plot_PHS_different_thicknesses(df, bus)

pl.plot_2D_hist_single_bus(df, bus)

#pl.plot_3D_new(df, bus)
#
ChVecW1 = [0, 4, 8, 12, 16]
ChVecW2 = [20, 24, 28, 32, 36]
ChVecW3 = [40, 44, 48, 52, 56]
ChVecW4 = [60, 64, 68, 72, 76]
ChVecVecW =  [ChVecW1, ChVecW2, ChVecW3, ChVecW4]

ChVecG1 =  [100, 95, 90, 85, 80]
ChVecG2 =   [100, 105, 110, 115, 120]
ChVecVecG =  [ChVecG1, ChVecG2]
yLimW = [0,500]
yLimG = [0,3000]


for vec in ChVecVecW:
    pl.plot_PHS_several_channels(df, bus, vec, yLimW)

for vec in ChVecVecG:
    pl.plot_PHS_several_channels(df, bus, vec, yLimG)

 

#name_vec = ['20180628-124623_1', '20180628-125019_1', '20180628-130039_1']
#pl.plot_DeltaT_and_compare(name_vec)


#bus_vec = [2,1,0]
#pl.plot_2D_side_1(bus_vec)
#pl.plot_2D_side_2(bus_vec)
#pl.plot_2D_side_3(bus_vec)                                           

#pl.plot_all_sides(bus_vec)

#df = clu.import_data()
#pl.plot_PHS_buses(df)

#name_vec = ['44kHz', '75kHz']
#pl.plot_DeltaT_and_compare(name_vec)

#pl.plot_2D_hit_buses()
#pl.plot_2D_multiplicity_buses()
#pl.plot_charge_frac_buses()
#pl.plot_charge_buses()
#pl.plot_DeltaT_events_buses()
#pl.plot_DeltaT_events_Compare44and75_buses()


#pl.plot_3d_surfaces()
#bus_vec = [2,1,0]
#thres=0
#pl.plot_all_sides_3D(bus_vec)
#pl.plot_all_sides(bus_vec)
#pl.plot_2D_hit_buses()
#pl.plot_2D_multiplicity_buses()



