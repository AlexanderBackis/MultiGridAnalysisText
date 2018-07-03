import clustering as clu
import plot as pl

#clu.import_and_save() 



#hfhfshfd

#name_vec = ['20180628-124623_1', '20180628-125019_1', '20180628-130039_1']
#pl.plot_DeltaT_and_compare(name_vec)


bus_vec = [0,1,2]
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

thres=330
pl.plot_all_sides_3D(bus_vec,thres)