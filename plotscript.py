# fname = 'data/4plt_stabilized_stst_cylinderwake__NwtnitNone_time2.0_nu0.0006_mesh3_Nts50_dt0.04NV19468NY4NU4alphau0.0001__sigout'
fname = 'data/4plt_stst_cylinderwake__NwtnitNone_time2.0_nu0.0006_mesh3_Nts50_dt0.04NV19468NY4NU4alphau0.0001__sigout_unco'
import plot_output as plo
import optcont_main as ocm
jsf = ocm.load_json_dicts(fname)
plo.plot_optcont_json(jsf, fname=fname)
