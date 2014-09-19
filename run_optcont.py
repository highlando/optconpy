from optcont_main import optcon_nse

closed_loop = 0
stst_control = 0

outernwtnstps = 2

nwtn_adi_dict = dict(adi_max_steps=100,
                     adi_newZ_reltol=1e-7,
                     nwtn_max_steps=20,
                     nwtn_upd_reltol=4e-8,
                     nwtn_upd_abstol=1e-7,
                     verbose=True,
                     ms=[-5.0, -3.0, -2.0, -1.5, -1.3, -1.1, -1.0],
                     full_upd_norm_check=False,
                     check_lyap_res=False)

# curnwtnstpdict = {None: {'v': None,
#                          'mtxtb': None,
#                          'w': None}}
scaletest = 1e1
optcon_nse(N=12, Nts=5*scaletest, nu=0.5*1e-3, clearprvveldata=True,
           closed_loop=closed_loop, stst_control=stst_control,
           ini_vel_stokes=True, t0=0.0, tE=0.1*scaletest,
           outernwtnstps=outernwtnstps,
           nwtn_adi_dict=nwtn_adi_dict)
