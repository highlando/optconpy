from optcont_main import optcon_nse
spec_tip_dict = dict(nwtn_adi_dict=dict(
                     adi_max_steps=199,
                     adi_newZ_reltol=1e-8,
                     nwtn_max_steps=16,
                     nwtn_upd_reltol=5e-8,
                     nwtn_upd_abstol=1e-7,
                     verbose=True,
                     full_upd_norm_check=False,
                     check_lyap_res=False
                     ),
                     compress_z=False,  # whether or not to compress Z
                     comprz_maxc=100,  # compression of the columns of Z by QR
                     comprz_thresh=5e-5,  # threshold for trunc of SVD
                     save_full_z=False  # whether to save the uncompressed Z
                     )

probsetupdict = dict(problemname='cylinderwake', N=3,
                     spec_tip_dict=spec_tip_dict,
                     clearprvveldata=True,
                     t0=0.0, tE=2.0, Nts=50, stst_control=True,
                     ini_vel_stokes=True, alphau=1e-4)

# for nu=1e-3, i.e. Re = 150, the sys is stable
probsetupdict.update(dict(nu=1e-3, comp_unco_out=False,
                     use_ric_ini_nu=None))
optcon_nse(**probsetupdict)


# for nu=8e-4 (Re=187.5) we need stabilizing initial guess
probsetupdict.update(dict(nu=8e-4, comp_unco_out=False,
                     use_ric_ini_nu=1e-3))
optcon_nse(**probsetupdict)

# for nu=8e-4 (Re=187.5) we need stabilizing initial guess
probsetupdict.update(dict(nu=6e-4, comp_unco_out=False,
                     use_ric_ini_nu=8e-4))
optcon_nse(**probsetupdict)

# compute the uncontrolled solution
optcon_nse(problemname='cylinderwake', N=3, nu=6e-4,
           spec_tip_dict=spec_tip_dict,
           clearprvveldata=True,
           t0=0.0, tE=2.0, Nts=50, stst_control=True,
           comp_unco_out=True,
           ini_vel_stokes=True, use_ric_ini_nu=8e-4, alphau=1e-4)
