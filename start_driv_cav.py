from optcont_main import optcon_nse

spec_tip_dict = dict(nwtn_adi_dict=dict(
                     adi_max_steps=200,
                     adi_newZ_reltol=1e-8,
                     nwtn_max_steps=16,
                     nwtn_upd_reltol=5e-8,
                     nwtn_upd_abstol=1e-7,
                     verbose=True,
                     full_upd_norm_check=False,
                     check_lyap_res=False
                     ),
                     compress_z=True,  # whether or not to compress Z
                     comprz_maxc=50,  # compression of the columns of Z by QR
                     comprz_thresh=5e-5,  # threshold for trunc of SVD
                     save_full_z=False  # whether to save the uncompressed Z
                     )

optcon_nse(N=25, Nts=40, nu=1e-2, t0=0.0, tE=1.0,
           spec_tip_dict=spec_tip_dict,
           stst_control=False,
           clearprvveldata=False,
           ini_vel_stokes=False,
           comp_unco_out=False,
           use_ric_ini_nu=None, alphau=1e-7)
