from dolfin import *
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import krypy.linsys
from scipy.linalg import qr
from scipy.io import loadmat, savemat

import dolfin_to_nparrays as dtn

###
# solve M\dot v + K(v) -B'p = fv
#                 Bv      = fpbc
###

def mass_fem_ip(q1,q2,M):
    """M^-1 inner product

    """
    return np.dot(q1.T.conj(),krypy.linsys.cg(M,q2,tol=1e-12)['xk'])

def smamin_fem_ip(qqpq1,qqpq2,Mv,Mp,Nv,Npc):
    """ M^-1 ip for the extended system

    """
    return mass_fem_ip(qqpq1[:Nv,],qqpq2[:Nv,],Mv) + \
            mass_fem_ip(qqpq1[Nv:-Npc,],qqpq2[Nv:-Npc,],Mp) + \
            mass_fem_ip(qqpq1[-Npc:,],qqpq2[-Npc:,],Mp) 



def halfexp_euler_smarminex(MSme,BSme,MP,FvbcSme,FpbcSme,B2BoolInv,PrP,TsP,vp_init,qqpq_init=None):
    """halfexplicit euler for the NSE in index 1 formulation
    
    """

    N, Pdof = PrP.N, PrP.Pdof
    Nts, t0, tE, dt, Nv, Np = init_time_stepping(PrP,TsP)
    tcur = t0

    Npc = Np-1

    # remove the p - freedom
    if Pdof == 0:
        BSme  = BSme[1:,:]
        FpbcSmeC = FpbcSme[1:,]
        MPc = MP[1:,:][:,1:]
    else:
        BSme  = sps.vstack([BSme[:Pdof,:],BSme[Pdof+1:,:]])
        raise Warning('TODO: Implement this')

    B1Sme = BSme[:,:Nv-(Np-1)]
    B2Sme = BSme[:,Nv-(Np-1):]

    M1Sme = MSme[:,:Nv-(Np-1)]
    M2Sme = MSme[:,Nv-(Np-1):]
    
    #### The matrix to be solved in every time step
    #
    #       1/dt*M11    M12  -B2'  0        q1
    #       1/dt*M21    M22  -B1'  0        tq2
    #       1/dt*B1     B2   0     0   *    p     = rhs
    #            B1     0    0     B2       q2  
    #                                    
    # cf. preprint
    #

    MFac = 1 
    ## Weights for the 'conti' eqns to balance the residuals
    WC = 0.5
    WCD = 0.5
    PFac = 1#dt/WCD
    PFacI = 1#WCD/dt

    IterA1 = MFac * sps.hstack([sps.hstack([1.0/dt*M1Sme, M2Sme]), 
        -PFacI*BSme.T])
    IterA2 = WCD * sps.hstack([sps.hstack([1.0/dt*B1Sme, B2Sme]),
        sps.csr_matrix((Np-1, Np-1))])
    IterASp = sps.vstack([IterA1,IterA2])

    IterA3 = WC * sps.hstack([sps.hstack([B1Sme,sps.csr_matrix((Np-1,2*(Np-1)))]),
        B2Sme])

    IterA = sps.vstack([
        sps.hstack([IterASp, sps.csr_matrix((Nv+Np-1,Np-1))]),
                IterA3])

    ## Preconditioning ...
    #
    if TsP.SadPtPrec:
        MLump = np.atleast_2d(MSme.diagonal()).T
        MLump2 =  MLump[-(Np-1):,]
        MLumpI = 1./MLump
        MLumpI1 = MLumpI[:-(Np-1),]
        MLumpI2 = MLumpI[-(Np-1):,]
        def PrecByB2(qqpq):
            qq = MLumpI*qqpq[:Nv,] 

            p  = qqpq[Nv:-(Np-1),]
            p  = spsla.spsolve(B2Sme, p)
            p  = MLump2*np.atleast_2d(p).T
            p  = spsla.spsolve(B2Sme.T,p)
            p  = np.atleast_2d(p).T

            q2 = qqpq[-(Np-1):,]
            q2 = spsla.spsolve(B2Sme,q2)
            q2 = np.atleast_2d(q2).T

            return np.vstack([np.vstack([qq, -p]), q2])
        
        MGmr = spsla.LinearOperator( (Nv+2*(Np-1),Nv+2*(Np-1)), matvec=PrecByB2, dtype=np.float32 )
        TsP.Ml = MGmr
    
    def smamin_ip(qqpq1, qqpq2):
        """inner product that 'inverts' the preconditioning
        
        for better comparability of the residuals, i.e. the tolerances
        """
        def _inv_prec(qqpq):
            qq = MLump*qqpq[:Nv,] 
            p  = qqpq[Nv:-(Np-1),]
            p  = B2Sme.T*p
            p  = MLumpI2*p
            p  = B2Sme*p
            q2 = qqpq[-(Np-1):,]
            q2 = B2Sme*q2
            return qq, p, q2

        if TsP.SadPtPrec:
            qq1,p1,q21 = _inv_prec(qqpq1)
            qq2,p2,q22 = _inv_prec(qqpq2)
        else:
            qq1,p1,q21 = qqpq1[:Nv,], qqpq1[Nv:-(Np-1),], qqpq1[-(Np-1):,]
            qq2,p2,q22 = qqpq2[:Nv,], qqpq2[Nv:-(Np-1),], qqpq2[-(Np-1):,]

        return mass_fem_ip(qq1,qq2,MSme) + mass_fem_ip(p1,p2,MPc) + mass_fem_ip(q21,q22,MPc)

    def smamin_prec_fem_ip(qqpq1,qqpq2):
        """ M ip for the preconditioned residuals

        """
        return np.dot(qqpq1[:Nv,].T.conj(),MSme*qqpq2[:Nv,]) + \
                np.dot(qqpq1[Nv:-Npc,].T.conj(),MPc*qqpq2[Nv:-Npc,]) + \
                np.dot(qqpq1[-Npc:,].T.conj(),MPc*qqpq2[-Npc:,]) 

    v, p   = expand_vp_dolfunc(PrP, vp=vp_init, vc=None, pc=None, pdof=None)
    TsP.UpFiles.u_file << v, tcur
    TsP.UpFiles.p_file << p, tcur

    vp_old = np.copy(vp_init)
    q1_old = vp_init[~B2BoolInv,]
    q2_old = vp_init[B2BoolInv,]

    if qqpq_init is None:
        # initial value for tq2
        ConV, CurFv = get_conv_curfv_rearr(v,PrP,tcur,B2BoolInv)
        tq2_old = spsla.spsolve(M2Sme[-(Np-1):,:], CurFv[-(Np-1):,])
        #tq2_old = MLumpI2*CurFv[-(Np-1):,]
        tq2_old = np.atleast_2d(tq2_old).T
        # state vector of the smaminex system : [ q1^+, tq2^c, p^c, q2^+] 
        qqpq_old = np.zeros((Nv+2*(Np-1),1))
        qqpq_old[:Nv-(Np-1),] = q1_old
        qqpq_old[Nv-(Np-1):Nv,] = tq2_old 
        qqpq_old[Nv:Nv+Np-1,] = PFac * vp_old[Nv:,]
        qqpq_old[Nv+Np-1:,] = q2_old
    else:
        qqpq_old = qqpq_init

    ContiRes, VelEr, PEr, TolCorL = [], [], [], []

    for etap in range(1,TsP.NOutPutPts +1 ):
        for i in range(Nts/TsP.NOutPutPts):

            ConV, CurFv = get_conv_curfv_rearr(v,PrP,tcur,B2BoolInv)

            gdot = np.zeros((Np-1,1)) # TODO: implement \dot g

            Iterrhs = 1.0/dt * np.vstack([MFac * M1Sme*q1_old, WCD*B1Sme*q1_old]) \
                        + np.vstack([MFac*(FvbcSme+CurFv-ConV),WCD*gdot])
            Iterrhs = np.vstack([Iterrhs,FpbcSmeC])
            
            #Norm of rhs of index-1 formulation
            if TsP.TolCorB:
                NormRhsInd1 = np.sqrt(smamin_fem_ip(Iterrhs,Iterrhs,MSme,MPc,Nv,Npc))[0][0]
                TolCor = 1.0 / np.max([NormRhsInd1,1])

            else:
                TolCor = 1.0

            if TsP.linatol == 0:
                q1_tq2_p_q2_new = spsla.spsolve(IterA,Iterrhs) 
                qqpq_old = np.atleast_2d(q1_tq2_p_q2_new).T
            else:
                # Values from previous calculations to initialize gmres
                if TsP.UsePreTStps:
                    dname = 'ValSmaMinNts%dN%dtcur%e' % (Nts, N, tcur)
                    try:
                        IniV = loadmat(dname)
                        qqpq_old = IniV['qqpq_old']
                    except IOError:
                        pass

                q1_tq2_p_q2_new = krypy.linsys.gmres(IterA, Iterrhs,
                        x0=qqpq_old, Ml=TsP.Ml, Mr=TsP.Mr, 
                        inner_product=smamin_prec_fem_ip,
                        tol=TolCor*TsP.linatol, maxiter=TsP.MaxIter,
                        max_restarts=20)
                qqpq_old = np.atleast_2d(q1_tq2_p_q2_new['xk'])

                if TsP.SaveTStps:
                    from scipy.io import savemat
                    dname = 'ValSmaMinNts%dN%dtcur%e' % (Nts, N, tcur)
                    savemat(dname, { 'qqpq_old': qqpq_old })

                if q1_tq2_p_q2_new['info'] != 0:
                    print q1_tq2_p_q2_new['relresvec'][-5:]
                    raise Warning('no convergence')
                
                print 'Needed %d of max  %r iterations: final relres = %e\n TolCor was %e' %(len(q1_tq2_p_q2_new['relresvec']), TsP.MaxIter, q1_tq2_p_q2_new['relresvec'][-1], TolCor )

            q1_old = qqpq_old[:Nv-(Np-1),]
            q2_old = qqpq_old[-Npc:,]

            # Extract the 'actual' velocity and pressure
            vc = np.zeros((Nv,1))
            vc[~B2BoolInv,] = q1_old
            vc[B2BoolInv,] = q2_old
            print np.linalg.norm(vc)

            pc = PFacI * qqpq_old[Nv:Nv+Np-1,]

            v, p = expand_vp_dolfunc(PrP, vp=None, vc=vc, pc=pc, pdof = Pdof)
            
            tcur += dt

            # the errors and residuals 
            vCur, pCur = PrP.v, PrP.p 
            vCur.t = tcur
            pCur.t = tcur - dt

            ContiRes.append(comp_cont_error(v,FpbcSme,PrP.Q))
            VelEr.append(errornorm(vCur,v))
            PEr.append(errornorm(pCur,p))
            TolCorL.append(TolCor)

            if i + etap == 1 and TsP.SaveIniVal:
                from scipy.io import savemat
                dname = 'IniValSmaMinN%s' % N
                savemat(dname, { 'qqpq_old': qqpq_old })

        print '%d of %d time steps completed ' % (etap*Nts/TsP.NOutPutPts, Nts) 

        if TsP.ParaviewOutput:
            TsP.UpFiles.u_file << v, tcur
            TsP.UpFiles.p_file << p, tcur

    TsP.Residuals.ContiRes.append(ContiRes)
    TsP.Residuals.VelEr.append(VelEr)
    TsP.Residuals.PEr.append(PEr)
    TsP.TolCor.append(TolCorL)
        
    return

def halfexp_euler_nseind2(Mc,MP,Ac,BTc,Bc,fvbc,fpbc,vp_init,PrP,TsP):
    """halfexplicit euler for the NSE in index 2 formulation
    """
    ####
    #
    # Basic Eqn:
    #
    # 1/dt*M  -B.T    q+       1/dt*M*qc - K(qc) + fc
    #    B         *  pc   =   g
    #
    ########

    Nts, t0, tE, dt, Nv, Np = init_time_stepping(PrP,TsP)

    tcur = t0
    
    MFac = 1
    CFac = 1 #/dt
    PFac  = -1   #-1 for symmetry (if CFac==1)
    PFacI = -1 

    v, p   = expand_vp_dolfunc(PrP, vp=vp_init, vc=None, pc=None)
    TsP.UpFiles.u_file << v, tcur
    TsP.UpFiles.p_file << p, tcur

    IterAv = MFac*sps.hstack([1.0/dt*Mc,PFacI*(-1)*BTc[:,:-1]])
    IterAp = CFac*sps.hstack([Bc[:-1,:],sps.csr_matrix((Np-1,Np-1))])
    IterA  = sps.vstack([IterAv,IterAp])

    MPc = MP[:-1,:][:,:-1]

    vp_old = vp_init
    ContiRes, VelEr, PEr, TolCorL = [], [], [], []

    # M matrix for the minres routine
    # M accounts for the FEM discretization
    def _MInv(vp):
        v, p = vp[:Nv,], vp[Nv:,]
        Mv = krypy.linsys.cg(Mc,v,tol=1e-14)['xk']
        Mp = krypy.linsys.cg(MPc,p,tol=1e-14)['xk']
        return np.vstack([Mv,Mp])

    MInv = spsla.LinearOperator( (Nv+Np-1,Nv+Np-1), matvec=_MInv, dtype=np.float32 )
    
    def ind2_ip(vp1,vp2):
        """

        for applying the fem inner product
        """
        v1, v2 = vp1[:Nv,], vp2[:Nv,]
        p1, p2 = vp1[Nv:,], vp2[Nv:,]
        return mass_fem_ip(v1,v2,Mc) + mass_fem_ip(p1,p2,MPc)

    for etap in range(1,TsP.NOutPutPts + 1 ):
        for i in range(Nts/TsP.NOutPutPts):

            ConV  = dtn.get_convvec(v, PrP.V)
            CurFv = dtn.get_curfv(PrP.V, PrP.fv, PrP.invinds, tcur)

            Iterrhs = np.vstack([MFac*1.0/dt*Mc*vp_old[:Nv,],np.zeros((Np-1,1))]) \
                    + np.vstack([MFac*(fvbc+CurFv-ConV[PrP.invinds,]),
                        CFac*fpbc[:-1,]])

            if TsP.linatol == 0:
                vp_new = spsla.spsolve(IterA,Iterrhs)#,vp_old,tol=TsP.linatol)
                vp_old = np.atleast_2d(vp_new).T

            else:
                if TsP.TolCorB:
                    NormRhsInd2 = np.sqrt(ind2_ip(Iterrhs,Iterrhs))[0][0]
                    TolCor = 1.0 / np.max([NormRhsInd2,1])
                else:
                    TolCor = 1.0

                ret = krypy.linsys.minres(IterA, Iterrhs, 
                        x0=vp_old, tol=TolCor*TsP.linatol,
                        M=MInv)
                vp_old = ret['xk'] 

                print 'Needed %d iterations -- final relres = %e' %(len(ret['relresvec']), ret['relresvec'][-1] )
                print 'TolCor was %e' % TolCor 

            vc = vp_old[:Nv,]
            pc = PFacI*vp_old[Nv:,]

            v, p = expand_vp_dolfunc(PrP, vp=None, vc=vc, pc=pc)

            tcur += dt

            # the errors  
            vCur, pCur = PrP.v, PrP.p 
            vCur.t = tcur 
            pCur.t = tcur - dt

            ContiRes.append(comp_cont_error(v,fpbc,PrP.Q))
            VelEr.append(errornorm(vCur,v))
            PEr.append(errornorm(pCur,p))
            TolCorL.append(TolCor)

        print '%d of %d time steps completed ' % (etap*Nts/TsP.NOutPutPts, Nts) 

        if TsP.ParaviewOutput:
            TsP.UpFiles.u_file << v, tcur
            TsP.UpFiles.p_file << p, tcur

    TsP.Residuals.ContiRes.append(ContiRes)
    TsP.Residuals.VelEr.append(VelEr)
    TsP.Residuals.PEr.append(PEr)
    TsP.TolCor.append(TolCorL)
        
    return

def qr_impeuler(Mc,Ac,BTc,Bc,fvbc,fpbc,vp_init,PrP,TsP=None):
    """ with BTc[:-1,:] = Q*[R ; 0] 
    we define ~M = Q*M*Q.T , ~A = Q*A*Q.T , ~V = Q*v
    and condense the system accordingly """

    Nts, t0, tE, dt, Nv, Np = init_time_stepping(PrP,TsP)

    tcur = t0

    v, p   = expand_vp_dolfunc(PrP, vp=vp_init, vc=None, pc=None)
    TsP.UpFiles.u_file << v, tcur
    TsP.UpFiles.p_file << p, tcur

    Qm, Rm = qr(BTc.todense()[:,:-1],mode='full')

    Qm_1 = Qm[:Np-1,]   #first Np-1 rows of Q
    Qm_2 = Qm[Np-1:,]   #last Nv-(Np-1) rows of Q

    TM_11 = np.dot(Qm_1,np.dot(Mc.todense(),Qm_1.T))
    TA_11 = np.dot(Qm_1,np.dot(Ac.todense(),Qm_1.T))

    TM_21 = np.dot(Qm_2,np.dot(Mc.todense(),Qm_1.T))
    TA_21 = np.dot(Qm_2,np.dot(Ac.todense(),Qm_1.T))

    TM_12 = np.dot(Qm_1,np.dot(Mc.todense(),Qm_2.T))
    TA_12 = np.dot(Qm_1,np.dot(Ac.todense(),Qm_2.T))

    TM_22 = np.dot(Qm_2,np.dot(Mc.todense(),Qm_2.T))
    TA_22 = np.dot(Qm_2,np.dot(Ac.todense(),Qm_2.T))

    Tv1 = np.linalg.solve(Rm[:Np-1,].T, fpbc[:-1,])

    Tv2_old = np.dot(Qm_2, vp_init[:Nv,])

    Tfv2 = np.dot(Qm_2, fvbc) + np.dot(TA_21, Tv1)

    IterA2  = TM_22+dt*TA_22

    for etap in range(1,11):
        for i in range(Nts/10):
            tcur = tcur + dt

            Iter2rhs = np.dot(TM_22, Tv2_old) + dt*Tfv2
            Tv2_new = np.linalg.solve(IterA2, Iter2rhs)

            Tv2_old = Tv2_new

            # Retransformation v = Q.T*Tv
            vc_new = np.dot(Qm.T, np.vstack([Tv1, Tv2_new]))
            
            RhsTv2dot = Tfv2 - np.dot(TA_22, Tv2_new) 
            Tv2dot = np.linalg.solve(TM_22, RhsTv2dot)

            RhsP = np.dot(Qm_1,fvbc) - np.dot(TA_11,Tv1) \
                    - np.dot(TA_12,Tv2_new) - np.dot(TM_12,Tv2dot)

            pc_new = - np.linalg.solve(Rm[:Np-1,],RhsP)

        v, p = expand_vp_dolfunc(PrP, vp=None, vc=vc_new, pc=pc_new)
        TsP.UpFiles.u_file << v, tcur
        TsP.UpFiles.p_file << p, tcur
        print '%d of %d time steps completed ' % (etap*Nts/10,Nts) 
        
    return

def plain_impeuler(Mc,Ac,BTc,Bc,fvbc,fpbc,vp_init,PrP,TsP):

    Nts, t0, tE, dt, Nv, Np = init_time_stepping(PrP,TsP)

    v, p   = expand_vp_dolfunc(PrP, vp=vp_init, vc=None, pc=None)

    tcur = t0

    TsP.UpFiles.u_file << v, tcur
    TsP.UpFiles.p_file << p, tcur

    IterAv = sps.hstack([Mc+dt*Ac,-dt*BTc])
    IterAp = sps.hstack([-dt*Bc,sps.csr_matrix((Np,Np))])
    IterA  = sps.vstack([IterAv,IterAp]).todense()[:-1,:-1]

    vp_old = vp_init
    for etap in range(1,11):
        for i in range(Nts/10):
            tcur = tcur + dt

            Iterrhs = np.vstack([Mc*vp_old[:Nv,],np.zeros((Np-1,1))]) \
                    + dt*np.vstack([fvbc,fpbc[:-1,]])
            vp_new = np.linalg.solve(IterA,Iterrhs)
            vp_old = vp_new


        print '%d of %d time steps completed ' % (etap*Nts/10,Nts) 
        v, p = expand_vp_dolfunc(PrP, vp=vp_new, vc=None, pc=None)

        TsP.UpFiles.u_file << v, tcur
        TsP.UpFiles.p_file << p, tcur
        
    return

def comp_cont_error(v,fpbc,Q):
    """Compute the L2 norm of the residual of the continuity equation
        Bv = g
    """

    q = TrialFunction(Q)
    divv = assemble(q*div(v)*dx)

    conRhs = Function(Q)
    conRhs.vector().set_local(fpbc)

    #raise Warning('debugggg')

    ContEr = norm(conRhs.vector()-divv)

    return ContEr


def expand_vp_dolfunc(PrP, vp=None, vc=None, pc=None, pdof=None):
    """expand v and p to the dolfin function representation
    
    pdof = pressure dof that was set zero
    """

    v = Function(PrP.V)
    p = Function(PrP.Q)

    if vp is not None:
        if vp.ndim == 1:
            vc = vp[:len(PrP.invinds)].reshape(len(PrP.invinds),1)
            pc = vp[len(PrP.invinds):].reshape(PrP.Q.dim()-1,1)
        else:
            vc = vp[:len(PrP.invinds),:]
            pc = vp[len(PrP.invinds):,:]

    ve = np.zeros((PrP.V.dim(),1))

    # fill in the boundary values
    for bc in PrP.velbcs:
        bcdict = bc.get_boundary_values()
        ve[bcdict.keys(),0] = bcdict.values()

    ve[PrP.invinds] = vc

    if pdof is None:
        pe = np.vstack([pc,[0]])
    elif pdof == 0:
        pe = np.vstack([[0],pc])
    elif pdof == -1:
        pe = pc
    else:
        pe = np.vstack([pc[:pdof],np.vstack([[0.02],pc[pdof:]])])

    v.vector().set_local(ve)
    p.vector().set_local(pe)

    return v, p

def get_conv_curfv_rearr(v,PrP,tcur,B2BoolInv):

        ConV  = dtn.get_convvec(v, PrP.V)
        ConV = ConV[PrP.invinds,]

        ConV = np.vstack([ConV[~B2BoolInv],ConV[B2BoolInv]])

        CurFv = dtn.get_curfv(PrP.V, PrP.fv, PrP.invinds, tcur)
        if len(CurFv) != len(PrP.invinds):
            raise Warning('Need fv at innernodes here')
        CurFv = np.vstack([CurFv[~B2BoolInv],CurFv[B2BoolInv]])

        return ConV, CurFv


def init_time_stepping(PrP,TsP):
    """what every method starts with """
    
    Nts, t0, tE = TsP.Nts, TsP.t0, TsP.tE
    dt = (tE-t0)/Nts
    Nv = len(PrP.invinds)
    Np = PrP.Q.dim()

    if Nts % TsP.NOutPutPts != 0:
        TsP.NOutPutPts = 1

    return Nts, t0, tE, dt, Nv, Np
