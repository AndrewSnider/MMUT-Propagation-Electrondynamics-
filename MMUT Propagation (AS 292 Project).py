#!/usr/bin/env python
# coding: utf-8

# ### Phys 292 Final Project Code (Final Version)

# A good deal of the code in the following cell comes out of the work done by other people working on the project. It would be too difficult to assign individual contributions so I will generally thank Dr. Christine Isborn, Dr. Harish Bhat, and Karnamohit Ranka for there work in getting the code to this point. Special thanks should go to harish, who is responsible for collecting a lot of the messy approaches into their respective modularized components. 
# 
# This code is basically the orginal we are currently working with on the project but with most of the ML related methods stripped out. You'll still see some references to the ML part (e.g. the class name) but I tried to isolate the propagation code as much as possible for clarity sake.

# In[1]:


#single trajectory
#Hamiltonian real/imag parts depend on both real/imag parts of density
#Hamiltonian has zeros in locations where density matrices have zeros; all other DOFs are active




import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import expm
import numpy as onp
import scipy.integrate as si
import scipy.linalg as sl

from functools import partial

class LearnHam:

    # class initializer
    def __init__(self, mol, outpath):
        
        # store the short string that indicates which molecule we're working on
        self.mol = mol
        
        self.offset = 2
        self.dt = 0.08268 
        self.intpts = 2000
        
        # field sign correction
        if self.mol == 'h2':
            self.fieldsign = -1
        else:
            self.fieldsign = 1
            
        # store the path to output files, i.e., saved research outputs like figures
        self.outpath = outpath
    # load and process field-free data
    def load(self,inpath):
        # store the path to input files, i.e., training data, auxiliary matrices, etc
        inpath = inpath
        rawden = onp.load(inpath + 'td_dens_re+im_rt-tdexx_delta_s0_'+mol+'_sto-3g.npz',allow_pickle=True)
        overlap = onp.load(inpath + 'ke+en+overlap+ee_twoe+dip_hf_delta_s0_'+mol+'_sto-3g.npz',allow_pickle=True)

        # put things into better variables
        self.kinmat = overlap['ke_data']
        self.enmat = overlap['en_data']
        self.eeten = overlap['ee_twoe_data']

        # need these for orthogonalization below
        s = overlap['overlap_data']
        self.sevals, self.sevecs = onp.linalg.eigh(s)
        self.xmat = self.sevecs @ onp.diag(self.sevals**(-0.5))
        
        
        # remove duplicates
        realpt = rawden['td_dens_re_data']
        imagpt = rawden['td_dens_im_data']
        den = realpt + 1j*imagpt
        self.drc = den.shape[1]
        # Read dipole data
        self.didat = [[]]*3
        self.didat[0] = onp.zeros(shape=(self.drc,self.drc))
        self.didat[1] = onp.zeros(shape=(self.drc,self.drc))
        self.didat[2] = onp.zeros(shape=(self.drc,self.drc))
        self.didat[0] = overlap['dipx_data']
        self.didat[1] = overlap['dipy_data']
        self.didat[2] = overlap['dipz_data']
        print('dipole',self.didat[2])
        denflat = den.reshape((-1,self.drc**2))
        dennodupflat = onp.array([onp.delete(denflat[:,i], onp.s_[101::100]) for i in range(self.drc**2)]).T
        self.denAO = dennodupflat.reshape((-1,self.drc,self.drc))

        # transform to MO using canonical orthogonalization
        # in this code, by "MO" we mean the canonical orthogonalization of the AO basis
        self.denMO = onp.zeros(self.denAO.shape,dtype=onp.complex128)
        self.denMOflat = self.denMO.reshape((-1,self.drc**2))
        onpts = self.denAO.shape[0]
        for i in range(1,onpts):
            self.denMO[i,:,:] = onp.diag(self.sevals**(0.5)) @ self.sevecs.T @ self.denAO[i,:,:] @ self.sevecs @ onp.diag(self.sevals**(0.5))

        # find off-diag DOFs of the supplied density matrices that are (sufficiently close to) zero across all time points
        self.realnzs = []
        self.imagnzs = []
        for j in range(realpt.shape[1]):
            for i in range(j+1):
                realnorm = onp.linalg.norm(onp.real(self.denMO[:,i,j]))
                # print("|| Re[den["+str(i)+","+str(j)+"]] || = " + str(realnorm))
                if not onp.isclose(realnorm,0):
                    self.realnzs.append((i,j))

                if i < j:
                    imagnorm = onp.linalg.norm(onp.imag(self.denMO[:,i,j]))
                    # print("|| Im[den["+str(i)+","+str(j)+"]] || = " + str(imagnorm))
                    if not onp.isclose(imagnorm,0):
                        self.imagnzs.append((i,j))

        # these turn out to be super useful when we build the ML Hamiltonian matrices much further down
        self.rnzl = [list(t) for t in zip(*self.realnzs)]
        self.inzl = [list(t) for t in zip(*self.imagnzs)]

        # build two dictionaries that help us find the absolute column number given human-readable (i,j) indices
        # for both the real and imaginary non-zero density DOFs
        # also build matrix equivalents for these dictionaries, which are needed by numba jit
        self.nzreals = {}
        self.nzrealm = -onp.ones((self.drc,self.drc),dtype=onp.int32)
        cnt = 0
        for i in self.realnzs:
            self.nzreals[i] = cnt
            self.nzrealm[i[0],i[1]] = cnt
            cnt += 1

        self.nzimags = {}
        self.nzimagm = -onp.ones((self.drc,self.drc),dtype=onp.int32)
        for i in self.imagnzs:
            self.nzimags[i] = cnt
            self.nzimagm[i[0],i[1]] = cnt
            cnt += 1

        # need all of the following for our fast Hessian assembler
        self.ndof = cnt
        self.allnzs = list(set(self.realnzs + self.imagnzs))
        self.nall = len(self.allnzs)
        self.nzrow = onp.zeros(self.nall, dtype=onp.int32)
        self.nzcol = onp.zeros(self.nall, dtype=onp.int32)
        for i in range(self.nall):
            self.nzrow[i] = self.allnzs[i][0]
            self.nzcol[i] = self.allnzs[i][1]
        
        # show that we got here
        return True
    


    
    
 # Karnamohit's function (July 1 version)
    # this computes the Coulomb and exchange parts of the potential
    def get_ee_onee_AO(self, dens, exchange=True):
        assert len(dens.shape) == 2
        assert len(self.eeten.shape) == 4
        assert dens.shape[0] == dens.shape[1], 'Density matrix (problem with axes 0 and 1, all axis-dimensions must be the same!)'
        assert self.eeten.shape[0] == self.eeten.shape[1], 'ERIs (problem with axes 0 and 1, all axis-dimensions must be the same!)'
        assert self.eeten.shape[2] == self.eeten.shape[3], 'ERIs (problem with axes 2 and 3, all axis-dimensions must be the same!)'
        assert self.eeten.shape[0] == self.eeten.shape[2], 'ERIs (problem with axes 0 and 2, all axis-dimensions must be the same!)'
        e = True
        if (dens.shape[0] == self.eeten.shape[0]):
            nbas = dens.shape[0]
            vee_data = onp.zeros((nbas, nbas), dtype=onp.complex128)
            e = False
            if (exchange == True):
                for u in range(nbas):
                    for v in range(u,nbas):
                        for l in range(nbas):
                            for s in range(nbas):
                                # coulomb - 0.5*exchange
                                vee_data[u,v] += 2*dens[l,s]*(self.eeten[u,v,l,s])
                                vee_data[u,v] -= 2*dens[l,s]*(0.5*self.eeten[u,l,v,s])
                        vee_data[v,u] = onp.conjugate(vee_data[u,v])
            elif (exchange == False):
                for u in range(nbas):
                    for v in range(u,nbas):
                        for l in range(nbas):
                            for s in range(nbas):
                                # coulomb
                                vee_data[u,v] += 2*dens[l,s]*(self.eeten[u,v,l,s])
                        vee_data[v,u] = onp.conjugate(vee_data[u,v])
            return vee_data
        elif (e == True):
            print('\nError: Shapes of density and ERI tensors are not compatible.')
            return
        
        
        
        

 # EXACT deltakick Hamiltonian, NO FIELD
    # this function is defined for propagation purposes
    def EXhamrhs(self, t, pin):  # time, density(p) in
        p = pin.reshape(self.drc,self.drc)
        
        pAO = self.xmat @ p @ self.xmat.conj().T
        twoe = self.get_ee_onee_AO(pAO)
        hAO = onp.array(self.kinmat - self.enmat, dtype=onp.complex128) + twoe
        h = -self.xmat.conj().T @ hAO @ self.xmat

        rhs = (h @ p - p @ h)/(1j)
        return rhs.reshape(self.drc**2)
    
    def fock_build(self,pin):
        p = pin.reshape(self.drc,self.drc)
        
        pAO = self.xmat @ p @ self.xmat.conj().T
        twoe = self.get_ee_onee_AO(pAO)
        hAO = onp.array(self.kinmat - self.enmat, dtype=onp.complex128) + twoe
        h = -self.xmat.conj().T @ hAO @ self.xmat
        return(h)
    

    
    
    # propagate one method forward in time from self.offset to intpts = "integration points"
    # use initial condition given by initcond
    # use RK45 integration with relative and absolute tolerances set to mytol
    def propagate(self, rhsfunc, initcond, intpts=2000, mytol=1e-12):
        self.intpts = intpts
        self.tvec = self.dt*onp.arange(intpts-self.offset)
        THISsol = si.solve_ivp(rhsfunc, [0, self.tvec[-1]], initcond, 'RK45', t_eval = self.tvec, rtol=mytol, atol=mytol)
        return THISsol.y
    
    
   # think of traj1 and traj2 as two different numerical solutions that we got by running propagate
    # and groundtruth as the ground truth
    # here we compare the two trajectories QUANTITATIVELY
    def quantcomparetraj(self, traj1, traj2, groundtruth, fname='tdHamerr.npz'):

        errors = onp.zeros(2)

        # error between propagating machine learned Hamiltonian and Gaussian data
        #errors[0] = onp.mean(onp.linalg.norm( traj1.T.reshape((-1,self.drc,self.drc)) - groundtruth[self.offset:self.intpts,:,:], axis = (1,2) ))
        
        # error between propagating exact Hamiltonian and Gaussian data
        errors[0] = onp.mean(onp.linalg.norm( traj1.T.reshape((-1,self.drc,self.drc)) - groundtruth[self.offset:self.intpts,:,:] , axis = (1,2) ))
        errors[1] = onp.mean(onp.linalg.norm( traj2.T.reshape((-1,self.drc,self.drc)) - groundtruth[self.offset:self.intpts,:,:] , axis = (1,2) ))
        # error between propagating exact Hamiltonian and propagating machine learned Hamiltonian
        #errors[2] = onp.mean(onp.linalg.norm( traj2.T.reshape((-1,self.drc,self.drc)) - traj1.T.reshape((-1,self.drc,self.drc)), axis = (1,2) ))
        
        # compute and save time-dependent propagation errors 
        tdexHamerr = onp.linalg.norm( traj1.T.reshape((-1,self.drc,self.drc)) - groundtruth[self.offset:self.intpts,:,:] , axis=(1,2))
        #tdmlHamerr = onp.linalg.norm( traj1.T.reshape((-1,self.drc,self.drc)) - groundtruth[self.offset:self.intpts,:,:] , axis=(1,2))
        #tdexmlerr = onp.linalg.norm( traj2.T.reshape((-1,self.drc,self.drc)) - traj1.T.reshape((-1,self.drc,self.drc)) , axis=(1,2))
        
        #onp.savez(self.outpath +mol+ fname,tdexHamerr=tdexHamerr)
        return errors   
    
    
    # think of traj1 and traj2 as two different numerical solutions that we got by running propagate
    # and groundtruth as the ground truth
    # here we compare the two trajectories GRAPHICALLY
    def graphcomparetraj(self, traj1, traj2, groundtruth, myfigsize=(8,16), includeField=False, fname='test.pdf', mytitle=None):

        fig = plt.figure(figsize=(myfigsize))
        mylabels = []
        if includeField:
            axs = fig.subplots(self.ndof+1)
            trueefield = 0.05*onp.sin(0.0428*self.tvec)
            trueefield[1776:] = 0.
            axs[0].plot(self.tvec, trueefield, 'k-')
            thislabel = 'E-field'
            mylabels.append(thislabel)
            ctr = 1
        else:
            axs = fig.subplots(self.ndof)
            ctr = 0
        
        if mytitle == None:
            mytitle = 'Gaussian(MMUT) (black), exact-H(RK45) (blue), and exact-H(MMUT)(red) propagation results'
        fig.suptitle(mytitle,y=0.9)

        for ij in self.nzreals:
            axs[ctr].plot(self.tvec, onp.real(traj2.T.reshape((-1,self.drc,self.drc))[:,ij[0],ij[1]]), 'b-')
            axs[ctr].plot(self.tvec, onp.real(traj1.T.reshape((-1,self.drc,self.drc))[:,ij[0],ij[1]]), 'r-')
            axs[ctr].plot(self.tvec, onp.real(groundtruth[self.offset:self.intpts,ij[0],ij[1]]), 'k-')
            ijprime = (ij[0]+1, ij[1]+1)
            thislabel = 'Re(P_' + str(ijprime) + ')'
            mylabels.append(thislabel)
            ctr += 1
        
        for ij in self.nzimags:
            axs[ctr].plot(self.tvec, onp.imag(traj2.T.reshape((-1,self.drc,self.drc))[:,ij[0],ij[1]]), 'b-')
            axs[ctr].plot(self.tvec, onp.imag(traj1.T.reshape((-1,self.drc,self.drc))[:,ij[0],ij[1]]), 'r-')
            axs[ctr].plot(self.tvec, onp.imag(groundtruth[self.offset:self.intpts,ij[0],ij[1]]), 'k-')
            ijprime = (ij[0]+1, ij[1]+1)
            thislabel = 'Im(P_' + str(ijprime) + ')'
            mylabels.append(thislabel)
            ctr += 1
        
        plt.subplots_adjust(wspace=0, hspace=0)

        cnt = 0
        for ax in axs.flat:
            ax.set(xlabel='time', ylabel=mylabels[cnt])
            if cnt % 2 == 0:
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
            cnt += 1
        
        for ax in axs.flat:
            ax.label_outer()
        
        fig.savefig(self.outpath + mol + fname)
        plt.close()
        return True
    


    
    


# In[ ]:





# In[7]:


### initialize 
mol = 'h2'
mlham = LearnHam(mol,'')
mlham.load('Merced/data/') 
EXsol = mlham.propagate(mlham.EXhamrhs, mlham.denMOflat[mlham.offset,:], mytol=1e-6)


#initial condition
p0 = mlham.denMOflat[mlham.offset,:]
propagated_dens = onp.zeros((mlham.drc**2,mlham.intpts-mlham.offset) ,dtype=onp.complex128)
propagated_dens[:,0] = p0

#go through algo steps
tpoints =mlham.tvec
for i in range(1,len(tpoints)):
    if i == 1:
        P_n_min_12 = p0
        P_n = P_n_min_12 + mlham.dt*mlham.EXhamrhs(i,P_n_min_12)
        P_n = P_n.reshape(2,2)
        F_n = mlham.fock_build(P_n)
        P_n_min_12 =  P_n_min_12.reshape(2,2)
    else:
        P_n_min_12 = P_n_plus_12
        F_n = mlham.fock_build(P_n_plus_1)
        
    
    
    P_n_plus_12 = expm(-1j*mlham.dt*F_n) @ P_n_min_12 @ expm(1j*mlham.dt*F_n)
    propagated_dens[:,i] = P_n_plus_12.reshape(mlham.drc**2)
    P_n_plus_1 = expm(-1j*mlham.dt/2*F_n) @  P_n_plus_12 @ expm(1j*mlham.dt/2*F_n)

    
    
err = mlham.quantcomparetraj(propagated_dens,EXsol, mlham.denMO)
print('RK45 error:',err[1])
print('MMUT_AS error:',err[0])


# These error values are a result of comparing propagated densities generated by 1) Runge-Kutta 4(5) order (aka old method) and 2)The MMUT algorithim shown below with the densities generated via the Gaussian electronic structure code (MMUT). 

# In[3]:


mlham.graphcomparetraj(propagated_dens, EXsol, mlham.denMO, fname='MMUT_RK45_Gauss_error.pdf', mytitle='Gaussian(MMUT) (black), exact-H(RK45) (blue), and exact-H(MMUT)(red) propagation results')


# Quick check on resulting figures:

# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
gauss = onp.zeros(mlham.intpts-mlham.offset)
mmut = onp.zeros(mlham.intpts-mlham.offset)
rk45 = onp.zeros(mlham.intpts-mlham.offset)
for i in range(0,mlham.intpts-mlham.offset):
    gauss[i] = onp.imag(mlham.denMOflat[i+mlham.offset,:][1])
    mmut[i] = onp.imag(propagated_dens[:,i][1])
    rk45[i] = onp.imag(EXsol[:,i][1])
plt.plot(mlham.tvec,mmut,'black')


# In[5]:


plt.plot(mlham.tvec,gauss,'black')
#plt.savefig('Gaussian(MMUT)')


# In[6]:


plt.plot(mlham.tvec,rk45,'black')
#plt.savefig(RK)


# One problem that I mostly skip over here is the issue of matrix exponentiation. 

# In[ ]:




