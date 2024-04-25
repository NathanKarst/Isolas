import viscosity
import skimming
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import scipy.io
import pickle


class Network:
    def __init__(self,params):
        self.seed = None
        for key in params:
            setattr(self, key, params[key])
            
        self.params = params
        
        self.set_adj_inc() # compute adjacency and incidence matrices
        
        # state vector is split into two pieces...
        self.p = np.array((self.nNodes)*[np.nan])
        self.q = np.array((self.nVessels)*[np.nan])
        self.h = np.array(self.nVessels*[np.nan]) # ... [hematocrits]
        self.equilibria = []
        

    #########################
    ##### BASIC METHODS #####
    #########################
    
    def set_parameter(self,paramName,paramValue,index=None):
        if index == None:
            self.__setattr__(paramName,paramValue)
        else:
            tmp = getattr(self,paramName)
            tmp[index] = paramValue
            self.__setattr__(paramName,tmp)        

    def set_adj_inc(self):
        n_v = len(self.v)
        n_e = len(self.e)
        adj = np.zeros((n_v,n_v))
        inc = np.zeros((n_v,n_e))

        for i,edge in enumerate(self.e):
            e0 = min(edge)
            e1 = max(edge)

            adj[e0,e1] = 1
            adj[e1,e0] = 1

            # directed edge from lower index to higher index
            inc[e0,i] = -1
            inc[e1,i] = 1
            
        self.adj = adj
        self.inc = inc
        
        self.nNodes = int(adj.shape[0])
        self.nVessels = int(inc.shape[1])

        self.interiorNodes = np.where(np.sum(np.abs(inc),axis=1) == 3)[0]
        self.exteriorNodes = np.where(np.sum(np.abs(inc),axis=1) == 1)[0]
        self.exteriorFlows = np.where(np.sum(np.abs(inc[self.exteriorNodes,:]),axis=0) == 1)[0]
        
        self.nInteriorNodes = len(self.interiorNodes)
        self.nExteriorNodes = len(self.exteriorNodes)
        self.nExteriorFlows = len(self.exteriorFlows) 
        
        
    ###############################
    ##### EQUILIBRIUM METHODS #####
    ###############################        
        
    def set_bh(self,h,bc_kind='pressure'):
        poiseuille = np.block([self.inc.T, np.zeros((self.nVessels,self.nVessels))])
        r = self.l/self.d**4*viscosity.viscosity_arrhenius(h,self.delta)
        for i in range(self.nVessels):
            poiseuille[i,self.nNodes+i] = r[i]    
        
        interior_flow_balance = np.block([np.zeros((len(self.interiorNodes),self.nNodes)),self.inc[self.interiorNodes,:]])
        boundary_conditions = np.zeros((len(self.exteriorNodes),self.nVessels + self.nNodes))
        if bc_kind == 'pressure':
            for i,node in enumerate(self.exteriorNodes):
                boundary_conditions[i,node] = 1
            tail = self.pq_bcs[self.exteriorNodes]
        else: 
            print('Error: only pressure boundary conditions implemented so far!')
                                  
#         elif bc_kind == 'flow':
#             boundary_conditions = N.inc[N.exteriorNodes,:]
#             tail = self.pq_bcs[self.exteriorFlows]
            
        self.bh = np.concatenate([poiseuille, interior_flow_balance, boundary_conditions],axis=0)
        self.bh_rhs = np.concatenate((np.zeros(self.nNodes + self.nVessels - len(self.exteriorNodes)), tail))                 
        
    def set_cq(self):
        rhs = np.zeros(self.nVessels)

        X = self.inc@np.diag(np.sign(self.q))
        exterior = np.array((np.abs(self.inc).sum(axis=1) == 1)).flatten()    
        interior = np.array((np.abs(self.inc).sum(axis=1) == 3)).flatten()    
        net_out = np.array(X.sum(axis=1) == -1).flatten()
        net_in = np.array(X.sum(axis=1) == 1).flatten()    
        
        CQ = self.inc[self.interiorNodes,:]@np.diag(self.q)
        
        div = np.where(np.bitwise_and(interior,net_out))[0]
        for node in div:
            idx_in = np.where(X[node,:] > 0)[0][0]
            idx_out = np.min(np.where(X[node,:] < 0)[0])
            row = np.zeros(self.nVessels)
            row[idx_in] = -skimming.skimming_kj(np.abs(self.q[idx_out]/self.q[idx_in]),self.pPlasma)[0]
            row[idx_out] = 1
            CQ = np.concatenate((CQ,row.reshape(1,-1)))  
        
        inlets =  np.where(np.bitwise_and(exterior,net_out))[0]
        for i,inlet in enumerate(inlets):
            row = np.zeros(self.nVessels)
            inlet_vessel = np.where(self.inc[inlet])[0][0]            
            
            row[inlet_vessel] = 1
            CQ = np.concatenate((CQ,row.reshape(1,-1)))              
            rhs[-(len(inlets)-i)] = self.h_bcs[inlet_vessel]

        self.cq = CQ
        self.cq_rhs = rhs 
    
    def set_pq(self,h):
        self.set_bh(h)
        pq = np.linalg.solve(self.bh,self.bh_rhs)
        self.p = pq[:self.nNodes]
        self.q = pq[self.nNodes:]
    
    def set_state(self,h):
        self.h = h
        self.set_pq(h)
        # self.set_bh(h) # not necessary, as B(H) is set in self.set_pq
        self.set_cq()
        self.r = self.l/self.d**4*viscosity.viscosity_arrhenius(h,self.delta)
        
    def find_equilibria(self,n,tol=1e-4,verbose=False):
        if self.seed: np.random.seed(self.seed)
        self.equilibria = []
        i = 0 
        while i < n:
            h = np.random.random(self.nVessels)
            
            outcome = fsolve(self.equilibrium_relation, h)
            residual = np.linalg.norm(self.equilibrium_relation(outcome))/len(outcome)
            if residual > tol:
                continue
        
            self.set_state(outcome)
             
            if verbose:
                print(f'{i+1}: |F(x*)| = {np.round(residual,6)}')
            if i == 0:
                self.equilibria = self.h.reshape(-1,1)
            else:
                self.equilibria = np.concatenate((self.equilibria,self.h.reshape(-1,1)),axis=1)
            i += 1        

            
            
    ################################
    ##### CONTINUATION METHODS #####
    ################################
        
    def set_jacobian(self,h):
        self._curr_h = h

        self.set_state(h)
        
        B = self.bh.copy()
        C = self.cq.copy()
        D = np.zeros((B.shape[0],C.shape[1]))
        E = np.zeros((C.shape[0],B.shape[1]))
        
        D[:self.nVessels,:self.nVessels] = np.diag(self.delta*self.q*self.r)
       
        
        X = self.inc@np.diag(np.sign(self.q))
        interior = np.array((np.abs(self.inc).sum(axis=1) == 3)).flatten()    
        net_out = np.array(X.sum(axis=1) == -1).flatten()
        
        div = np.where(np.bitwise_and(interior,net_out))[0]
        n_div = len(div)
        
        dsdh = np.zeros((n_div,self.nVessels))
        dsdq = np.zeros((n_div,self.nVessels))
        for i,node in enumerate(div):
            idx_in = np.where(X[node,:] > 0)[0][0]
            idx_out = np.min(np.where(X[node,:] < 0)[0])
           
            q_in = self.q[idx_in]
            q_out = self.q[idx_out]
        
            dsdh[i,idx_in] = -skimming.skimming_kj(np.abs(q_out/q_in),self.pPlasma)[0]
            dsdh[i,idx_out] = 1
            
            dfdq = skimming.skimming_kj_dq(np.abs(q_out/q_in),self.pPlasma)[0]
            
            dsdq[i,idx_in] = self.h[idx_in]*dfdq*np.abs(q_out)/(q_in**2)*np.sign(q_in)
            dsdq[i,idx_out] = -self.h[idx_in]*dfdq/np.abs(q_in)*np.sign(q_out)

        C[self.nInteriorNodes:self.nInteriorNodes + n_div, :] = dsdh            
            
            
        E[:self.nInteriorNodes,self.nNodes:] = self.inc[self.interiorNodes,:]@np.diag(h)            
        E[self.nInteriorNodes:self.nInteriorNodes + n_div, self.nNodes:] = dsdq
        
        self.jacobian = np.block([[B,D],[E,C]])
    
    def get_jacobian(self,h):
        self.set_jacobian(h)
        return self.jacobian
    
    def get_jacobian_from_full_state(self,x):
        p = x[:self.nNodes]
        q = x[self.nNodes:self.nNodes+self.nVessels]
        h = x[-self.nVessels:]
        
        poiseuille = np.block([self.inc.T, np.zeros((self.nVessels,self.nVessels))])
        r = self.l/self.d**4*viscosity.viscosity_arrhenius(h,self.delta)
        for i in range(self.nVessels):
            poiseuille[i,self.nNodes+i] = r[i]    
        
        interior_flow_balance = np.block([np.zeros((len(self.interiorNodes),self.nNodes)),self.inc[self.interiorNodes,:]])
        boundary_conditions = np.zeros((len(self.exteriorNodes),self.nVessels + self.nNodes))
        for i,node in enumerate(self.exteriorNodes):
            boundary_conditions[i,node] = 1
            
        B = np.concatenate([poiseuille, interior_flow_balance, boundary_conditions],axis=0)
        
        X = self.inc@np.diag(np.sign(q))
        exterior = np.array((np.abs(self.inc).sum(axis=1) == 1)).flatten()    
        interior = np.array((np.abs(self.inc).sum(axis=1) == 3)).flatten()    
        net_out = np.array(X.sum(axis=1) == -1).flatten()
        net_in = np.array(X.sum(axis=1) == 1).flatten()    
        
        C = self.inc[self.interiorNodes,:]@np.diag(q)
        
        div = np.where(np.bitwise_and(interior,net_out))[0]
        for node in div:
            idx_in = np.where(X[node,:] > 0)[0][0]
            idx_out = np.min(np.where(X[node,:] < 0)[0])
            row = np.zeros(self.nVessels)
            row[idx_in] = -skimming.skimming_kj(np.abs(q[idx_out]/q[idx_in]),self.pPlasma)[0]
            row[idx_out] = 1
            C = np.concatenate((C,row.reshape(1,-1)))  
        
        inlets =  np.where(np.bitwise_and(exterior,net_out))[0]
        for i,inlet in enumerate(inlets):
            row = np.zeros(self.nVessels)
            inlet_vessel = np.where(self.inc[inlet])[0][0]            
            
            row[inlet_vessel] = 1
            C = np.concatenate((C,row.reshape(1,-1)))                 
        
        D = np.zeros((B.shape[0],C.shape[1]))
        E = np.zeros((C.shape[0],B.shape[1]))
        
        D[:self.nVessels,:self.nVessels] = np.diag(self.delta*q*r)
       
        X = self.inc@np.diag(np.sign(q))
        interior = np.array((np.abs(self.inc).sum(axis=1) == 3)).flatten()    
        net_out = np.array(X.sum(axis=1) == -1).flatten()
        
        div = np.where(np.bitwise_and(interior,net_out))[0]
        n_div = len(div)
        
        dsdh = np.zeros((n_div,self.nVessels))
        dsdq = np.zeros((n_div,self.nVessels))
        for i,node in enumerate(div):
            idx_in = np.where(X[node,:] > 0)[0][0]
            idx_out = np.min(np.where(X[node,:] < 0)[0])
           
            q_in = q[idx_in]
            q_out = q[idx_out]
        
            dsdh[i,idx_in] = -skimming.skimming_kj(np.abs(q_out/q_in),self.pPlasma)[0]
            dsdh[i,idx_out] = 1
            
            dfdq = skimming.skimming_kj_dq(np.abs(q_out/q_in),self.pPlasma)[0]
            
            dsdq[i,idx_in] = h[idx_in]*dfdq*np.abs(q_out)/(q_in**2)*np.sign(q_in)
            dsdq[i,idx_out] = -h[idx_in]*dfdq/np.abs(q_in)*np.sign(q_out)

        C[self.nInteriorNodes:self.nInteriorNodes + n_div, :] = dsdh            
            
            
        E[:self.nInteriorNodes,self.nNodes:] = self.inc[self.interiorNodes,:]@np.diag(h)            
        E[self.nInteriorNodes:self.nInteriorNodes + n_div, self.nNodes:] = dsdq
        
        J = np.block([[B,D],[E,C]])
        
        return J
    
    def get_hessian_tensor(self,x):
        d = 1e-3
        
        H = []
        for i in range(len(x)):
            x1 = x.copy()
            x2 = x.copy()
            x1[i] += -d
            x2[i] += d
            
            J1 = self.get_jacobian_from_full_state(x1)
            J2 = self.get_jacobian_from_full_state(x2)
            DJ = (J2 - J1)/2/d
            H.append(np.expand_dims(DJ,2))
            
        H = np.concatenate(H,axis=2)
        
        return H
        
    def equilibrium_relation(self,h):
        self.set_state(h)
        
        return h - np.linalg.solve(self.cq,self.cq_rhs)
    
    def fold_relation(self,h):
        self.set_jacobian(h)
        
        return(np.linalg.det(self.jacobian))
    
    def isola_center_relation(self,h):
        self.set_jacobian(h)
        
        L,V = sp.linalg.eig(self.jacobian.T)
        
        i = np.argmin(np.abs(L))
        psi = V[:,i]

        return np.real(psi[self.nVessels + self.nInteriorNodes])*10
        
        
    #########################
    ##### ISOLA METHODS #####
    #########################
    
    def isola_stuff(self,x):
        J = self.get_jacobian_from_full_state(x)
        
        test_lambda = np.zeros(J.shape[1])
        test_lambda[0] = 1.0

        test_tau = np.zeros(J.shape[1])
        test_tau[self.nNodes + self.nVessels + 1] = 1.0

        for i in range(J.shape[0]):
            if (J[i,:] == test_lambda).all():
                idx_lambda = i
            elif (J[i,:] == test_tau).all():   
                idx_tau = i
        
        L,V = sp.linalg.eig(J)
        i = np.argmin(np.abs(L))
        phi = V[:,i]
        
        L,V = sp.linalg.eig(J.T)
        i = np.argmin(np.abs(L))
        psi = V[:,i]       
        
        F_lambda = np.zeros(J.shape[0])
        F_lambda[idx_lambda] = -1
        
        F_tau = np.zeros(J.shape[0])
        F_tau[idx_tau] = -1
       
        
        K = np.concatenate((J,100*phi.reshape((-1,1)).T),axis=0)          
        Z1 = np.linalg.lstsq(K,np.append(-F_lambda,0))[0]
        
        K = np.concatenate((J,psi.reshape((-1,1)).T),axis=0)          
        Z2 = np.linalg.lstsq(K,np.append(-F_tau,0))[0]
                             
        print(Z1)
        print(Z2)
        print()
        
        print(np.dot(Z1,phi))
        print(np.dot(Z2,phi))
                            
        print(J@Z1)
        print(J@Z2)
        
        H = self.get_hessian_tensor(x)
        
        a = np.dot(psi,H@phi@phi)
        b = np.dot(psi,H@phi@Z1)
        c = np.dot(psi,H@Z1@Z1)
        d = np.dot(psi,F_tau)
        
        
        tau_1 = 0
        tau_2 = np.sign(-a*d)
                
#         acot = lambda z: 1j/2*(np.log((z-1j)/z) - np.log((z+1j)/z))
#         beta = 1/2*acot((a-c)/2/b)
        
        
        return (H,phi)
       
            
    ###################################################        
    ######## TRANSIT TIME DISTRIBUTION METHODS ########
    ###################################################    
            
    def directed_adj_dict(self):
        q = self.pq[self.nNodes:]
        A = {}
        for i in range(len(q)):
            v0 = min(self.e[i])
            v1 = max(self.e[i])
            if q[i] > 0:
                A[v0] = A.get(v0,[]) + [v1]
            else:
                A[v1] = A.get(v1,[]) + [v0]

        for key,item in A.items():
            A[key] = set(item)

        for i in set(range(int(self.adj.shape[0]))) - set(A.keys()):
            A[i] = set([])

        self._adj_dict = A
     
    def get_paths_by_node_from_inlet(self,inlet):
        stack = [(inlet,[inlet])]
        paths = []
        while stack:
            (vertex, path) = stack.pop()

            if len(self._adj_dict[vertex]) == 0:
                paths.append(path)
            for next in self._adj_dict[vertex] - set(path):
                stack.append((next, path + [next]))            
    
        return paths
       
    def get_paths_by_node(self):
        self.directed_adj_dict()
        paths = []
        for i in self.exteriorNodes:
            new = self.get_paths_by_node_from_inlet(i)
            if len(new[0]) == 1:
                continue
            paths += new   
        self._paths_by_node = paths
    
    def get_paths_by_edge(self):
        self.get_paths_by_node()
        self._paths_by_edge = [[np.abs(self.inc)[[path[i],path[i+1]],:].sum(axis=0).argmax() for i in range(len(path)-1)] for path in self._paths_by_node]        

    def compute_conditional_probabilities_downstream(self):
        self._rbc = self.h*np.abs(self.q)   

        rbc_normalizer = np.zeros(self._rbc.shape)

        X = self.inc@np.diag(np.sign(self.q))
        for node in range(self.nNodes):
            row = X[node,:]
            if np.abs(row).sum() == 1:
                if row.sum() == -1:
                    vessel = np.where(row)[0][0]
                    rbc_normalizer[vessel] = np.sum(self._rbc[self.exteriorFlows])/2
            else:
                if row.sum() == -1:
                    inflow = np.where(row == 1)[0][0]
                    outflow_0 = np.where(row == -1)[0][0]
                    outflow_1 = np.where(row == -1)[0][1]                    
                    rbc_normalizer[outflow_0] = self._rbc[inflow]
                    rbc_normalizer[outflow_1] = self._rbc[inflow]                    
                elif row.sum() == 1:
                    outflow = np.where(row == -1)[0][0]
                    rbc_normalizer[outflow] = self._rbc[outflow]
        
        self._cond_prob = self._rbc/rbc_normalizer        
        
    def compute_ttd(self,verbose=False):
        self.compute_conditional_probabilities_downstream()
        if verbose:
            print(f'Cond prob: {self._cond_prob}')
        
        self.get_paths_by_edge()
        
        probs = []
        for path in self._paths_by_edge:
            probs.append(np.product(self._cond_prob[path]))

        if verbose: 
            for i,prob in enumerate(probs):
                print(f'P(RBC -> {self._paths_by_edge[i]}) \t= {np.round(prob,6)}')
            print(f'Check sum of total probability : {np.sum(probs)}')

        vol = np.pi*(self.d/2)**2*self.l
        tau = vol/np.abs(self.q)

        delays = []
        for path in self._paths_by_edge:
            delays.append(np.sum(tau[path]))

        if verbose: 
            for i,delay in enumerate(delays):
                print(f'{self._paths_by_edge[i]} :\t {delay}')
                
        ttd = TransitTimeDistribution(self._paths_by_edge, delays, probs)
#         if np.abs(np.sum(ttd.probs) - 1) > 1e-3:
#             print('Warning: Cumul. prob. of candidate TTD is not equal to 1!')
        self.ttd = ttd

    
    
    
    ###################################
    ###### VISUALIZATION METHODS ######
    ###################################    
    
    def plot(self,width=[],colors=[],directions=[],annotate=False,ms=10,x_min=0,x_max=1,y_min=0,y_max=1,annot_offset_x=[],annot_offset_y=[]):
        if len(colors) == 0: colors = len(self.e)*['k']
        if len(width) == 0: width = len(self.e)*[1]
        if len(annot_offset_x) == 0: annot_offset_x = len(self.e)*[0]
        if len(annot_offset_y) == 0: annot_offset_y = len(self.e)*[0]

        print(annot_offset_y)
        for i,edge in enumerate(self.e):
            i0 = min([edge[0],edge[1]])
            i1 = max([edge[0],edge[1]])        

            x0 = self.v[i0,0]
            y0 = self.v[i0,1]

            x1 = self.v[i1,0]
            y1 = self.v[i1,1]

            if not self.w[i]:
                plt.plot([x0, x1], [y0,y1],'-', c=colors[i], lw=width[i])
            else:
                if x1 > x0:
                    plt.plot([x0, x1-(x_max-x_min)], [y0,y1], '-', c=colors[i], lw=width[i])
                    plt.plot([x0+(x_max-x_min), x1], [y0,y1], '-', c=colors[i], lw=width[i])
                else:
                    plt.plot([x0-(x_max-x_min), x1], [y0,y1], '-', c=colors[i], lw=width[i])
                    plt.plot([x0, x1+(x_max-x_min)], [y0,y1], '-', c=colors[i], lw=width[i])                
            if len(directions):
                if directions[i] == 1:
                    if len(self.w):
                        if self.w[i] == 0:
                            plt.arrow(x0,y0,(x1-x0)/2,(y1-y0)/2,head_width=.2,lw=0)                    
                elif directions[i] == -1:
                    if len(self.w):
                        if self.w[i] == 0:
                            plt.arrow(x1,y1,(x0-x1)/2,(y0-y1)/2,head_width=.2,lw=0)

            if annotate == True: 
                if self.w[i] == 0:
                    plt.annotate(str(i),(x0+(x1-x0)/2+annot_offset_x[i],y0+(y1-y0)/2+annot_offset_y[i]),fontsize=16,ha='center',va='center')
                else:
                    plt.annotate(str(i),(np.max((x0,x1)),y1),fontsize=16,color='r',ha='center',va='center')

        for node in range(len(self.v)):
            x = self.v[node,0]
            y = self.v[node,1]
            plt.plot(x,y ,'wo',mec='k',ms=ms)
            x,y = self.v[node,0], self.v[node, 1]
            if annotate:
                plt.text(x,y,str(node),fontsize=12,ha='center',va='center')

        plt.gca().set_xticks([])
        plt.gca().set_yticks([])    

#         plt.xlim(x_min, x_max)
#         plt.ylim(y_min, y_max)        

        plt.gca().set_aspect('equal')                

    
    
    #########################
    ###### I/O METHODS ######
    #########################    
    
    def save(self,prefix):
        scipy.io.savemat(f'data/{prefix}_eqs.mat',{'eqs':self.equilibria})
        f = open(f'data/{prefix}_params.p','wb')
        pickle.dump(self.params,f)
        f.close()