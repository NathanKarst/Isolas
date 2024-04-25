import numpy as np


class Continuer():
    def __init__(self,relation,init,step,options,jacobian=None):
        self.F = relation
        self.y0 = init
        self.t0 = np.zeros(init.shape)
        self.t0[-1] = 1
        self.step = step
        self.options = options
        self.jacobian = jacobian
        
    def get_continuation(self):
        print('Starting continuation...')
        Y1 = self.pseudo_arclength(direction=1)
        Y2 = self.pseudo_arclength(direction=-1)
        return np.concatenate((Y2[:,::-1],Y1),1)
    
    def pseudo_arclength(self,direction=1):
        # discrete deriv. delta x
        d = 1e-12
        maxStep = self.step
        y = self.y0.copy()     # w/o this, init will be changed in memory and will affect later continuations from same init
        t = self.t0.copy()*direction # direction = -1 will go the other way

        n = np.size(y)
        Y = np.array([y]).T
        T = np.array([t]).T

        numSteps = 0
        
        step = self.step/len(y)
        D = Differentiator(self.F,d)
        while numSteps < self.options.maxSteps:
            if numSteps%1000 == 0:
                print('\tSteps: ', numSteps)

            step = min(1.001*step,maxStep)
            y += step*t;

            I = np.eye(n)
            dy = 1;
            count = 0

            while np.linalg.norm(dy) / np.linalg.norm(y) > 1e-3:
                count += 1
                
                if not self.jacobian:
                    J = np.zeros((n,n))

                    if count > 400:
                        if abs(step) < 1e-9:
                            print('\tFailed to converge!')            
                            return Y
                        y = Y[:,-1].copy()
                        t = T[:,-1].copy()
                        step = step*0.5
                        continue            

                    y1 = [y + d*I[:,i] for i in range(n)]
                    y2 = [y - d*I[:,i] for i in range(n)]

                    res = map(D, zip(y1,y2))
                    tmp = np.array(np.vstack(tuple(res))).transpose()
                    J[0:n-1,:] = tmp
                    J[n-1,:] = t.T
                else:
                    J = self.jacobian(y[:-1])
                    print(J.shape)
                
                f = self.F(y)
                g = np.dot(t, y - Y[:,-1]) - step
                dy = -np.linalg.solve(J,np.append(f,g))
                
                if np.isnan(g).any() or np.isnan(f).any() or np.isnan(J).any() or np.isnan(y).any():
                    if abs(step) < 1e-9:
                        print('\tNaN repeatedly encountered!')            
                        return Y
                    y = Y[:,-1].copy()
                    t = T[:,-1].copy()
                    step = step*0.5
                    continue                 
                
                y += dy

            if (y > self.options.maxValues).any() or (y < self.options.minValues).any():
                print('\tContinuation: Maxed or mined out variable.')
                return Y

            J[n-1,:] = t.T
            t = np.linalg.solve(J,np.append(np.zeros(n-1),1))
            t = t/np.linalg.norm(t)

            Y = np.concatenate((Y,np.array([y]).T),1)
            T = np.concatenate((T,np.array([t]).T),1)
            numSteps += 1

        return Y 

class Differentiator:
    def __init__(self,F,d):
        self.F = F
        self.d = d
    
    def __call__(self,tup):
        return (self.F(tup[0]) - self.F(tup[1]))/(2*self.d)
        
class Options:
    def __init__(self,maxSteps,minValues,maxValues):
        self.maxSteps = maxSteps
        self.maxValues = maxValues
        self.minValues = minValues