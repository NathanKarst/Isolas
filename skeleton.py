import numpy as np
import scipy.spatial
import scipy.io

class NetworkSkeleton:
    def __init__(self):
        pass
    
    def get(self):
        return {'v':self.v, 'e':self.e, 'l':self.l, 'd':self.d, 'w':self.w}
    
class PeriodicNetworkSkeleton(NetworkSkeleton): 
    def __init__(self,points,x_min,x_max,y_min,y_max):
        # create fake points to the left and the right in order to do periodic wrap
        offset_x = (x_max-x_min)
        w = np.array([points[:,0] - offset_x, points[:,1]]).T
        e = np.array([points[:,0] + offset_x, points[:,1]]).T
        points = np.concatenate([points,w,e])  

        # let scipy.spatial do the heavy lifting
        from scipy.spatial import Voronoi, voronoi_plot_2d
        vor = Voronoi(points)
        
        # initialize the three main return data structure
        vertices = np.copy(vor.vertices) 
        edges = []
        wrap = []


        new_vert_idx = len(vor.vertices)
        for ridge in vor.ridge_vertices:
            wrapFlag = False
            # -1 represents point at infinity 
            if -1 in ridge:
                continue

            # ridge has format [[x0,y0],[x1,y1]]; pull these (we'll need i0 and i1 later)
            i0 = ridge[0]
            x0 = vor.vertices[i0,0]
            y0 = vor.vertices[i0,1]

            i1 = ridge[1]
            x1 = vor.vertices[i1,0]
            y1 = vor.vertices[i1,1]    

            # booleans for whether each point is inside the domain of interest
            vt0_inside = (x_min <= x0 <= x_max) & (y_min <= y0 <= y_max)
            vt1_inside = (x_min <= x1 <= x_max) & (y_min <= y1 <= y_max)

            if not vt0_inside and not vt1_inside:
                continue
            if vt0_inside and not vt1_inside:
                # for points in the bottom right or bottom left, two of these can be activated
                if x1 > x_max and y1 < y_max:
                    d = [(vor.vertices[i,0] + offset_x - x1)**2 + (vor.vertices[i,1] - y1)**2 for i in range(len(vor.vertices))]
                    i1 = np.argmin(d)
                    if y1 > y_min:
                        wrapFlag = True
                if x1 < x_min and y1 < y_max:
                    d = [(vor.vertices[i,0] - offset_x - x1)**2 + (vor.vertices[i,1] - y1)**2 for i in range(len(vor.vertices))]
                    i1 = np.argmin(d)
                    if y1 > y_min: 
                        wrapFlag = True
                if y1 > y_max:
                    slope = (y1-y0)/(x1-x0)
                    x1_hat = (y_max - y0)/slope + x0
                    y1_hat = y_max

                    i1 = new_vert_idx
                    new_vert_idx += 1

                    vertices = np.concatenate([vertices,np.matrix([x1_hat,y1_hat])])
                if y1 < y_min:
                    slope = (y1-y0)/(x1-x0)
                    x1_hat = (y_min - y0)/slope + x0
                    y1_hat = y_min            

                    i1 = new_vert_idx
                    new_vert_idx += 1

                    vertices = np.concatenate([vertices,np.matrix([x1_hat,y1_hat])])            
            elif not vt0_inside and vt1_inside:
                if x0 > x_max and y0 < y_max:
                    d = [(vor.vertices[i,0] + offset_x - x0)**2 + (vor.vertices[i,1]-y0)**2 for i in range(len(vor.vertices))]
                    i0 = np.argmin(d)
                    if y0 > y_min:
                        wrapFlag = True                
                if x0 < x_min and y0 < y_max:
                    d = [(vor.vertices[i,0] - offset_x - x0)**2 + (vor.vertices[i,1]-y0)**2 for i in range(len(vor.vertices))]
                    i0 = np.argmin(d)
                    if y0 > y_min:
                        wrapFlag = True                
                if y0 > y_max:
                    slope = (y1-y0)/(x1-x0)
                    x0_hat = (y_max - y1)/slope + x1
                    y0_hat = y_max

                    i0 = new_vert_idx
                    new_vert_idx += 1

                    vertices = np.concatenate([vertices,np.matrix([x0_hat,y0_hat])])
                if y0 < y_min:
                    slope = (y1-y0)/(x1-x0)
                    x0_hat = (y_min - y1)/slope + x1
                    y0_hat = y_min            

                    i0 = new_vert_idx
                    new_vert_idx += 1

                    vertices = np.concatenate([vertices,np.matrix([x0_hat,y0_hat])])

            if [i0,i1] in edges or [i1, i0] in edges:
                continue

            edges.append([i0,i1])
            wrap.append(wrapFlag)
        wrap = np.array(wrap)
        
        self.v = vertices
        self.e = edges
        self.l = None
        self.d = None
        self.w = wrap
    
    def reduce_vertex_labels(self):
        vertices_idxs = (list(set([vertex for edge in self.e for vertex in edge])))
        vertices_idxs.sort()
        
        mapping = {}
        new_vertices = []
        for i,vertex_idx in enumerate(vertices_idxs):
            new_vertices.append(self.v[vertex_idx,:])
            mapping[vertex_idx] = i

        vertices = np.concatenate(new_vertices)
        edges = [[mapping[i],mapping[j]] for [i,j] in self.e]    

        self.v = vertices
        self.e = edges
    
    def reorder_vertex_labels(self):
        # reorder by y-coordinate for readability
        order = np.argsort(self.v[:,1].T).tolist()[0][::-1]
        mapping = {} 
        for i,v in enumerate(order):
            mapping[v] = i

        vertices = self.v[order,:]
        edges = [[mapping[i],mapping[j]] for [i,j] in self.e]

        self.v = vertices
        self.e = edges        
    
class PeriodicHoneycombSkeleton(PeriodicNetworkSkeleton):
    def __init__(self,m,n,pert,seed=None):
        self.seed = seed
        
        points = []
        for j in range(m):
            points += [(j,(i*2*np.sin(np.pi/3)+(j%2)*np.sin(np.pi/3))*2/3) for i in range(n)]

        if self.seed:
            np.random.seed(self.seed)
            
        points = np.array(points)
        points += np.random.random(np.shape(points))*pert
        points = np.array(points)
    
        N = ((n-1)*2*np.sin(np.pi/3)+np.sin(np.pi/3))*2/3
        
        super(PeriodicHoneycombSkeleton,self).__init__(points,0,m,0,N)
        
        super(PeriodicHoneycombSkeleton,self).reduce_vertex_labels()
        super(PeriodicHoneycombSkeleton,self).reorder_vertex_labels()
        
        l = np.zeros(len(self.e))
        for i in range(len(self.e)):
            v0 = self.v[self.e[i][0],:]
            v1 = self.v[self.e[i][1],:]
            if not self.w[i]:
                l[i] = np.sqrt((v0[0,0]-v1[0,0])**2 + (v0[0,1]-v1[0,1])**2)
            else:
                if v1[0,0] > v0[0,0]:
                    l[i] = np.sqrt((v0[0,0]-(v1[0,0]-m))**2 + (v0[0,1]-v1[0,1])**2)
                else:
                    l[i] = np.sqrt((v0[0,0]-m-v1[0,0])**2 + (v0[0,1]-v1[0,1])**2)
        if self.seed:
            np.random.seed(self.seed)    
            if pert > 0:          
                d = np.random.uniform(.9,1.1,len(self.e))
            else:
                d = np.ones(len(self.e))
        self.l = l
        self.d = d

class LadderNetworkSkeleton(NetworkSkeleton):
    def __init__(self,n):
        v = np.zeros((2*n+4,2))

        for i in range(n+2):
            v[2*i,:] = [0,n+1-i]
            v[2*i+1,:] = [1,n+1-i]

        e = []
        for i in range(n):
            e.append([2*i,2*i+2])
            e.append([2*i+1,2*i+3])
            e.append([2*i+2,2*i+3])

        e.append([2*n,2*n+2])
        e.append([2*n+1,2*n+3])

        l = np.ones(len(e))
        d = np.ones(len(e))

        w = len(e)*[False]

        self.v = v
        self.e = e
        self.l = l
        self.d = d
        self.w = w  

class RepeatedLadderSkeleton(NetworkSkeleton):
    def __init__(self,m,n):
        v = []
        e = []
        w = []
        for i in range((2*m+2)*(n+1)):
            v.append([i%(n+1),m-i//(n+1)])

            if i <= n:
                e.append([i,i+n+1])
                w.append(False)
            elif i > n and i < (2*m+2)*(n+1) - (n+1):
                e.append([i,i+n+1])
                w.append(False)
                if (i//(n+1))%2 == 1:
                    if i%2 == 0:
                        e.append([i,i+1])
                        w.append(False)
                else: 
                    if i%2 == 1:
                        if i%(n+1) == (-1)%(n+1):
                            e.append([i,i-n])
                            w.append(True)
                        else:
                            e.append([i,i+1])
                            w.append(False)
            elif i >= (m+4)*(n+1) - (n+1):
                pass


        for i in range((2*m+2)*(n+1) - 2*(n+1), (2*m+2)*(n+1) - (n+1)):
            e.append([i,i+n+1])
            w.append(False)

        v = np.array(v).astype(float)

        l = np.ones(len(e))
        d = np.ones(len(e))
        
        self.v = v
        self.e = e
        self.l = l
        self.d = d
        self.w = w
        
        
class ArcadeSkeleton(NetworkSkeleton):
    def __init__(self,n):
        v = []
        e = []
        w = []
        
        e = []
        for i in range(n-1):
            for j in range(2**i,2**(i+1)):
                e += [(j,2*j)]
                e += [(j,2*j+1)]

        e = [(0,1)] + e        

        f = [(i,i+2**(n-1)) for i in range(2**(n-1),2**n)]

        g = [(2**(n+1)-ee[1]-1, 2**(n+1)-ee[0]-1) for ee in e]
        g = g[::-1]

        e = e + f + f + g
        
        
        n_nodes = 2**(n+1)
        v = []
        for i in range(n_nodes):
            v += [(1,1)]
        
        self.v = np.array(v).astype(float)
        self.e = e
        self.l = np.ones(len(e))
        self.d = np.ones(len(e))
        self.w = [False]*len(e)
        
        
class PeriodicRandomSkeleton(PeriodicNetworkSkeleton):
    def __init__(self,n,seed=None):
        self.seed = seed
                
        if self.seed:
            np.random.seed(self.seed)           

        points = [(np.random.random(), np.random.random()) for i in range(n)]
        points = np.array(points)
        
        super(PeriodicRandomSkeleton,self).__init__(points,0,1,0,1)
        
        super(PeriodicRandomSkeleton,self).reduce_vertex_labels()
        super(PeriodicRandomSkeleton,self).reorder_vertex_labels()
        
        l = np.zeros(len(self.e))
        for i in range(len(self.e)):
            v0 = self.v[self.e[i][0],:]
            v1 = self.v[self.e[i][1],:]
            if not self.w[i]:
                l[i] = np.sqrt((v0[0,0]-v1[0,0])**2 + (v0[0,1]-v1[0,1])**2)
            else:
                if v1[0,0] > v0[0,0]:
                    l[i] = np.sqrt((v0[0,0]-(v1[0,0]-1))**2 + (v0[0,1]-v1[0,1])**2)
                else:
                    l[i] = np.sqrt((v0[0,0]-1-v1[0,0])**2 + (v0[0,1]-v1[0,1])**2)
        self.l = l
        self.d = np.random.uniform(.9,1.1,len(self.e)) 
