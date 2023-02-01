import numpy as np
import torch

def select_grads(grads, method='fed_avg'):
        '''performs aggregation'''
        if method == 'fed_avg':
            '''classical average aggregation'''
            return grads, None
        
        elif method[:4] == 'NDC_':
            '''norm tresholding aggregation'''
            try:
                M = float(method[4:])
            except:
                raise TypeError("ill-defined NDC treshold")

            for i in range(len(grads)):
                norm_client = torch.norm(grads[i])
                for j in range(len(grads[i])):
                    normalize = max(1, norm_client.item() / M)
                    grads[i][j] = grads[i][j] / normalize

            return grads, None
        
        elif method[:11] == 'multi_krum_':
            '''multi-krum aggregation'''
            try:
                f = int(method[11:])
            except:
                raise TypeError("ill-defined byzantine worker count")
            if method[11:].isdigit() == False:
                raise TypeError("f should be an int")
            if 2*f+2 > len(grads):
                raise ValueError("f should be smaller")

            score = np.empty(0)
            for v1 in grads:
                dist = 0
                tracking = []
                for v2 in grads:
                    diff = torch.norm(v1-v2).item()**2
                    dist += diff
                    tracking.append(diff)
                dist -= sum(sorted(tracking, reverse=True)[:f+1])
                score = np.append(score, dist)

            idx = list(np.argpartition(score, len(grads)-f))
            return None, idx[len(grads)-f:]

        elif method == 'RFA':
            '''median aggregation using smoothed Weiszfeld algorithm '''
            print("median aggregation")
            raise(NotImplementedError)
            v = 0.00001
            R = 100
            grads_avg = [0] * len(grads[0])
            print(f"MA; v = {v}")

            for r in range(R):
                nc2 = [1 / torch.max(v, np.linalg.norm(grads[i] - np.array(grads_avg))) for i in range(len(grads))]
                grads_avg = torch.average(grads, axis=0, weights=nc2)
                    
            return grads_avg, 
        
        elif method[:14] == 'trimmed_means_':
            '''trimmed-means aggregation'''
            print("trimmed-means aggregation")
            raise(NotImplementedError)
            try:
                b = float(method[14:])
            except:
                raise TypeError("ill-defined trimmed-means treshold")
            if b >= 0.5:
                raise ValueError("b should be less than 0.5")
            print(f"TM; b = {b}")

            nb_to_remove = int(len(w)*b)
            print(2*nb_to_remove)

            w_temp = []
            nc_trimmed = []
            w_avg = []

            for i in range(len(w[0])):
                w_column = list(np.array(w)[:,i])
                nc_temp = list(nc)
                for b in range(nb_to_remove):
                    imax = w_column.index(np.max(w_column)) 
                    w_column.pop(imax)
                    nc_temp.pop(imax)
                    imin = w_column.index(np.min(w_column))
                    w_column.pop(imin)
                    nc_temp.pop(imin)

                w_temp.append(w_column)
                nc_trimmed.append(nc_temp)
                w_avg.append(np.average(w_temp[i], axis=0, weights=nc_trimmed[i]))

            print(w_avg)
            self.update_global_model(w_avg)
            print(self.global_model)
            print(self.global_model.get_weights())
            return copy.deepcopy(self.global_model)
            
        else:
            raise ValueError("unknown aggregation method")
