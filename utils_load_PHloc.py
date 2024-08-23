## Utils file to load PH, critical points and critical sizes

import numpy as np
import matplotlib.pyplot as plt
import pickle

########### AUXILIARY FUNCTIONS

def test_increasing_dim(diagrams) :
    aux = diagrams[:,2]
    if np.abs(aux[1:] - aux[:-1]).sum() > 2  :
        raise Exception('diagrams not ordered by increasing dim')
        
def nbpts_PH(diagrams) :
    test_increasing_dim(diagrams)
    ph0 = sum(np.isclose(diagrams[:,2],0))
    ph1 = sum(np.isclose(diagrams[:,2],1))
    ph2 = sum(np.isclose(diagrams[:,2],2))
    return ph0, ph1, ph2

def get_PH_alldims(diagrams, threshold):
    # discard low-persistence points
    diagrams = diagrams[diagrams[:,1] >= diagrams[:,0] + threshold]
    
    ph0, ph1, ph2 = nbpts_PH(diagrams)
    PH0 = diagrams[:ph0, :2]
    PH1 = diagrams[ph0:ph0+ph1, :2]
    PH2 = diagrams[ph0 + ph1:, :2]
    return [PH0, PH1, PH2]

### keops (fast) or sklearn (slow) to obtain density heatmaps

import torch
from pykeops.torch import LazyTensor

def gaussian_kde_keops(X, Y, weights = None, sigma = 1) : 
    ''' 
    Estimate gaussian density of X (nb of points, D) evaluated on points Y.
    Using the keops library makes sense only if you have a high number of points in the persistence diagrams,
    otherwise use any other gaussian KDE method to estimate the same.
    There is a compiling time for keops the first time you define the symbolic formula.
    '''
    
    M = X.shape[0]
    N = Y.shape[0]
    if len(X.shape) == 1 :
        D = 1
    else :
        D = X.shape[1]
    X = torch.tensor(X)
    Y = torch.tensor(Y)

    x_i = LazyTensor(X.view(M, 1, D))
    y_j = LazyTensor(Y.view(1, N, D))

    D_ij = ((x_i - y_j)**2).sum(dim=2) 
    K_ij = (- D_ij / (2 * sigma**2)).exp()
    
    if weights is not None :
        weights = torch.tensor(weights)
        w_i = LazyTensor(weights.view(-1,1,1))
        K_ij = w_i * K_ij 
        sum_w_i = weights.sum()
    else :
        sum_w_i = M
    
    b_j = K_ij.sum(dim=0).view(-1) # torch.FloatTensor
    
    divide_factor = 2*torch.tensor(np.pi)*sigma**2 * sum_w_i

    res = b_j.numpy()
    res = res / divide_factor.numpy()
    #res = res / res.sum() #raise Exception('not proba distrib!')
    return res


from sklearn.neighbors import KernelDensity

def gaussian_kde_sklearn(X, Y, weights = None, sigma = 1) : 
    kde = KernelDensity(kernel='gaussian', bandwidth=sigma).fit(X, sample_weight = weights)
    log_dens = kde.score_samples(Y) # log-density!!!!!!!
    res = np.exp(log_dens)
    return res


### Functions to extract heatmaps without plotting

def extract_heatmap_grid(X, weights = None, sigma = 1, 
                  xlim = None, ylim = None, nb_bins_per_side = 100,
                   option = 'sklearn')   :
    'gives the np.array corresponding to the density heatmap of the point cloud X'
    'NO PLOT INVOLVED HERE'  
    # X shape (Npoints, 2)

    II,JJ = np.mgrid[ylim[1] : ylim[0] : nb_bins_per_side * 1j,
                     xlim[0] : xlim[1] : nb_bins_per_side * 1j]
    
    if len(X) == 0 :
        return np.zeros(II.shape)
    
    y = II.flatten()
    x = JJ.flatten()
    Y = np.vstack((x,y)).T
    Y = np.ascontiguousarray(Y)
    
    if option == 'sklearn' :
        z = gaussian_kde_sklearn(X,Y, weights = weights, sigma = sigma)
    elif option == 'keops' :
        z = gaussian_kde_keops(X,Y, weights = weights, sigma = sigma)
    
    img = z.reshape(II.shape)
    return img


def extract_PH_heatmap(diagrams, dim, weights = True, sigma = 1,
                             xlim = None, ylim = None, nb_bins_per_side = 100,
                      option = 'sklearn'):

    ph0 = len(np.where(diagrams[:,2] == 0)[0])
    ph1 = len(np.where(diagrams[:,2] == 1)[0])
    ph2 = len(np.where(diagrams[:,2] == 2)[0])

    if dim == 0 :
        diag = diagrams[:ph0, :2]
    if dim == 1 :
        diag = diagrams[ph0:ph0+ph1, :2]
    if dim == 2 :
        diag = diagrams[ph0 + ph1:, :2]

    if weights :
        weights = diag[:,1] - diag[:,0] # persist
    else :
        weights = None
    img = extract_heatmap_grid(diag, weights = weights, sigma = sigma,
                        xlim = xlim, ylim = ylim, nb_bins_per_side = nb_bins_per_side,
                         option = option)
    return img

        
def extract_PH_heatmaps_single_sample(diagrams, THR, WEIGHTS, SIGMA, XLIMS, YLIMS, NB_BINS_PER_SIDE,
                                     discard_PH0_NW = True, option = 'sklearn') :

    test_increasing_dim(diagrams) # check for errors

    if discard_PH0_NW :
        # discard values in PH0 NW (disconnected components)
        belongs_PH0_NW = np.isclose(diagrams[:,2],0) * (diagrams[:,0] < 0) * (diagrams[:,1] > 0)
        diagrams = diagrams[belongs_PH0_NW == False]
        print('excluded {} pts belonging to PH0 NW'.format(sum(belongs_PH0_NW)))

    # discard not persistent values
    diagrams = diagrams[diagrams[:,1] >= diagrams[:,0] + THR]
    print('excluded low-persistence points')

    PH_heatmaps = []
    for dim in range(3) :
        
        img = extract_PH_heatmap(diagrams, dim, weights = WEIGHTS, sigma = SIGMA,
        xlim = XLIMS[dim], ylim = YLIMS[dim], nb_bins_per_side = NB_BINS_PER_SIDE, option = option)

        PH_heatmaps += [img]
        
    PH_heatmaps = np.array(PH_heatmaps)
    return PH_heatmaps

### Function to plot pre-extracted heatmaps

def contour_heatmaps(PH_heatmaps, XLIMS, YLIMS) :
    
    fig, axes = plt.subplots(1,3,figsize = (8, 3))
    
    for dim in range(3) :
        ax = axes[dim]
        
        img = PH_heatmaps[dim]
        
        xlim = XLIMS[dim] ; ylim = YLIMS[dim]
        EXTENT = [xlim[0], xlim[1], ylim[0], ylim[1]]
        
        ax.imshow(img, extent = EXTENT, cmap = 'magma')

        ax.contour(img, levels = 4, extent = EXTENT, #img.max()*np.arange(1,5)/5
                   origin = 'upper', cmap = 'inferno_r')
        
        if False :
            ax.axhline(y=0, color='k', linewidth = 1, alpha = 0.8)
            ax.axvline(x=0, color='k', linewidth = 1, alpha = 0.8)

            idlim_min = min(xlim[0], ylim[0])
            idlim_max = max(xlim[1], ylim[1])
            ax.plot([idlim_min, idlim_max],[idlim_min, idlim_max], 
                    c = 'k',alpha = .8, linewidth = 1, label='_nolegend_')

    for dim in range(3) :
        axes[dim].set_xlabel('PH{}'.format(dim))

    plt.tight_layout()
    ##if save_fig : plt.savefig(figure_path, dpi = 200) ; print('saved in', figure_path)
    plt.show()   
    

## Functions to collect critical sizes (in paired form or individual form) from PH diagrams
    
def collect_paired_crit_sizes(diagrams, threshold = 0, trunc_mag = None) :
    
    diagrams = diagrams[diagrams[:,1] >= diagrams[:,0] + threshold]
    
    PH0, PH1, PH2 = get_PH_alldims_Qi(diagrams, threshold)
    
    X0,Y0 = PH0.T
    X1,Y1 = PH1.T
    X2,Y2 = PH2.T
    
    if trunc_mag is not None :    
        # discard any couple of critical values such that one of them is of magnitude greater than trunc_mag
        PH0 = PH0[ (np.abs(X0) < trunc_mag) * (np.abs(Y0) < trunc_mag) ]
        PH1 = PH1[ (np.abs(X1) < trunc_mag) * (np.abs(Y1) < trunc_mag) ]
        PH2 = PH2[ (np.abs(X2) < trunc_mag) * (np.abs(Y2) < trunc_mag) ]
        
        X0,Y0 = PH0.T
        X1,Y1 = PH1.T
        X2,Y2 = PH2.T

    # PH0 SW (-r0, -r1)
    r0_r1 = -PH0[Y0 < 0].T
    
    # PH0 NW (-r0, g1)
    r0_g1 = PH0[(X0 < 0) * (Y0 > 0)].T ; r0_g1[0] *= -1
    
    # PH1 SW (-r1, -r2)
    r1_r2 = -PH1[Y1 < 0].T
    
    # PH1 NW (-r1, g2)
    r1_g2 = PH1[(X1 < 0) * (Y1 > 0)].T ; r1_g2[0] *= -1
    
    # PH1 NE (g1, g2)
    g1_g2 = PH1[X1 > 0].T
    
    # PH2 NW (-r2, g3)
    r2_g3 = PH2[(X2 < 0) * (Y2 > 0)].T ; r2_g3[0] *= -1
    
    # PH2 NE (g2, g3)
    g2_g3 = PH2[X2 > 0].T
    
    # checking that birth <= death in all quadrants (no need to check for NW quadrant in PH0, PH1, PH2)
    cond = np.all(r0_r1[0] >= r0_r1[1])
    cond *= np.all(r1_r2[0] >= r1_r2[1])
    cond *= np.all(g1_g2[0] <= g1_g2[1])
    cond *= np.all(g2_g3[0] <= g2_g3[1])
    if not cond :
        raise Exception('Failed checking pairings')
        
    return r0_r1, r0_g1, r1_r2, r1_g2, g1_g2, r2_g3, g2_g3

def collect_indiv_crit_sizes(diagrams, threshold = 0, trunc_mag = None) :
    
    r0_r1, r0_g1, r1_r2, r1_g2, g1_g2, r2_g3, g2_g3 = collect_paired_crit_sizes(diagrams, threshold = threshold, trunc_mag = trunc_mag)
    
    r0 = np.hstack((r0_r1[0], r0_g1[0]))
    r1 = np.hstack((r0_r1[1], r1_r2[0], r1_g2[0]))
    r2 = np.hstack((r1_r2[1], r2_g3[0]))
    g1 = np.hstack((r0_g1[1], g1_g2[0]))
    g2 = np.hstack((r1_g2[1], g1_g2[1], g2_g3[0]))
    g3 = np.hstack((r2_g3[1], g2_g3[1]))
    
    return r0,r1,r2,g1,g2,g3


########### DATASETS INFO (UPDATED 08/06/2023 with inclusion of second patient P2)

datasets = {}
datasets['CTRL'] = [26,38,59,79]
datasets['U937'] = [37,58,72,67,31,36,57]
datasets['HL60'] = [56,44,65]
datasets['P1'] = [68,66,73,77,55,27]
datasets['P2'] = [60,74,42]
datasets['MNC'] = [71,76,61,80]

list_levels = {'CTRL' : [0, 0, 0, 0],
'U937' : [ 1,  1 , 7 , 8, 10, 10 ,10],
'HL60' : [23, 25 ,25],
'P1' : [10, 40, 44, 51, 60, 76],
'P2' : [59, 88, 90],
'MNC' : [53, 67, 75, 86] }

datasets_of_interest = [26,38,59,79,
                        37,58,72,67,31,36,57,
                        56,44,65,
                        68,66,73,77,55,27,
                        60,74,42,
                        71,76,61,80]

phases_of_interest = [0,0,0,0,
         1,1,1,1,1,1,1,
         1,1,1,
         1,2,2,2,2,1,
         2,2,2,
         1,2,2,2]

phases_labels = ['O','O','O','O',
         'I','I','I','I','I','I','I',
         'I','I','I',
         'I','II','II','II','II','I',
         'II','II','II',
         'I','II','II','II']

levels_of_interest = [0, 0, 0, 0,
                      1,  1,  7,  8, 10, 10, 10,
                      23, 25, 25,
                      10, 40, 44, 51, 60, 76,
                      59, 88, 90,
                      53, 67, 75, 86]

injected_datasets_of_interest = ['CTRL','CTRL','CTRL','CTRL',
                                'U937','U937','U937','U937','U937','U937','U937',
                                'HL60','HL60','HL60',
                                'P1','P1','P1','P1','P1','P1',
                                 'P2','P2','P2',
                                'MNC','MNC','MNC','MNC']
