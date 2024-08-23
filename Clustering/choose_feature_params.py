# Choose parameters for feature maps

ns = 5 # 5
# number of YX slices in each sample: choose odd number
if ns % 2 != 1 :
    raise ValueError('nb_slices should be odd integer')
        
anatomy = 'both' 

# COMPUTE FEATURES : only accept knee / long
# 'both', 'long' #'knee'
# does Kmeans clustering on feature vectors involve only knees, only longs, or entire bones?

threshold,trunc_mag,radius,step = .5,None,40,5

# .5,15,50,5
# .5,20,50,5 # maybe trunc_mag 20 too large and includes outliers
# .5,20,25,3 # radius too small: tiny spaced non nan neighborhoods in clustering
# .1,10,50,5 # good for middle slice (ns = 1)

use_keys = 'no_weight'

# 'cust5' for only 5 specific features : ['R mean' 'r mean' 'L mean' '1-l/L mean' 'S mean']
# 'no_weight' for 15 features

Kmethod = 'center_norma'


# '' : use raw features
# 'rank' : "equalize" by attributing to each feature its integer ranking after ordering all values of that feature in data
# 'center_norma'

####### parameters determine other variables
import numpy as np

anat_list = [anatomy] if anatomy in ['knee','long'] else ['knee','long']

if use_keys == 'no_weight' :
    feature_names = np.array(['R mean', 'R std', '1-r/R mean', '1-r/R std', 
                     'r mean', 'r std', 'L mean', 'L std', 'L/r mean', 'L/r std', 
                     '(r,L) std', '1-l/L mean', '1-l/L std', 'S mean', 'S std'])
elif use_keys == 'cust5' :
    feature_names = np.array(['R mean', 'r mean', 'L mean', '1-l/L mean', 'S mean'])
#feature_8_names = feature_names[[0,4,5,6,7,8,12,13]]
#print(feature_8_names)

## Kmeans folder

if use_keys == 'no_weight' : aux = ''
elif use_keys == 'cust5' : aux = '_' + use_keys
if Kmethod == '' : aux_Kmethod = ''
else : aux_Kmethod = '_' + Kmethod

from datasets_info_Qi import PH_folder
Kmeans_folder = PH_folder + 'feature_maps/Kmeans_{}{}{}/'.format(anatomy,aux,aux_Kmethod)
Kmeans_inside_folder = PH_folder + 'feature_maps/Kmeans_inside_{}{}{}/'.format(anatomy,aux,aux_Kmethod)

