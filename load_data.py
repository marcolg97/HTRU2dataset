# -*- coding: utf-8 -*-
"""

@author: Giacomo Vitali and Marco La Gala

"""

import numpy as np
import utils

def load(filename):
    D_list = []
    L_list = []
    with open(filename, 'r') as f:
        for line in f:
            attrs = line.split(',')
            if attrs[0] != '\n':
                for i in range(len(attrs)-1):
                    # Without float I have ['140.5625' '102.5078125' '136.75' ...]
                    # instead of [ 1.40562500e+02  1.02507812e+02  1.36750000e+02 ...]
                    attrs[i] = float(attrs[i])
                # Without rstrip labels are ['0\n' '0\n' '0\n' ... '0\n' '0\n' '0\n']
                attrs[-1] = int(attrs[-1].rstrip('\n')) 
                # Now create a 1-dim array and reshape it as a column vector,
                # then append it to the appropriate list
                D_list.append(utils.mcol(np.array(attrs[0:8])))
                L_list.append(attrs[-1])
    # From column vector we create a matrix stack horizontally the column vector
    dataset_matrix = np.hstack(D_list[:])
    # 1-dim array with class values
    class_label_array = np.array(L_list)
    return dataset_matrix, class_label_array