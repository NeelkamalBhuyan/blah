# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 22:21:36 2021

@author: neelk
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import warnings
warnings.filterwarnings("ignore")

# 1. TimeGAN model
from timegan import timegan
# =============================================================================
# # 2. Data loading
# from data_loading import real_data_loading, sine_data_generation
# # 3. Metrics
# from metrics.discriminative_metrics import discriminative_score_metrics
# from metrics.predictive_metrics import predictive_score_metrics
# from metrics.visualization_metrics import visualization
# =============================================================================

import pandas as pd

import argparse
 
 
# Initialize parser
parser = argparse.ArgumentParser()
 
# Adding optional argument
parser.add_argument("--dataset", help = "Show Output")
 
# Read arguments from command line
args = parser.parse_args()

# =============================================================================
# chunk_data = pd.read_csv("C:\\Users\\neelk\\Desktop\Aalborg\original_datasets\\" + str(args.dataset) + ".csv")   
# ori_chunk_data = []
# for i in range(len(chunk_data.index)):
#     ori_chunk_data.append(list(chunk_data.iloc[i][1:]))
# 
# ori_chunk_data = np.asarray(ori_chunk_data)
# size = ori_chunk_data.shape
# ori_chunk_data = np.reshape(ori_chunk_data,(size[0], size[1],1))
# =============================================================================

chunk_data = pd.read_csv("C:\\Users\\neelk\\Desktop\Aalborg\original_datasets\chunk_throughput\\" + str(args.dataset) + ".csv")
ori_chunk_data = []
for i in range(len(chunk_data.index)):
    temp = []
    for j in range(1,len(chunk_data.iloc[i])):
        l = chunk_data.iloc[i][j].split(',')
        chunk = float(l[0][1:])
        #print(l[1][1:])
        throughput = float(l[1][1:-1])
        temp.append([chunk,throughput])
    ori_chunk_data.append(temp)

ori_chunk_data = np.asarray(ori_chunk_data)
size = ori_chunk_data.shape

parameters = dict()

parameters['module'] = 'gru' 
parameters['hidden_dim'] = 8
parameters['num_layer'] = 4
parameters['iterations'] = 10000
parameters['batch_size'] = 8

generated_data = timegan(ori_chunk_data, parameters)
print('Finish Synthetic Data Generation')

# =============================================================================
# data_out = np.reshape(generated_data,(generated_data.shape[0],generated_data.shape[1]))
# df_out = pd.DataFrame(data_out)
# df_out.to_csv("C:\\Users\\neelk\\Desktop\Aalborg\\generated_synthetic_data_long_form\\" + str(args.dataset) + ".csv")
# =============================================================================

data_out = generated_data[:,:,0]
df_out = pd.DataFrame(data_out)
df_out.to_csv("C:\\Users\\neelk\\Desktop\Aalborg\\generated_synthetic_data_long_form\chunk_throughput\\" + str(args.dataset) + "_chunks.csv")

data_out = generated_data[:,:,1]
df_out = pd.DataFrame(data_out)
df_out.to_csv("C:\\Users\\neelk\\Desktop\Aalborg\\generated_synthetic_data_long_form\chunk_throughput\\" + str(args.dataset) + "_throughputs.csv")
