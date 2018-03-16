import os
import pandas as pd
import random

def load_data(filename):
    current_directory = os.getcwd()
    input_folder = current_directory + '/../Data'
    bitcoin_data = pd.read_csv(input_folder + '/' + filename + '.csv').sort_values(['Timestamp'])
    return bitcoin_data


#u_inp is the name of file from which the data needs to be sampled
#numInterval is the num of continuous intervals need to be returned
def randomSample(u_inp, numInterval):
     wholeData = load_data(u_inp)
     endIndex = random.randint(numInterval, wholeData.shape[0]) 
     beginIndex = endIndex - numInterval
     rawChunk=wholeData.iloc[beginIndex:endIndex]
     chunk=[]
     for index, row in rawChunk.iterrows():
       chunk.append(row.tolist())     
     return chunk


u_inp = raw_input("\nEnter file name to load\n")
chunk=randomSample(u_inp,5)
print(chunk)  
