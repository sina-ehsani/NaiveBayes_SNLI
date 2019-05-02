import os
import json
import numpy as np 

def savedata(datapre, name, dir = './processeddata/'):
    data = {}  
    data['left'] = datapre[0] 
    data['right'] = datapre[1]
    data['label'] = datapre[2].tolist()
    fout = gen_name(dir, name )
    if not os.path.exists(dir):
        os.mkdir(dir)
    
    with open(fout, 'w', encoding='utf-8') as writer:  
        json.dump(data, writer)
        
    
def gen_name(dir, name, suffix='json'):
    fname = '{}.{}'.format(name, suffix)
    return os.path.join(dir, fname)

def readfile(dataname , dir = './processeddata/'):
    fout = gen_name(dir, dataname )
    with open(fout) as json_file:  
        data = json.load(json_file)
        data_list=[data['left'],data['right'],np.array(data['label'])]
        data_tuple=tuple(data_list)
    return(data_tuple)