import numpy as np
import pandas as pd
import re
import time
import os
from datasketch import MinHash, MinHashLSHForest

def preprocess_text(text):
    text=re.sub(r'[^\w\s]','',text)
    tokens=text.lower()
    tokens=re.split(r'[\s_]',tokens)
    return tokens

def get_forest(data,perms):
    start_time=time.time()
    minhash=[]
    
    m=MinHash(num_perm=perms)
    for entity in data:
        tokens=preprocess_text(entity)
        m=MinHash(num_perm=perms)
        for s in tokens:
            m.update(s.encode('utf8'))
        minhash.append(m)    
    
    forest=MinHashLSHForest(num_perm=perms)
    for i,m in enumerate(minhash):
        forest.add(i,m)
    
    forest.index()
    print('It took {} seconds to build forest.'.format(time.time()-start_time))
    return forest

def predict(query_entity,entities,perms,num_results,forest):    
    tokens=preprocess_text(query_entity)
    m=MinHash(num_perm=perms)
    for s in tokens:
        m.update(s.encode('utf8'))
    
    idx_array=np.array(forest.query(m,num_results))
    if len(idx_array)==0:
        return None
    
    results=[]
    for idx in idx_array: results.append(entities[idx])
    return results