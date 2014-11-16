"""
GNU GENERAL PUBLIC LICENSE Version 2

Created on Thu Nov 13 22:47:53 2014

@author: Wasit
"""
import os
import pickle
from sctree import tree
from ss import dataset
import numpy as np
rootdir="ss"
#load the trees

def loadForest():
    npic=0
    forest=[]
    for root, dirs, files in os.walk(rootdir):
        for f in files:
            if f.endswith('pic'):
                npic=npic+1
                print f
                #reading the tree pickle file
                pickleFile = open(rootdir + "/" + f, 'rb')
                root = pickle.load(pickleFile)
                pickleFile.close()
                #init the test tree
                t=tree()
                t.settree(root)
                forest.append(t)
    return forest,npic



#init
forest,npic=loadForest()
dset=dataset()
#compute recall rate
ntree=npic
p=np.zeros(len(forest[0].getP([0],dset)))
correct=0;
for x in xrange(dset.size):
    cL=dset.getL(x)
#print prob
    for i in xrange(ntree):
        p=p+forest[i].getP(np.array([x]),dset)
    p=p/np.sum(p)
    ids = p.argsort()[::-1][:11]
    L=ids[0]        
    
#print max likelihood
    #L=t.getL(np.array([x]),dset)
    
    if(any(ids==cL)):
        correct=correct+1
        if cL!=0:
            print("\n%03d: correct L"%cL)
            for i in xrange(len(ids)):
                print("%03d_%03d"%(ids[i],100*p[ids[i]])),

    dset.setL(x,L)
print("\n--> recall rate: {}%".format(correct/float(dset.size)*100))