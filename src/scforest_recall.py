"""
GNU GENERAL PUBLIC LICENSE Version 2

Created on Thu Nov 13 22:47:53 2014

@author: Wasit
"""
import os
import pickle
from sctree import tree
from ss_recall import dataset
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


from PIL import Image
import matplotlib.pyplot as plt
#init
forest,npic=loadForest()
dset=dataset()
#compute recall rate
ntree=npic
clmax=len(forest[0].getP([0],dset))
correct=0;
for im in xrange(len(dset.imgf)):
    p=np.zeros(clmax)
    for j in xrange(dset.spi):
        x=im*dset.spi+j
        cL=dset.getL(x)
    #print prob
        for i in xrange(ntree):
            p=p+forest[i].getP(np.array([x]),dset)
        p=p/np.sum(p)
    ids = p.argsort()[::-1][:11]
    L=ids[0]        
    for j in xrange(dset.spi):
        x=im*dset.spi+j
        dset.setL(x,L)
#print max likelihood
    #L=t.getL(np.array([x]),dset)
    plt.figure(1)
    im=np.array(Image.open(dset.imgf[im]).convert('L'))
    plt.imshow(im)
    plt.set_cmap('gray')
    plt.show()
    
    plt.figure(2)
    print ("\n%s\n"%dset.imgf[im]),
    for i in xrange(len(ids)):
        print("[%03d]%03d "%(ids[i],100*p[ids[i]])),
    i=0    
    while 1:
        if i>=10:
            break
        if ids[i]!=0:
            plt.subplot(i/5,5,i)
            im_icon=np.array(Image.open("icons/%3d.jpg").convert('L'))
            plt.imshow(im)
            i=i+1
    plt.set_cmap('gray')
    plt.show()
    plt.figure(1)
    plt.ginput(1)