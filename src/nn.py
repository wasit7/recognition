# -*- coding: utf-8 -*-
"""
Created on Tue Dec 02 00:47:40 2014

@author: Wasit
"""

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import os
from scipy.spatial import kdtree
import cPickle
from scipy.ndimage import filters
import sys
import scipy.ndimage
sys.setrecursionlimit(10000)

rootdir="dataset_7"
wd=40

def G(x,mu,s):
    return 1.0/ np.sqrt(2.0*np.pi)*np.exp(((x-mu)**2)/(-2.0*s**2))
def normFFT(im_file):
    im=np.array(Image.open(im_file).convert('L'))
    #frequency domain
    #f=np.log(np.abs(np.fft.fftshift(np.fft.fft2(im))))
    f=np.log(np.abs(np.fft.fft2(im)))
    #scaling
    s=(200./f.shape[0],200./f.shape[1])
    #normalized frequency domian
    return scipy.ndimage.zoom(f,s,order = 2)
    
def getHash3(im_file):
    ci=10
    f=normFFT(im_file)
    
    rmax,cmax=f.shape    
    sg=np.zeros((2*wd,wd-ci))
    
    sg[0:wd,:]=f[rmax-wd:rmax,ci:wd]
    sg[wd:2*wd,:]=f[0:wd,ci:wd]
    filters.gaussian_filter(sg, (4,4), (0,0), sg)
    sg=np.reshape(sg,(1,-1))[0]
    sg.astype(np.float32)
    return sg
    fsg=np.zeros(wd)
    for b in xrange(wd):
        for r in xrange(wd):
            for c in xrange(wd):
                rad=np.sqrt(r**2+c**2)            
                fsg[b]=fsg[b]+sg[r,c]*G(rad,float(b),0.2)
        fsg[b]=fsg[b]/(np.pi*float(b+1.0))
        fsg=fsg/np.linalg.norm(fsg)
        fsg.astype(np.float32)
    return fsg
    
def getHash2(im_file):
    ci=10
    im = np.array(Image.open(im_file).convert('L'))
    f=np.fft.fft2(im)
    rmax,cmax=f.shape    
    sg=np.zeros((2*wd,wd-ci))
    
    sg[0:wd,:]=np.log(np.abs(f[rmax-wd:rmax,ci:wd]))
    sg[wd:2*wd,:]=np.log(np.abs(f[0:wd,ci:wd]))
    filters.gaussian_filter(sg, (4,4), (0,0), sg)
    sg=np.reshape(sg,(1,-1))[0]
    sg.astype(np.float32)
    return sg
    fsg=np.zeros(wd)
    for b in xrange(wd):
        for r in xrange(wd):
            for c in xrange(wd):
                rad=np.sqrt(r**2+c**2)            
                fsg[b]=fsg[b]+sg[r,c]*G(rad,float(b),0.2)
        fsg[b]=fsg[b]/(np.pi*float(b+1.0))
        fsg=fsg/np.linalg.norm(fsg)
        fsg.astype(np.float32)
    return fsg
def getHash(im_file):
    im = np.array(Image.open(im_file).convert('L'))
    f=np.fft.fft2(im)
    rmax,cmax=f.shape    
    sg_r=np.zeros((2*wd,wd))
    sg_r[0:wd,:]=np.abs(f.real[0:wd,0:wd])
    sg_r[wd:2*wd,:]=np.abs(f.real[rmax-wd:rmax,0:wd])
    sg_i=np.zeros((2*wd,wd))
    sg_i[0:wd,:]=np.abs(f.imag[0:wd,0:wd])
    sg_i[wd:2*wd,:]=np.abs(f.imag[rmax-wd:rmax,0:wd])
    
    bsg=np.zeros((2*wd,2*wd-2),dtype=bool)
    bsg[:,0:wd-1]=sg_r[:,0:wd-1]<sg_r[:,1:wd]
    bsg[:,wd-1:2*wd-2]=sg_i[:,0:wd-1]<sg_i[:,1:wd]
    
    return bsg
    
    f2=np.zeros((rmax,cmax))+1j*np.zeros((rmax,cmax))
    
    f2.real[0:wd,0:wd-1]=bsg[0:wd,0:wd-1]
    f2.imag[wd:2*wd,0:wd-1]=bsg[0:wd,wd-1:2*wd-2]
    f2.real[rmax-wd:rmax,0:wd-1]=bsg[wd:2*wd,0:wd-1]
    f2.imag[rmax-wd:rmax,0:wd-1]=bsg[wd:2*wd,wd-1:2*wd-2]
    #plt.close("all")
    plt.figure()
    plt.imshow(bsg,interpolation='none')
    plt.set_cmap("gray")
#    plt.figure()
#    im_b2=np.fft.ifft2(f2)
#    plt.imshow(im_b2.real,interpolation='none')
#    
#    plt.figure()
#    plt.imshow(im)
    
def js(b1,b2):
    #jaccard similarity
    return np.sum(np.bitwise_and(b1,b2))/float(np.sum(np.bitwise_or(b1,b2)))
def find1(f,all_bsg,all_files):
    print "find: %s\n"%f
    bsg=getHash(f)
    for i,t in enumerate(all_bsg):
        print "sim: %.3f %s"%(js(bsg,t),all_files[i])
        
def find2(f,all_bsg,all_files):
    print "find: %s\n"%f
    fsg=getHash2(f)
    for i,t in enumerate(all_bsg):
        print "%03d sim: %.3f %s"%(i,np.dot(fsg,t),all_files[i])
def find3(tree,f,all_files):
    fsg=getHash2(f)
    d,ids=tree.query(fsg,k=10)
    for i,index in enumerate(ids):
        print "%03d dis: %.3f %s"%(index,d[i],all_files[index])
def findall(tree,all_files):
    for f in all_files:
        fsg=getHash3(f)
        d,ids=tree.query(fsg,k=10)
        plt.figure(1)  
        for i,index in enumerate(ids):
            print "%03d dis: %.3f %s"%(index,d[i],all_files[index])
            plt.subplot(2,5,i+1)
            plt.title("%03d dis: %.3f"%(index,d[i]),fontsize=10)
            im_icon=np.array(Image.open(all_files[index]).convert('L'))
            plt.imshow(im_icon)
            plt.axis('off')
        plt.set_cmap('gray')    
        plt.show()        
        plt.ginput(1)

if __name__ == '__main__':
    # read image to array
    
    all_files=[]
    for root, dirs, files in os.walk(rootdir):
        for f in files:
                if f.endswith('jpg') or f.endswith('JPG'):
                    all_files.append(os.path.join(root,f))
    #    for subdir in dirs:
    #        for iroot,idirs,ifiles in os.walk(os.path.join(root,subdir)):
    #            for f in ifiles:
    #                if f.endswith('jpg'):
    #                    all_files.append(os.path.join(iroot,f))
    # patch module-level attribute to enable pickle to work
    kdtree.node = kdtree.KDTree.node
    kdtree.leafnode = kdtree.KDTree.leafnode
    kdtree.innernode = kdtree.KDTree.innernode
    ####construct tree
    j=0;
    end=0
    while end is not 1:
        sub_bsg=[]
        sub_files=[]
        for i in xrange(2000):
            if len(all_files) is 0:
                end=1;            
                break
            f=all_files.pop(0)
            sub_files.append(f)    
            print '%02d %s'%(i,f)
            bsg=getHash3(f)    
            sub_bsg.append(bsg)
        tree = kdtree.KDTree(sub_bsg)
        pickleFile = open('%s/tree%02d.dat'%(rootdir,j), 'wb')
        cPickle.dump((sub_files,tree,wd), pickleFile, cPickle.HIGHEST_PROTOCOL)
        pickleFile.close()
        j=j+1
    ####load tree
    pickleFile = open('%s/tree%02d.dat'%(rootdir,0), 'rb')
    (all_files,tree,wd) = cPickle.load(pickleFile)
    pickleFile.close()
    findall(tree,all_files)

