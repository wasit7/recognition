"""
GNU GENERAL PUBLIC LICENSE Version 2

Created on Tue Oct 14 18:52:01 2014

@author: Wasit
"""
import numpy as np
import os
from PIL import Image
from scipy.ndimage import filters
try:
    import json
except ImportError:
    import simplejson as json

num_img=100
spi=20

rootdir="dataset"
mrec=64
mtran=64
margin=mrec+mtran
class dataset:
    def __init__(self):
        '''
        To create and initialise        
        self.dimtheta--(m)dimension of theta. theta is a column vector
        self.size------(n)number of samples in the root bag
        self.I---------prepocessed data
        self.samples---the marix which has size of [(p+1)xn],
                       where p is size of vector that identify location 
                       of a sample in self.I. 
                       Note that the fist row of self.sample is label
        '''
        
        #1 self.cmax: maximum number of classes
        self.clmax=520
        #2 self.spi: number of samples per image
        self.spi=spi
        #3 self.dim_theta: [r1,c1,r2,c2,bin]^T
        self.dim_theta=5
        self.dim_bin=2
        
#loading json files
        imgf=[]
        for root, dirs, files in os.walk(rootdir):
            for subdir in dirs:
                for iroot,idirs,ifiles in os.walk(os.path.join(root,subdir)):
                    for f in ifiles:
                        if f.endswith('jpg'):
                            imgf.append(os.path.join(iroot,f))
        #set sampling-rate here 
        self.imgf=[imgf[i] for i in np.random.permutation(len(imgf))[0:num_img]]
        print "len(self.imgf)=%d"%len(self.imgf)        
        #4 self.size: number of all samples in the root bag
        self.size=len(self.imgf)*self.spi;
        #5 self.I: the data
        #6 self.samples: samples[x]=[class,img, row, column]^T
        self.I=[]
        self.samples=np.zeros((4,self.size),dtype=np.uint)
        
        for i, jf in enumerate(self.imgf):
        #self.I: the data
            img_path= jf           
            #print img_path
            #print(img_path)
            im=np.array(Image.open(img_path).convert('L'))
            rmax,cmax=im.shape
            #imx and imy are graduent images in x and y directions
            imx = np.zeros(im.shape)
            imy = np.zeros(im.shape)
            #sigma for gausian window
            sigma=2
            filters.gaussian_filter(im, (sigma,sigma), (0,1), imx)
            filters.gaussian_filter(im, (sigma,sigma), (1,0), imy)
            #categorise directions of gradient into 4 groups (sw,se,nw and ne) 
            I_tem=np.zeros((rmax,cmax,self.dim_bin),dtype=np.uint16)
            
#            g= (0<imx).astype(np.uint16) + 2*(0<imy).astype(np.uint16)
#            I_tem[:,:,0] = (g[:,:]==0).astype(np.uint16).cumsum(0).cumsum(1)
#            I_tem[:,:,1] = (g[:,:]==1).astype(np.uint16).cumsum(0).cumsum(1)
#            I_tem[:,:,2] = (g[:,:]==2).astype(np.uint16).cumsum(0).cumsum(1)
#            I_tem[:,:,3] = (g[:,:]==3).astype(np.uint16).cumsum(0).cumsum(1)
            
            I_tem[:,:,0] = (0<imx).astype(np.uint16).cumsum(0).cumsum(1)
            I_tem[:,:,1] = (0<imy).astype(np.uint16).cumsum(0).cumsum(1)
            
            self.I.append(I_tem)
            
        #samples[x]=[class,img, row, column]^T
            ki=i*self.spi
            kf=ki+self.spi
            #image index
            self.samples[1,ki:kf]=i
            #row
            r=np.random.randint(margin,rmax-margin,self.spi)
            self.samples[2,ki:kf]=r;
            #column
            c=np.random.randint(margin,cmax-margin,self.spi)
            self.samples[3,ki:kf]=c;
            #label
            self.samples[0,ki:kf]=0
        #self.I=np.array(self.I)
        self.samples=self.samples.astype(np.uint16)                     
                
    def __del__(self):
        del self.clmax
        del self.spi
        del self.size
        del self.I
        del self.samples
    def getL(self,x):
        '''
        input: 
            [1D ndarray dtype=np.uint32]
        output: 
            [1D ndarray dtype=np.uint32]
        '''
        return self.samples[0,x]
    def setL(self,x,L):
        self.samples[0,x]=L
    def getI(self,theta,x):
        '''
        input:
            x: [1D ndarray dtype=np.uint32]\n
            theta: [1D ndarray float]
        output: 
            [1D ndarray dtype=np.uint32]
        Description:
            In spiral case, it uses only first row of the thetas
        '''
        #engine.getQH() call this
        r1=self.samples[2,x]+theta[0]
        c1=self.samples[3,x]+theta[1]
        r2=self.samples[2,x]+theta[2]
        c2=self.samples[3,x]+theta[3]
        bins=theta[self.dim_theta-1]
        f=np.zeros(len(x))
        for i,ix in enumerate(x):
            img=self.samples[1,ix]
            L1=self.I[img][r1[i],c1[i],bins]
            L2=self.I[img][r1[i],c2[i],bins]
            L3=self.I[img][r2[i],c1[i],bins]
            L4=self.I[img][r2[i],c2[i],bins]
            f[i]=float(L4+L1-L2-L3)
        return f
    def show(self):
        import matplotlib.pyplot as plt
        print("number of images: {}".format(len(self.I)))    
        for i in xrange(len(self.imgf)):
            img_path= self.imgf[i] 
                
            print(img_path)
            im=np.array(Image.open(img_path).convert('L'))
            plt.hold(False)        
            plt.imshow(im)
            plt.hold(True)
            for j in range(self.size):
                #samples[x]=[0_class,1_img, 2_row, 3_column]^T
                if self.samples[1,j]==i:
                    plt.text(self.samples[3,j], self.samples[2,j], "%03d"%self.samples[0,j], fontsize=12,color='red')
                    #plt.plot(self.samples[3,j],self.samples[2,j],markers[self.samples[0,j]])
            plt.set_cmap('gray')
            plt.show()
            plt.ginput(1)
        plt.close('all')
if __name__ == '__main__':
#    import matplotlib.pyplot as plt
    dset=dataset()
    dset.show()
    
    
    
    
#    print("number of images: {}".format(len(dset.I)))    
#    markers=['ko','ro','go','bo','po']
#    for i in xrange(len(dset.jsonfiles)):
#        f=open(dset.jsonfiles[i],"r")
#        js=json.loads(f.read())
#        f.close()
#        img_path= rootdir + js['path'][1:]
#        print(img_path)
#        im=np.array(Image.open(img_path).convert('L'))
#        plt.hold(False)        
#        plt.imshow(im)
#        plt.hold(True)
#        for j in range(dset.size):
#            #samples[x]=[0_class,1_img, 2_row, 3_column]^T
#            if dset.samples[1,j]==i:
#                plt.plot(dset.samples[3,j],dset.samples[2,j],markers[dset.samples[0,j]])
#        plt.set_cmap('gray')
#        plt.show()
#        plt.ginput()
#    plt.close('all')
#--