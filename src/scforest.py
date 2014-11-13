"""
Created on Thu Nov 13 22:47:53 2014

@author: Wasit
"""
import os
rootdir="ss"
for root, dirs, files in os.walk(rootdir):
    for f in files:
        if f.endswith('json'):
            print f
#    #reading the tree pickle file
#    pickleFile = open(rfile, 'rb')
#    root = pickle.load(pickleFile)
#    pickleFile.close()
#    #init the test tree
#    t=tree()
#    t.settree(root)