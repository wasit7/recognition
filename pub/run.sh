#!/bin/bash
rm -rf ss
mkdir ss
nohup ./et05.sh > /home/sgeadmin/ss/main.out 2> /home/sgeadmin/ss/main.out < /dev/null &
