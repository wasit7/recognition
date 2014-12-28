#!/bin/bash
dset="ss"

for var in 1 2 3 4 5 6 7 8 9 10
do
	echo $var
	run="ipython scmain.py ${dset}>> ${dset}/log.out"
	tt=$(date +"%y%m%d_%H%M_%S")
	echo ">>>>start:${tt}, ${dset}"
	#echo "$run"
	eval "$run"
	tt=$(date +"%y%m%d_%H%M_%S")
	echo ">>>>end:${tt}"
done