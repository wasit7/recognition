#!/bin/bash
dset="ss"
#rm -rf ${dset}
#mkdir ${dset}

for var in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
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