#!/bin/bash
Adset=("ss")
for dsetname in ${Adset[@]}
do
	rm -rf ${dsetname}
	mkdir ${dsetname}
	run="ipython scmain.py ${dsetname} >> ${dsetname}/log.out"
	tt=$(date +"%y%m%d_%H%M_%S")
	echo ">>>>start:${tt}, ${dsetname}"
	echo "$run"
	eval "$run"
	tt=$(date +"%y%m%d_%H%M_%S")
	echo ">>>>end:${tt}"
done
