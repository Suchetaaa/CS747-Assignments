#!/bin/bash

method=$4
inp_file=$2

if [ $method = "hpi" ]; then
	# echo "hpi"
	python howard_pi.py $inp_file
fi

if [ $method = "lp" ]; then
	# echo "lp"
	python lp_solve.py $inp_file
fi


