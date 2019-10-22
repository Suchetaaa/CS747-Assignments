#!/bin/sh

instance=$2
algorithm=$4
randomseed=$6
epsilon=$8
horizon=${10}

if [ $instance = "../instances/i-1.txt" ]; then
	instance_num=0
fi

if [ $instance = "../instances/i-2.txt" ]; then
	instance_num=1
fi

if [ $instance = "../instances/i-3.txt" ]; then
	instance_num=2
fi


if [ $algorithm = "round-robin" ]; then
	# echo "round-robin"
	python roundRobin.py $instance $epsilon $horizon $randomseed

elif [ $algorithm = "epsilon-greedy" ]; then
	# echo "epsilon-greedy"
	python epsilonGreedy.py $instance $epsilon $horizon $randomseed

elif [ $algorithm = "ucb" ]; then
	# echo "ucb"
	python ucb.py $instance $epsilon $horizon $randomseed

elif [ $algorithm = "kl-ucb" ]; then
	# echo "kl-ucb"
	python klucb.py $instance $epsilon $horizon $randomseed

else
	# echo "thompson-sampling"
	python thompsonSampling.py $instance $epsilon $horizon $randomseed

fi