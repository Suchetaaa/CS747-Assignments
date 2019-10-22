#!/bin/sh

j=1
epsilon=0.2
for instance in ../instances/i-1.txt ../instances/i-2.txt ../instances/i-3.txt
do
	for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 
	do
		echo "Iteration $j"
		
		python roundRobin.py $instance $epsilon 204800 $seed
		j=$((j + 1))
	done
done

epsilon=0.2
for instance in ../instances/i-1.txt ../instances/i-2.txt ../instances/i-3.txt 
do
	for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 
	do
		echo "Iteration $j"
		
		python epsilonGreedy.py $instance $epsilon 204800 $seed
		j=$((j + 1))
	done
done

epsilon=0.02
for instance in ../instances/i-1.txt ../instances/i-2.txt ../instances/i-3.txt
do
	for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 
	do
		echo "Iteration $j"
		
		python epsilonGreedy.py $instance $epsilon 204800 $seed
		j=$((j + 1))
	done
done

epsilon=0.002
for instance in ../instances/i-1.txt ../instances/i-2.txt ../instances/i-3.txt
do
	for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49  
	do
		echo "Iteration $j"
		
		python epsilonGreedy.py $instance $epsilon 204800 $seed
		j=$((j + 1))
	done
done

epsilon=0.2
for instance in ../instances/i-1.txt ../instances/i-2.txt ../instances/i-3.txt 
do
	for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
	do
		echo "Iteration $j"
		
		python ucb.py $instance $epsilon 204800 $seed
		j=$((j + 1))
	done
done

epsilon=0.2
for instance in ../instances/i-1.txt ../instances/i-2.txt ../instances/i-3.txt
do
	for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 
	do
		echo "Iteration $j"
		
		python klucb.py $instance $epsilon 204800 $seed
		j=$((j + 1))
	done
done

epsilon=0.2
for instance in ../instances/i-1.txt ../instances/i-2.txt ../instances/i-3.txt 
do
	for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 
	do
		echo "Iteration $j"
		
		python thompsonSampling.py $instance $epsilon 204800 $seed
		j=$((j + 1))
	done
done






