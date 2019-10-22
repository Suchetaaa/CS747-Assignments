#!/bin/sh

i=1

echo "Test $i"
./bandit.sh --instance ../instances/i-2.txt --algorithm round-robin --randomSeed 33 --epsilon 0.7 --horizon 20
sleep 1
i=$((i + 1))

echo "Test $i"
./bandit.sh --instance ../instances/i-2.txt --algorithm round-robin --randomSeed 0 --epsilon 0.1 --horizon 10
sleep 1
i=$((i + 1))

echo "Test $i"
./bandit.sh --instance ../instances/i-1.txt --algorithm epsilon-greedy --randomSeed 0 --epsilon 0.333 --horizon 198
sleep 1
i=$((i + 1))

echo "Test $i"
./bandit.sh --instance ../instances/i-3.txt --algorithm epsilon-greedy --randomSeed 2 --epsilon 0.002 --horizon 77
sleep 1
i=$((i + 1))

echo "Test $i"
./bandit.sh --instance ../instances/i-1.txt --algorithm epsilon-greedy --randomSeed 5982 --epsilon 0.01 --horizon 20000
sleep 1
i=$((i + 1))

echo "Test $i"
./bandit.sh --instance ../instances/i-3.txt --algorithm ucb --randomSeed 33 --epsilon 0.403 --horizon 2
sleep 1
i=$((i + 1))

echo "Test $i"
./bandit.sh --instance ../instances/i-3.txt --algorithm kl-ucb --randomSeed 33 --epsilon 0.403 --horizon 95
sleep 1
i=$((i + 1))

echo "Test $i"
./bandit.sh --instance ../instances/i-2.txt --algorithm thompson-sampling --randomSeed 10 --epsilon 0.3 --horizon 4
sleep 1
i=$((i + 1))

echo "Test $i"
./bandit.sh --instance ../instances/i-3.txt --algorithm thompson-sampling --randomSeed 49 --epsilon 0.1 --horizon 67780
sleep 1
i=$((i + 1))

echo "Test $i"
./bandit.sh --instance ../instances/i-1.txt --algorithm kl-ucb --randomSeed 2 --epsilon 0.1 --horizon 201
sleep 1
i=$((i + 1))
