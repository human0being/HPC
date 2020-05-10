#!/bin/sh

mpicc automata_time.c
N=1000
out="out_0"

for ((i=1; i<5; i++))
do	
	for ((j=1; j<5; j++))
	do
		mpirun --oversubscribe -np $j ./a.out 184 $N >> $out
	done
	out=$out$i
	N=$N"0"
	echo $N
	echo $out
done
