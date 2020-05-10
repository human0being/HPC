#!/bin/sh

rm out.txt
mpicc ping_pong.c

word="AllworkandnoplaymakesJackadullboy.AllworkandnoplaymakesJackadullboy.AllworkandnoplaymakesJackadullboy."
for ((i=0; i < 11; i++))
do
#	echo $word
	echo $i
	mpirun --oversubscribe -np 4 a.out $word >> out.txt
	word=$word$word

done
