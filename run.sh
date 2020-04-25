rm out.txt
for ((i=0; i<9; i++))
do
	./hw2.exe 100000000 $i
done
