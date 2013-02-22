#!/bin/bash

make clean > /dev/null
make bench-test > /dev/null

randm=$RANDOM
randn=$RANDOM
let "randm = randm % 9001 + 1000"
let "randn = randn % 69 + 32"

testm=(1022	1608	7577	2048	7000	9414 	9833	$randm)
testn=(32 	48	 	59		64		80		96		38		$randn)

for (( c=0; c<${#testm[@]}; c++ ))
do
	./bench-test ${testm[c]} ${testn[c]}
done

make clean > /dev/null
