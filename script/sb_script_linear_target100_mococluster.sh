#!/bin/bash 

for VARIABLE in 024 054 084 104 134 144 174 184
do
	id=`sbatch run_linear_eval_target100_nondist.sh $VARIABLE`;
    echo "$id : $VARIABLE";
done