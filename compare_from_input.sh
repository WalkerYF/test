# input
# $1 the first files
# $2 the second files
# $3 the number of example
# $4 = inputfiles direstory default in "in"
# index=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)
# i 负责迭代

if [ ! $4 ]; then
	in="input"
else
	in=$4
fi
output1="output_from_$1"
output2="output_from_$2"
mkdir -p $output1
mkdir -p $output2
for i in `seq $3` # from 1 to 7
do
`./$1 < $in/in$i > $output1/out$i`
`./$2 < $in/in$i > $output2/out$i`
done

# for i in {1..$3}
for i in `seq $3` 
do
temp=`diff --suppress-common-lines $output1/out$i $output2/out$i`
if [ "$temp" == "" ]; then
	echo "out$i : same"
else
	echo "out$i : different"
    echo $temp
fi 
echo "-----------------------"
done
# ./a.exe < in/in1 > out1/out1