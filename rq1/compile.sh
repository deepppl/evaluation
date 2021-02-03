backend=$1
mode=$2
file=$3
echo XXX $file >> /tmp/out-$backend-$mode
echo XXX $file >> /tmp/err-$backend-$mode
stanc --$backend --mode $mode "$file" \
	>> /tmp/out-$backend-$mode 2>> /tmp/err-$backend-$mode
res=$?
echo $file: $res >> logs-$backend-$mode
echo -n .
