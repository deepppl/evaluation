backend=$1
mode=$2
logfile=$3
file=$4
name=`basename $file .stan`
mkdir -p _tmp
echo XXX $file >> /tmp/out-$backend-$mode
echo XXX $file >> /tmp/err-$backend-$mode
stanc --$backend --mode $mode "$file" --o "_tmp/$name.py" \
	>> /tmp/out-$backend-$mode 2>> /tmp/err-$backend-$mode
res=$?
echo $file, $res >> $logfile
echo -n .
