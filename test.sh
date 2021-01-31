#!/usr/bin/env bash
  
FLAG=$1
[ "$FLAG" != "TILING" ] &&  \
[ "$FLAG" != "ELEPACK" ] && \
[ "$FLAG" != "ELE" ] &&     \
[ "$FLAG" != "ROWPACK" ] && \
FLAG="ROW"
echo $FLAG
wget -q https://judgegirl.csie.org/downloads/testdata/10100/3.in
wget -q https://judgegirl.csie.org/downloads/testdata/10100/3.out
nvcc -D$FLAG -Xcompiler "-O2 -fopenmp" mm.cu -o mm > /dev/null
time ./mm < 3.in
echo "Expected:"
cat 3.out
rm mm 3.in 3.out
