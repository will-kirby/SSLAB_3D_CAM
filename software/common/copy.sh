#! /bin/bash

Syntax(){
echo "Invalid syntax"
echo "Syntax: ./copy.sh [-d remote_dest_directory] [-f remote_filename] remote_username file_to_copy"
exit
}


username="";
sourcefile="";
remote_filename="";
destdir="";
dir_set=0;
name_set=0;
ip_address="192.168.55.100" #Assigned to Host computer when conneceted to Jetson w/ microusb

while getopts d:f: flag; do
   case $flag in 
      d) destdir=$OPTARG ; dir_set=1 ;;
      f) remote_filename=$OPTARG ; name_set=1 ;;
      \?) Syntax;;
   esac
done 

set_sum=$(($dir_set + $name_set));

if [ $# -ne $(($((2*$set_sum)) + 2)) ] 
then
  Syntax
fi

case $set_sum in
   0) username=$1 ; sourcefile=$2 ;;
   1) username=$3 ; sourcefile=$4 ;;
   2) username=$5 ; sourcefile=$6; remote_filename="/${remote_filename}" ;;
   \?) Syntax;;
esac



scp "${sourcefile}" "${username}@${ip_address}:${destdir}${remote_filename}"
