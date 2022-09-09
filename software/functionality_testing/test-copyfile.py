import sys 
import subprocess

args = sys.argv
args[0] = "../common/copy.sh" #needs to be path to shell script
#call shell script from python script
subprocess.check_call(args)

