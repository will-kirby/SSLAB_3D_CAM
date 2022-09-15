from cProfile import label
import subprocess

user = 'lab'
host = '192.168.55.1'
password = 'ss_pass'
cmd = 'python3 IO_video_loop.py Alan test_stitch_vid.avi'

# Python 3
subprocess.Popen(f"echo {password} | ssh {user}@{host} {cmd}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()