from cProfile import label
import subprocess
import sys
import paramiko




user = 'lab'
host = '192.168.55.1'
port = 22 #ssh port
password = 'ss_pass'
cmd = 'python3 -u ~/SSLAB_3D_CAM/software/test_program.py 2>&1' #'python3 -u ~/SSLAB_3D_CAM/software/IO_video_loop.py 2>&1' #-u for unbuffered stdout, 2>&1 redirects stderr to stdout
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, port, user, password)
stdin, stdout, stderr = client.exec_command(cmd)
print("executing IO_video_loop.py......")
try:
   while True:
     # if (stdout.channel.recv_ready()):
     #    line = stdout.channel.recv(1024)
       #  print(line)
         
      line = stdout.readline()
      #err = stderr.readline()
      if line:
         print(line.rstrip())


      
      #sys.stdout.flush()
   
except KeyboardInterrupt:
   print("Quitting VideoLoop....")
   stdin.write('q')
   
   #while True:
    # line = stdout.readline()
    # print(line.rstrip())
    # if not line:
    #  break
   

client.close()
# Python 3
#subprocess.Popen(f"ssh {user}@{host} {cmd}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()