import select
import sys
import time
i = 0
while (True):
   print("Hello! " + str(i))
   i += 1
   if(select.select([sys.stdin,],[],[],0)[0] and sys.stdin.read(1) == 'q'):
        break

   time.sleep(1)
