import Jetson.GPIO as GPIO
import time

led_pin=12
#Assumes there is a readable file called flask_output.log that has stdout for this script
f = open("flask_output.log", "r")

ip = "192.168.55.1" #localhost
port = 5000

GPIO.setmode(GPIO.BOARD) 
GPIO.setup(led_pin, GPIO.OUT)

f.seek(0,2)    

#turn on LED when server is listening

    while True:
       line = f.readline()
       if not line:
         time.sleep(.1)
       else:
          output = line.rstrip()
          if(output == f" * Running on http://{ip}:{port}/ (Press CTRL+C to quit)"):
             GPIO.output(led_pin, GPIO.HIGH)
             #f.close()
             #break



