from CameraSystemClass import CameraSystem
import cv2 as cv
import numpy as np
import socket,pickle,struct, select, sys
import os
#import thread

try:	
  print("Constructing camera system")
  #cam = CameraSystem([2,1],compressCameraFeed=False)
  cam = CameraSystem([0,1,2])

  frames = cam.captureCameraImages()
  print("Calculating homography for 0 and 1")
  #H, matchesMask = cam.calibrateMatrix()
  Hl, Hr = cam.calibrateMatrixTriple(frames[0], frames[1], frames[2])

  if (Hl is not None and Hr is not None):
    # Save homo to file
    print("Saving homo to file")
    cam.saveHomographyToFile([Hl, Hr],"savedHomographyMatrix.npy")
   
  else:
    print("Not enough matches detected to compute homography")
    Hl, Hr = cam.openHomographyFile("/home/lab/SSLAB_3D_CAM/software/savedHomographyMatrix_perm.npy") #load stored backup 

  # Save homo
  #cam.saveHomographyToFile([H])
  #cam.saveHomographyToFile([Hl, Hr], "/home/lab/SSLAB_3D_CAM/software/savedHomographyMatrix.npy" )

  # Open saved matrix
  #H = cam.openHomographyFile()


  # Socket Create
  server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
  server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  #host_name  = socket.gethostname()
  host_ip = "192.168.55.1" #socket.gethostbyname(host_name)
  print('HOST IP:',host_ip)
  port = 9001
  socket_address = (host_ip,port)

  # Socket Bind
  server_socket.bind(socket_address)

  # Socket Listen
  server_socket.listen(5) #IDK what this does exactly, the "5"
  print("LISTENING AT:",socket_address)
  print("Ready!")
  enable = True
  loop = True

  while loop:

    client_socket,addr = server_socket.accept()
    print('GOT CONNECTION FROM:',addr)
    while(client_socket):
         # capture images
   
         frames = cam.captureCameraImages()

         # Display the resulting frame
         #cv.imshow('raw', np.concatenate(frames, axis=1))
      
         #print("Recalibrating")
         #temp_Hl, temp_Hr = cam.calibrateMatrixTriple(frames[0], frames[1], frames[2])
         #if temp_Hl is None or temp_Hr is None:
              #print("Not enough matches to successfully calibrate")

         #else:
              #cam.saveHomographyToFile([temp_Hl, temp_Hr], "/home/lab/SSLAB_3D_CAM/software/savedHomographyMatrix.npy" )
              #Hl = temp_Hl
              #Hr = temp_Hr
              #print("Done recalibrating")
               
         
               

         # Stitch
         #stitched = cam.homographyStitch(frames[0], frames[1], H[0])
         stitched = cam.tripleHomographyStitch(frames[0], frames[1], frames[2], Hl, Hr)
         #cv.imshow('stitched', stitched)
         a = pickle.dumps(stitched)
         message = struct.pack("Q",len(a))+a

         client_socket.sendall(message)
            

         if(select.select([sys.stdin,],[],[],0)[0]):
             byte = sys.stdin.read(1)
             if(byte == 'q'):
                print("Exiting...")
                client_socket.close()
                loop = False
                break
             elif (byte == 'r'):
                 print("Recalibrating")
                 temp_Hl, temp_Hr = cam.calibrateMatrixTriple(frames[0], frames[1], frames[2])
                 if temp_Hl is None or temp_Hr is None:
                     print("Not enough matches to successfully calibrate")

                 else:
                     cam.saveHomographyToFile([temp_Hl, temp_Hr], "/home/lab/SSLAB_3D_CAM/software/savedHomographyMatrix.npy" )
                     Hl = temp_Hl
                     Hr = temp_Hr
                     print("Done recalibrating")
         if(enable):
              print("Transmitting!")
              enable = False

except KeyboardInterrupt:
  print("Stitching program interrupted")


                
