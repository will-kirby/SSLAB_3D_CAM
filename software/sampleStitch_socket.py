from CameraSystemClass import CameraSystem
import cv2 as cv
import numpy as np
import socket,pickle,struct, select, sys
#import thread

	
print("Constructing camera system")
#cam = CameraSystem([2,1],compressCameraFeed=False)
cam = CameraSystem([0,1,2])

frames = cam.captureCameraImages()
print("Calculating homography for 0 and 1")
#H, matchesMask = cam.calibrateMatrix()
#Hl, Hr = cam.calibrateMatrixTriple(frames[0], frames[1], frames[2])

# Save homo
#cam.saveHomographyToFile([H])
#cam.saveHomographyToFile([Hl, Hr], "/home/lab/SSLAB_3D_CAM/software/savedHomographyMatrix.npy" )

# Open saved matrix
#H = cam.openHomographyFile()
Hl, Hr = cam.openHomographyFile("/home/lab/SSLAB_3D_CAM/software/savedHomographyMatrix_perm.npy")

# Socket Create
server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host_name  = socket.gethostname()
host_ip = "192.168.55.1" #socket.gethostbyname(host_name)
print('HOST IP:',host_ip)
port = 9004
socket_address = (host_ip,port)

# Socket Bind
server_socket.bind(socket_address)

# Socket Listen
server_socket.listen(5) #IDK what this does exactly, the "5"
print("LISTENING AT:",socket_address)
print("Ready!")

while True:
    connection = False
    client_socket,addr = server_socket.accept()
    print('GOT CONNECTION FROM:',addr)
    if client_socket:
        connection = True
    while(connection):
         # capture images
   
         frames = cam.captureCameraImages()

         # Display the resulting frame
         #cv.imshow('raw', np.concatenate(frames, axis=1))
      
         #print("Recalibrating")
         #Hl, Hr = cam.calibrateMatrixTriple(frames[0], frames[1], frames[2])
        # if(Hl == None or Hr == None):
         #   Hl, Hr = cam.openHomographyFile("/home/lab/SSLAB_3D_CAM/software/savedHomographyMatrix.npy")
         #else:
         #cam.saveHomographyToFile([Hl, Hr], "/home/lab/SSLAB_3D_CAM/software/savedHomographyMatrix.npy" )
         #print("Done recalibrating")
                 
         
               

         # Stitch
         #stitched = cam.homographyStitch(frames[0], frames[1], H[0])
         stitched = cam.tripleHomographyStitch(frames[0], frames[1], frames[2], Hl, Hr)
         #cv.imshow('stitched', stitched)
         a = pickle.dumps(stitched)
         message = struct.pack("Q",len(a))+a
         client_socket.sendall(message)
         if cv.waitKey(1) == ord('r'):
             print("Recalibrating")
             #H, matchesMask = cam.calibrateMatrix()
             Hl, Hr = cam.calibrateMatrixTriple(frames[0], frames[1], frames[2])
             print("Done recalibrating")

         if cv.waitKey(1) == ord('q'):
             client_socket.close()
             break

         if(select.select([sys.stdin,],[],[],0)[0]):
             byte = sys.stdin.read(1)
             if(byte == 'q'):
                client_socket.close()
                break
             elif (byte == 'r'):
                 print("in recalibrate")
                 try:
                     print("Recalibrating")
                     Hl, Hr = cam.calibrateMatrixTriple(frames[0], frames[1], frames[2])
                     cam.saveHomographyToFile([Hl, Hr], "/home/lab/SSLAB_3D_CAM/software/savedHomographyMatrix.npy" )
                     print("Done recalibrating")
                 
                 except:
                     print("Not enough matches to successfully calibrate")





                
