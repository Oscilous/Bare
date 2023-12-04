import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
import sys

from picamera.array import PiRGBArray
from picamera import PiCamera

#GPIO.cleanup()
GPIO.setwarnings(False)
servo_pin = 13
led_pin = 21

GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_pin,GPIO.OUT)

font = cv2.FONT_HERSHEY_PLAIN

pwm = GPIO.PWM(servo_pin,50) 
pwm.start(8.8) #init af servo motor
time.sleep(1)
pwm.ChangeDutyCycle(0)
counter=0
ticker=0
arrayNr = 1
GPIO.output(led_pin, 1)
startTime = time.time()
failCounter=0
PinchCD = time.time()
shutdownC = 0
adjust = 0
    
def meanPix(arrayAvg, diff,W_enable,otsu,peripheral):
    global adjust 
    stopTime = time.time()
    adjust = adjust -1
    #print(enabl)
    #print(W_enable)
    #print(adjust)
    if (W_enable < 10 and adjust < 0)  :
        #print(W_enable)     
        #print(otsu,diff)
        #time.sleep(0.1)
        servoDelay = 0.17
        camDelay = 0.1
        
        
        global startTime
        global counter
        global failCounter
        global botArray
        global histArray
        global Csys
        global maskGain
        global shutdownC
        
        
        
        if (38>otsu>0.5  and diff<2 and maskGain < 0.85) :
            print("38>otsu>0.5  and diff<2 and maskGain < 0.85")
            maskGain = maskGain+0.05
            otsu=50
            peripheral=0
            #GPIO.output(led_pin, 0)
            
            #GPIO.output(led_pin, 1)
            adjust=5
            #time.sleep(camDelay)
                   
            
        if (38>otsu>0.5  and diff<0.18 and maskGain > 0.84) or (peripheral>0.45 and diff<0.18 ):
            print("38>otsu>0.5  and diff<0.18 and maskGain > 0.84) or (peripheral>0.45 and diff<0.18 ")
            stopTime = time.time()
            elapsedTime=stopTime-startTime
            print("Fail  -  " + str(elapsedTime))
            startTime = time.time()
            cv2.putText(vis, "Threshold val: "+str(otsu) , (28,30),font,2,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(vis, "Settling val: "+str(diff) , (32,58),font,2,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(vis, "Peripheral val: "+str(peripheral) , (32,932),font,2,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(vis, "Mask gain: "+str(maskGain) , (32,90),font,2,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(vis, "Fail", (740,920),font,4,(0,0,255),2,cv2.LINE_AA)
            cv2.imwrite('/home/pi/Desktop/pakning image/fail/fail' + str(counter) + '.jpg',vis)
            print(counter, otsu, diff,peripheral)
            #histogram()
               #Csys=(483,452)
            counter += 1
            GPIO.output(led_pin, 0)
            pwm.ChangeDutyCycle(6.8) # rotate to 0 degrees
            time.sleep(camDelay)
            GPIO.output(led_pin, 1)
            time.sleep(servoDelay+0.07)
                #botArray[1]=8
            pwm.ChangeDutyCycle(3)# rotate to 90 degrees
            
            
            time.sleep(servoDelay)            
            pwm.ChangeDutyCycle(5) # rotate to 90 degrees
            time.sleep(servoDelay)
            pwm.ChangeDutyCycle(0)
            time.sleep(camDelay)
            failCounter=failCounter+1
            maskGain = 0.70
            peripheral=0
           

            
        elif 0.01<=otsu<=0.5 and diff<0.08:
            print("0.01<=otsu<=0.5 and diff<0.08")
            stopTime = time.time()
            elapsedTime=stopTime-startTime
            print("Pass  -  " + str(elapsedTime))
            cv2.putText(vis, "Timer val: "+str(elapsedTime) , (28,942),font,2,(255,255,255),2,cv2.LINE_AA)
            startTime = time.time()
            #cv2.putText(vis, "Threshold val: "+str(otsu) , (28,30),font,2,(255,255,255),2,cv2.LINE_AA)
            #cv2.putText(vis, "Settling val: "+str(diff) , (32,58),font,2,(255,255,255),2,cv2.LINE_AA)
            #cv2.putText(vis, "Pass", (740,920),font,4,(0,255,0),2,cv2.LINE_AA)
            #cv2.imwrite('/home/pi/Desktop/pakning image/pass/pass' + str(counter) + '.jpg',vis)
            #histogram()
            #Csys=(484,484)
            #print(counter, otsu, diff,peripheral)
            counter += 1
            GPIO.output(led_pin, 0)
            pwm.ChangeDutyCycle(6.8)
            time.sleep(camDelay)
            GPIO.output(led_pin, 1)# rotate to 0 degrees
            time.sleep(servoDelay+0.07)
            pwm.ChangeDutyCycle(11.2)# rotate to 90 degrees
            
            #botArray[1]=8
            
            time.sleep(servoDelay)
            pwm.ChangeDutyCycle(8.8) # rotate to 90 degrees
            time.sleep(servoDelay)
            pwm.ChangeDutyCycle(0)
            
            failCounter=0
            maskGain = 0.70
            peripheral=0
            shutdownC=0
            
        elif 0.01>=otsu and diff<0.08:
            print("0.01>=otsu and diff<0.08")
            shutdownC=3
            
        elif 40<otsu and W_enable < 20:
            print("40<otsu and W_enable < 20")
            GPIO.output(led_pin, 0)
            time.sleep(0.1)
            GPIO.output(led_pin, 1)
            time.sleep(0.3)
            failCounter=failCounter+1
            #time.sleep(servoDelay)
            if failCounter==10:
                print("Consecutive fail counter overflow")
                cv2.putText(vis, "Threshold val: "+str(otsu) , (28,30),font,2,(255,255,255),2,cv2.LINE_AA)
                cv2.putText(vis, "Settling val: "+str(diff) , (32,58),font,2,(255,255,255),2,cv2.LINE_AA)
                cv2.putText(vis, "Peripheral val: "+str(peripheral) , (32,932),font,2,(255,255,255),2,cv2.LINE_AA)
                cv2.putText(vis, "Fail", (740,920),font,4,(0,0,255),2,cv2.LINE_AA)
                cv2.imwrite('/home/pi/Desktop/pakning image/fail/fail' + str(counter) + '.jpg',vis)
                time.sleep(servoDelay+1)
                pwm.ChangeDutyCycle(3)
                time.sleep(servoDelay+1)
                pwm.ChangeDutyCycle(5)
                time.sleep(servoDelay)
                pwm.ChangeDutyCycle(0)
                failCounter=0
                shutdownC = shutdownC+1
                print(shutdownC)
                
        elif(shutdownC==3):
                print("shutdownC==3")
                sys.exit()
        
        
def nothing(x):
    pass
 
#cv2.namedWindow("Trackbars")
 
#cv2.createTrackbar("B2", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("Gain", "Trackbars", 0, 150, nothing)
#cv2.createTrackbar("Rad", "Trackbars", 300, 500, nothing)
#cv2.createTrackbar("W", "Trackbars", 0, 255, nothing)
#W = cv2.getTrackbarPos("W", "Trackbars")

camera = PiCamera()
camera.resolution = (960, 960)
camera.framerate = 30
camera.brightness = 47 #48 til clen mask5
camera.contrast = 0 #1 giver bedst detection
#camera.image_effect='blur'
#camera.IMAGE_EFFECTS
camera.exposure_mode = 'backlight'
#camera.EXPOSURE_MODES
camera.awb_mode = 'fluorescent'
maskGain = 0.7 #er alt hvidt = lavt gain, intet hvidt = for højt gain

rawCapture = PiRGBArray(camera, size=camera.resolution)

maskB = cv2.imread('mask clean11.jpg' , cv2.IMREAD_GRAYSCALE)
maskC = cv2.resize(maskB,camera.resolution)

circle = np.zeros(camera.resolution, dtype="uint8")
Csys=(480,468) #x,y coordinates 0,0 i venstre top
Dia=(401)
cv2.circle(circle, Csys,Dia,255,-1)
#cv2.imshow("Circle",circle)

circle2 = np.zeros(camera.resolution, dtype="uint8")
 #x,y coordinates 0,0 i venstre top
cv2.circle(circle2, Csys,478,255,35)
cv2.rectangle(circle2,(960,0),(10,40),(0,0,255),-1)

#circel2Neg=cv2.bitwise_not(circle2)
#cv2.imshow("Circle",circle2)

arraySize =8
botArray = np.zeros((arraySize,),dtype=float)
topArray = np.zeros((arraySize,),dtype=float)
botArray[1] = 100

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array #rå pixel array

    #Threshold values
    B = 215 #threshold for pellet center
    W = 255
    W2 =250 #threshold for no pellet
    B2 =185 #threshold for the outline of the pack
    B2 = cv2.getTrackbarPos("B2", "Trackbars") # Adjustable threshold
    maskGain = (cv2.getTrackbarPos("Gain", "Trackbars")/100)
    W2 = cv2.getTrackbarPos("W", "Trackbars")

    gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #converts pixel array to grayscale from HSV
    grayINV = cv2.bitwise_not(gray1)
    maskINV  = cv2.bitwise_not(maskC)
    maskINVScale = cv2.multiply(maskINV,maskGain)
    gray2 = grayINV-maskINVScale
    gray = cv2.bitwise_not(gray2)
    cv2.imshow("base", gray2)

    outline = cv2.bitwise_and(circle2,gray2)
    testnavn, outlineT= cv2.threshold(outline,B2,255,cv2.THRESH_BINARY)

    #cv2.imshow("base thresh", outlineT)
    #cv2.imshow("base", )
    #cv2.imshow("scalar", maskINVScale)

    Wthresh, Wbinary= cv2.threshold(gray1,W2,W,cv2.THRESH_BINARY) #run a binary threshold mask over the grayscale image to see if there is a seal in the sensor
    masked_data2 = cv2.bitwise_and(circle,Wbinary) #afgrænser outputtet til pakningens midte 
    #thresh1, binaryImg= cv2.threshold(gray,B,W,cv2.THRESH_OTSU) #kørere et binært tærskel maske over grayscale billede
    #masked_data1 = cv2.bitwise_and(circle,binaryImg) #afgrænser outputtet til pakningens midte
    thresh2, OtsuImg= cv2.threshold(gray,B,W,cv2.THRESH_BINARY)
    #cv2.imshow("base thresh", OtsuImg)
    #thresh2, OtsuImg2= cv2.threshold(gray,B,W,cv2.THRESH_OTSU)#rafinere outputtet med OTSU algoritmen
    OTSUImg=cv2.bitwise_not(OtsuImg)
    #OTSUImg2=cv2.bitwise_not(OtsuImg2)
    masked_data3 = cv2.bitwise_and(circle,OTSUImg)
    #masked_data32 = cv2.bitwise_and(circle,OTSUImg2)


    image_area=cv2.circle(image, Csys,Dia,(0,67,180),1) #tegner en cyrcel på det originale billede for at vise det søgte område


    overlay = cv2.addWeighted(gray1,0.5, (masked_data3+outlineT), 0.7,0)
    #overlay2 = cv2.addWeighted(gray1,0.5, masked_data32, 0.7,0)#tilføjer resultatet som et overlay på grayscale billede
    Overlay_area=cv2.circle(overlay,Csys,Dia,(10,220,30),1)
    Overlay_area=cv2.circle(overlay,Csys,458,(10,220,30),2)
    #Overlay_area2=cv2.circle(overlay2,Csys,Dia,(10,220,30),1)


    W_en = np.mean(masked_data2) #tæskel for hvor meget ren hvid der må være i billede
    #p = np.mean(masked_data1) #udregning af hvor stor en endel af pakningen der er dårlig.
    Otsu = np.mean(masked_data3)
    #print(W_en)
    peripheral = np.mean(outlineT)

    botArray[0]= (Otsu+peripheral) #gemmer nuværende værdi et det rullende gennemsnit af de første X værdier 
    botArray = np.roll(botArray,1)
    topArray[0]=botArray[4] #gemmer sidste værdi fra det rullende gennemsnit af de første X værdier til et for de sidste X værdier
    topArray = np.roll(topArray,1)


    botSum = np.cumsum(botArray,dtype=float) #sumation af arrayet
    topSum = np.cumsum(topArray,dtype=float) #summation af arrayet
    #print(meanArray)
    #print(x)
    arrayAvgbot = botSum[arraySize-1]/arraySize #finder middelværdi af de første X værdier
    arrayAvgtop = topSum[arraySize-1]/arraySize #finder middelværdi af de sidste X værdier

    arrayDiff=abs(arrayAvgbot - arrayAvgtop) #finder diffecansen mellem de to gennemsnit


    #printning af de udregnet værdier
    #print(en)
    #print(W_en)
    #print(">>><<<")
    #print(arrayAvgbot)
    #print(arrayAvgtop)
    #print(Otsu)
    #print(arrayDiff)
    #print("--------------")
    meanPix(arrayAvgbot,arrayDiff,W_en,Otsu,peripheral) #servo og pass fail function
    Overlay_area_rgb=cv2.cvtColor(Overlay_area,cv2.COLOR_GRAY2BGR)
    #Overlay_area_rgb2=cv2.cvtColor(Overlay_area2,cv2.COLOR_GRAY2BGR)
    vis =np.concatenate((Overlay_area_rgb,image_area),axis=1)
    image_area=cv2.rectangle(image_area,((15+78*ticker),945),(10,920),(0,0,255),-1)
    cv2.putText(overlay, "Threshold val: "+str(Otsu) , (28,30),font,2,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(overlay, "Settling val: "+str(arrayDiff) , (32,58),font,2,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(overlay, "Peripheral val: "+str(peripheral) , (32,932),font,2,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(overlay, "maskgain: "+str(maskGain) , (700,932),font,2,(255,255,255),2,cv2.LINE_AA)
    #visning af de forskellige skidt i processen
    #cv2.imshow("result Bin",vis)
    cv2.imshow("image+area", image_area) 
    cv2.imshow("result", Overlay_area)
    #cv2.imshow("Otsu", Overlay_area2)
    #cv2.imshow("inv_otsu",masked_data3)
    #cv2.imshow("White", masked_data2)


    key = cv2.waitKey(1)
    rawCapture.truncate(0)
    if key == ord("q"): #program shutdown
        break

cv2.destroyAllWindows()