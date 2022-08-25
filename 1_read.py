import cv2 as cv

'''
# Reading images
img = cv.imread('img/pepe.png')  # read image and save in a variable called img

cv.imshow('Pepe', img)  # method to show read image ('Name of window',variable_name)

cv.waitKey(0)  # wait for a specific delay for a keyboard key to be pressed, 0 is infinite
'''

# Reading videos

capture = cv.VideoCapture('vid/explosion.mp4')  # 0 for webcam

# use a while loop to read the video frame by frame
while True:
    isTrue, frame = capture.read()
    cv.imshow('Video', frame)
    
    if cv.waitKey(20) & 0xFF == ord('d'):  # if 'd' key is pressed, terminate loop
        break

capture.release()
cv.destroyAllWindows()
    


