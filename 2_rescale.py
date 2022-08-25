import cv2 as cv


# rescale function that rescales the picture
def rescaleFrame(f, scale=0.5):
    # works for images, videos and live videos
    width = int(f.shape[1] * scale)  # get .shape property from the frame, times it by the scale and save it as an int
    height = int(f.shape[0] * scale)
    dimensions = (width, height)
    
    return cv.resize(f, dimensions, interpolation=cv.INTER_AREA)  # return the rescaled frame


# function that only changes the resolution of live videos
def changeRes(width, height):
    # works for live video only
    capture.set(3, width)  # property 3 references width
    capture.set(4, height)  # property 4 references height

    return 0


# Reading images
img = cv.imread('img/pepe.png')  # read image and save in a variable called img

resized_img = rescaleFrame(img)  # pass image into rescale function

cv.imshow('Pepe', img)  # method to show read image ('Name of window',variable_name)
cv.imshow('Pepe resized', resized_img)

# Reading videos
capture = cv.VideoCapture('vid/explosion.mp4')  # 0 for webcam


# use a while loop to read the video frame by frame
while True:
    isTrue, frame = capture.read()

    frame_resized = rescaleFrame(frame)

    cv.imshow('Video', frame)
    cv.imshow('Video Resized', frame_resized)

    if cv.waitKey(20) & 0xFF == ord('d'):  # if 'd' key is pressed, terminate loop
        break

capture.release()
cv.destroyAllWindows()
