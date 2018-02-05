from tkinter import *
from tkinter import font
from PIL import Image, ImageTk
from tkinter import filedialog
import cv2 as cv

faceclassifier = cv.CascadeClassifier("classifiers/haarcascade_frontalface_default.xml")
eyeclassifier = cv.CascadeClassifier("classifiers/haarcascade_eye.xml")
smileclassifier = cv.CascadeClassifier("classifiers/haarcascade_smile.xml")
#destroy application
def destroy():
    app.destroy()
# get face eyes and smile
def detectAttributes():
    #get image using file dialog
    path = filedialog.askopenfilename()
    if(path):
        image = cv.imread(path)
        dic = (400, int(image.shape[0]*(400.0/image.shape[1])))
        #print(image.shape[0]*r)
        image = cv.resize(image, dic, interpolation=cv.INTER_AREA)
        copyimage = image.copy()
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        faces = faceclassifier.detectMultiScale(
    		gray,
    		scaleFactor=1.1,
    		minNeighbors=5,
    		minSize=(30, 30)
	       )
        cv.putText(image,"Original Image", (30,30),cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 55),2)
        for (x,y,w,h) in faces:
            cv.rectangle(copyimage, (x,y), (x+w,y+h), (0,255,0),2)
            cv.putText(copyimage,"Modified Image", (30,30),cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,55),2) #Draw the text
            roi_gray = gray[y:y+h, x:x+w]
            smileEyeImage = copyimage[y:y+h, x:x+w]

            eyes = eyeclassifier.detectMultiScale(roi_gray)
            smile = smileclassifier.detectMultiScale(
                roi_gray,
                scaleFactor= 1.1,
                minNeighbors=30,
                minSize=(50, 50)
            )

            for (ex,ey,ew,eh) in eyes:
                cv.rectangle(smileEyeImage,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            for (x, y, w, h) in smile:
            	cv.rectangle(smileEyeImage, (x, y), (x+w, y+h), (0, 255, 0), 2)


        image = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        copyimage = Image.fromarray(cv.cvtColor(copyimage, cv.COLOR_BGR2RGB))

        image = ImageTk.PhotoImage(image)
        copyimage = ImageTk.PhotoImage(copyimage)
        return image,copyimage
    else:
        return False, False
#set images to panel
def setPanel():
    global leftPanel, rightPanel
    image,copyimage = detectAttributes()
    if image:
        if leftPanel is None or rightPanel is None:
            leftPanel = Label(image=image)
            leftPanel.image = image
            leftPanel.pack(side="left", padx=10,pady=10)

            rightPanel = Label(image=copyimage)
            rightPanel.image = copyimage
            rightPanel.pack(side="right", padx=10,pady=10)
        else:
            leftPanel.configure(image=image)
            leftPanel.image = image
            rightPanel.configure(image=copyimage)
            rightPanel.image = copyimage

#gui window
app = Tk()
font = font.Font(family="Noto Sans Regular", size=12, weight="normal")
app.title("Human Face Detector")
app.minsize(width=200, height=50)
# both panel to display image
leftPanel = None
rightPanel = None
#frame to set button
btnframe = Frame(app)
btnframe.pack(side=BOTTOM)
#button to choose image
btn = Button(btnframe, text="Choose Image", font=font, command=setPanel)
btn.pack(side=LEFT, fill=BOTH, expand=NO, padx=10, pady=10)
#button to exit app
exit = Button(btnframe, text="Exit", font=font, command=destroy)
exit.pack(side=RIGHT, fill=BOTH, expand=NO, padx=10, pady=10)
# loop the app
app.mainloop()
