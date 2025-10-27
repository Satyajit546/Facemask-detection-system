import streamlit as st
import numpy as np
import cv2
import os
import tempfile
from keras.models import load_model
from keras.utils import load_img,img_to_array
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
facemodel=cv2.CascadeClassifier('face.xml')
maskmodel = load_model('mask.keras', compile=False)
st.title("Face mask Detection system")

choice=st.sidebar.selectbox('My Menu',('Home','Image','Video','Camera'))
if(choice=='Home'):
    
    st.image("front.jpg",caption="AI-Powered Mask Detection")

    st.markdown("""
    ### Welcome to the Face Mask Detection App
    Upload an image, video, or use your camera to detect if a person is wearing a mask.
    """)
elif(choice=="Image"):
    st.image("https://scitechdaily.com/images/COVID-Mask-Fit.gif")
    st.markdown(
    """
    <p style='color:#C70606; font-weight: bold; font-size: 20px;'>
        No Mask No Safety!
    </p>
    """,
    unsafe_allow_html=True
)
    file=st.file_uploader('Upload Image',type=['jpg','png','jpeg'])
    if file:
        b=file.getvalue()
        d=np.frombuffer(b,np.uint8)
        img=cv2.imdecode(d,cv2.IMREAD_COLOR)
        
      
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = facemodel.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x,y,w,h) in face:
            crop_face=img[y:y+h,x:x+w]
            cv2.imwrite('temp.jpg',crop_face)
            crop_face=load_img('temp.jpg',target_size=(150,150,3))
            crop_face=img_to_array(crop_face)
            crop_face=np.expand_dims(crop_face,axis=0)
            pred=maskmodel.predict(crop_face)[0][0]
            if pred ==1:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),4)
                label = "No Mask"
            else:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),4)
                label = "Mask"
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,3,(255, 255, 255), 2)
        st.image(img,channels='BGR',width=400)

elif (choice == "Video"):
    file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    window = st.empty()

    if file:
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        vid = cv2.VideoCapture(tfile.name)

        while vid.isOpened():
            ret, frame = vid.read()
            if not ret:
                break

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facemodel.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                # Crop and preprocess face
                face_crop = frame[y:y+h, x:x+w]
                face_crop = cv2.resize(face_crop, (150, 150))
                face_crop = img_to_array(face_crop)
                face_crop = np.expand_dims(face_crop, axis=0)

                # Prediction
                pred = maskmodel.predict(face_crop)[0][0]

                # Threshold (adjust if inverted)
                if pred > 0.5:
                    color, label = (0, 255, 0), "Mask"
                   
                    
                else:
                    color, label = (0, 0, 255), "No Mask"

                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Display processed frame
            window.image(frame, channels="BGR",width=400)

        vid.release()
        os.remove(tfile.name)


elif (choice=="Camera"):
    st.write("Use '0' for default webcam or paste IP camera link (e.g. http://192.168.x.x:8080/video)")
    k = st.text_input('Camera Source')
    btn = st.button('Start Camera')
    window = st.empty()
  

    if btn:
        # ✅ If no webcam or no URL, fall back to demo video
        if k.strip() == "" or k == "0":
            st.warning("No webcam detected. Using demo video instead.")
            video_path = "demo.mp4"  # make sure this file exists in your project folder
            if not os.path.exists(video_path):
                st.error(f"Demo video not found at {video_path}. Please add a sample video file.")
                st.stop()
            vid = cv2.VideoCapture(video_path)
        else:
            vid = cv2.VideoCapture(k)

        # Check if video source opened successfully
        if not vid.isOpened():
            st.error("Failed to open camera or video source. Check URL or file path.")
            st.stop()

        while vid.isOpened():
            flag, frame = vid.read()
            if not flag:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facemodel.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                face_crop = frame[y:y+h, x:x+w]
                cv2.imwrite('temp.jpg', face_crop)
                img_face = load_img('temp.jpg', target_size=(150, 150, 3))
                img_array = img_to_array(img_face)
                img_array = np.expand_dims(img_array, axis=0)
                pred = maskmodel.predict(img_array)[0][0]

                if pred > 1:
                    color, label = (0, 255, 0), "Mask"
                else:
                    color, label = (0, 0, 255), "No Mask"

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            window.image(frame, channels="BGR")

        vid.release()


