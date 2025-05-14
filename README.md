# **Smart Human Detection System**

Real-time face detection and recognition using NVIDIA Jetson Nano, YOLOv4-Tiny, and TensorRT

![image](https://github.com/user-attachments/assets/ed4f9f13-680a-4eac-a868-8e32eaf87630)


=========================================================================

## **📌 Overview:**

This project implements a real-time human detection and face recognition system on an edge device (NVIDIA Jetson Nano).

It combines:

YOLOv4-Tiny for fast human detection.

face_recognition library for identity matching.

TensorRT for optimized inference.

Pushover API for instant mobile notifications.

SQLite/Google Cloud Storage for logging and remote access.

Ideal for security, surveillance, or smart home applications.

=========================================================================

## **🛠 Key Features**

### **Hardware:**

NVIDIA Jetson Nano (4GB).

Raspberry PI CSI camera and/or Night camera

An adapter with 10W output power for MAXN power mode (10W) to provide real-time performance (~20-25 FPS) on jetson.

Active cooling (PWM fan) for thermal management.

=========================================================================

### **Software:**

JetPack SDK (Ubuntu 18.04, CUDA, TensorRT, cuDNN, VisionWorks, GStreamer)

Optimized YOLOv4-Tiny model (TensorRT FP16)

Face recognition using the face_recognition Python library

Push notifications via the Pushover API

=========================================================================

Python libabries used:

| Library  | Version |
| -------- | ------- |
| numpy    | 1.19.5  |
| onnx     | 1.9.0   |
| pycuda   | 2020.1  |
| tensorrt | 8.2.1.9 |
| torch    | 1.8.0  |


=========================================================================



## **📦Installation Steps:**

### **Install Python Libraries:**

    $ sudo apt update && sudo apt upgrade -y
  
    $ pip install face_recognition numpy opencv-python pycuda requests dlib
  
### **Installing PyTorch & TorchVision on Jetson Nano**

  **Install Dependencies:**
    
    $ sudo apt update
    
    $ sudo apt install -y python3-pip libopenblas-base libjpeg-dev zlib1g-dev`
    
    $ pip3 install --upgrade pip

    
  **Install PyTorch(1.8.0) for Jetson** (Pre-built wheels from NVIDIA)
  
    $ wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl
    
    $ pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl

  **Install TorchVision(0.9.0)** (Version must match PyTorch!)

  
    $ sudo apt install -y libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
    
    $ pip3 install torchvision==0.9.0 --no-deps

    $ pip3 install 'pillow<7.0.0'  # Required for compatibility
    

  **Cloning tensorrt_demos**

  
    $ git clone https://github.com/jkjung-avt/tensorrt_demos.git

  **Begin Installing**

  Go to yolo subdirectory inside tensorrt_demos
    
    $ cd tensorrt_demos/yolo

    $ ./install_pycuda.sh

    $ sudo pip3 install protobuf

    $ sudo pip3 install onnx==1.9.0

  Go to plugins subdirectory inside tensorrt_demos

    $ cd tensorrt_demos/plugins
    
    $ make
    
  Go back to yolo subdirectory inside tensorrt_demos
  
    $ cd tensorrt_demos/yolo

    $ ./download_yolo.sh

    $ python3 yolo_to_onnx.py -m yolov4-tiny-416
    
    $ python3 onnx_to_tensorrt.py -m yolov4-tiny-416

  ## **Next Steps**
  
  In the the main folder (tensorrt_demos): **Delete trt_yolo.py**

  Clone the provided trt_yolo.py in this project and place it tensorrt_demos main folder.

  **PushOver Installing Steps**
  
  1.Sign up a new account in the pushover website.
  
  2.Get the **User Key** & **API Token**.
  
  3.Install the application via appstore or playstore.

  **Configure Environment Variables**:

  Add the 2 export function in the ~/.bashrc

    $ gedit ~/.bashrc
  
  export PUSHOVER_TOKEN="**your_api_token_here**"
  
  export PUSHOVER_USER="**your_user_key_here**"

  **Recognition Steps**:

  1.Create a folder called **known_faces** in **tensorrt_demos**
  
  2.Add your images that are needed for recognition (eq. John.jpg)
  
  3.Make Sure the images are Resized to 416x416 or 1280x720


  **Execution Command** (Run the whole code):
  
    $ python3 trt_yolo.py --gstr "nvarguscamerasrc sensor-id=1 ! video/x-raw(memory:NVMM), width=600, height=600, format=NV12, framerate=30/1 ! nvvidconv flip-method=2 ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink" --model yolov4-tiny-416


  ## **📋Results**:
  
  
  Detection Process Using day Camera:
  
  ![image](https://github.com/user-attachments/assets/b07990ce-9106-4b24-8eeb-ff42b11fef53)
  
  Detection Process Using night Camera:
  
  ![image](https://github.com/user-attachments/assets/4b197d04-276e-4587-88f2-f605054d3d95)
  
  
  Detection Process 25 meters away:
  
  ![image](https://github.com/user-attachments/assets/795c26be-a5fb-444f-89f4-958f71c3837f)

  
  Detection & Recognition Process for unkown person (His photo is not placed in the known_faces subdirectory):
  
  ![image](https://github.com/user-attachments/assets/a5fbf60c-fa04-4d98-85c1-6a0b337ce5f6)
  

  Sampled Photo Placed in known_faces subdirectory called weam2.jpg:
  
  ![image](https://github.com/user-attachments/assets/af924460-7b0e-405a-9de1-31416918ad2d)

  
  Detection & Recognition for known person:
  
  ![image](https://github.com/user-attachments/assets/0a4cf651-574b-4c5b-b458-b96053c9beff)





  ## **✍Aditional Feature**:

  The Jetson has 2 CSI Ports for cameras. An Improvement could be the Auto-Switching between the night camera and the day one. The Normal camera operate from 6 am to 18 pm where the night one operates from 18 pm to6 am.

  To add This feature refer to the provided code in this project(AutoSwitchingCamera) to **trt_yolo.py**
  
  **Note:** Make Sure the Night camera is connected to port 0 and the day camera to port 1.

  The function Below is used to run the whole code (Only with AutoSwitchingCamera feature):

    $ python3 trt_yolo.py --model yolov4-tiny-416

  **Note:** The resolution in the AutoSwitchingCamera Code is 416x416 you can adjust it according to your hardware capabilities. 
    
  

  
  

    

    

    
