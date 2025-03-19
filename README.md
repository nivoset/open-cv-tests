# Playing card preliminary preview

This project uses **YOLOv8** and **OpenCV** to detect playing cards using a trained model.

## Setup Instructions

### 1. Create and Activate a Virtual Environment (Recommended)

It's best to use a virtual environment to manage dependencies.

#### **On Windows (cmd or PowerShell)**
```sh
python -m venv venv
venv\Scripts\activate
```

#### **On macOS/Linux**
```sh
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
Once inside the virtual environment, install the required Python packages:

```sh
pip install -r requirements.txt
```

### 3. Run the Test Program
To test the model with your webcam or a video file:

```sh
python display.py
```
> **Note:** On some systems, use `python3` instead of `python`.

### 4. Deactivate Virtual Environment (When Done)
If you need to exit the virtual environment, simply run:

```sh
deactivate
```

