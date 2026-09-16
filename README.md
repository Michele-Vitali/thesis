# 🎓 Vitali Michele's Thesis: A Data-Driven Pipeline for Robotic Manipulation via Vision-Language-Action Models

This repository contains the code used in my thesis project adapted to the needs of the robotic arm on which we will test the VLA Model resulting from all of this.

## 🚀 Get started

Follow these simple steps to ensure the project works smoothly on your pc.

> **Note**:
Ensure you already have python3.10 installed as newer versions will conflict with robosuite and other major libraries!  
If not installed please refer to this link and proceed with the installation of **Python3.10** (not newer versions!)

### 1. Clone the repository
```bash
git clone [https://github.com/Michele-Vitali/thesis.git](https://github.com/Michele-Vitali/thesis.git)
cd thesis/scripts
```
### 2. Create and activate a Virtual Environment
```bash
python -m venv venv
```
If you're using Windows:
```bash
.\venv\Scripts\activate
```
If you're using Linux or Mac:
```bash
source venv/bin/activate
```

### 3. Install the dependencies
In the repository you can find a 'requirements.txt' file which contains all the needed libraries for this project to run, to install them just execute:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the project!
Once you moved to the 'scripts' folder in the project just execute: 
```bash
python main.py
```