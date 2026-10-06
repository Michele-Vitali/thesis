# 🎓 Vitali Michele's Thesis: A Data-Driven Pipeline for Robotic Manipulation via Vision-Language-Action Models

This repository contains the code used in my thesis project adapted to the needs of the robotic arm on which we will test the VLA Model resulting from all of this.

## 🚀 Get started

Follow these simple steps to ensure the project works smoothly on your pc.

> **Note**:
Due to strict library dependencies (Robosuite, MuJoco, numpy, ...) this project **requires python 3.10** and not any newer versions!  
If you have installed a different version please go to this [link](https://www.python.org/downloads/) and download it.

### 1. Clone the repository
```bash
git clone https://github.com/Michele-Vitali/thesis.git
cd thesis
```
### 2. Create and activate a Virtual Environment
If you're using Windows:
```bash
py -3.10 -m venv venv
.\venv\Scripts\activate
```
If you're using Linux or Mac:
```bash
python3.10 -m venv venv
source venv/bin/activate
```

**Note**:
Once activated you should see the (venv) at the beginning of your terminal line; this indicates that you are currently using the isolated virtual environment as intended.

### 3. Install the dependencies
In the repository you can find a 'requirements.txt' file which contains all the needed libraries for this project to run.  
Make sure you are in the root directory of the project, where the 'requirements' file is located, then run:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the project!
Navigate to the scripts directory and you can start the simulation:
```bash
cd scripts
python main.py
```

## Troubleshoot section
If at any time you encounter any particular problem running the project from Windows you can first follow the troubleshooting section at the bottom of this page of the robosuite official docs (as most of the time the problems are related to that specific library setup): [robosuite docs troubleshoot](https://robosuite.ai/docs/installation.html).
If that's not the case feel free to let me know the problem by opening an issue here: [github](https://github.com/Michele-Vitali/thesis/issues).