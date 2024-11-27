# Detection of Solidification Defects During Continuous Casting in Steelmaking  
## Modeling for the Company [EBDS ENGINEERING](https://be.linkedin.com/company/ebds-engineering)

### Table of Contents  
1. [Context](#Context)  
2. [Usage](#Usage)  
3. [File Descriptions](#file-descriptions)  
4. [Contributors](#Contributors)  

---

## Context: Continuous Casting Defect Detection
Continuous casting is a key process in steel production, where molten steel is poured into a cooled mold to solidify into semi-finished products. One significant challenge in this process is the occurrence of "sticking," where solidified steel adheres to the mold, leading to cracks and defects. This phenomenon can cause costly production stoppages, equipment damage, and product quality degradation. To mitigate this, advanced monitoring systems using fiber optics have been adopted by EBDS Engineering to detect subtle temperature variations in the mold that often precede sticking events. In this project, we introduce our approach of using linear regression to detect these phenomena and prevent the formation of defects during the production process.


## Usage  
To set up the environment for this project, the following packages must be installed:  
* numpy
* pandas
* matplotlib
* scikit-learn
* seaborn

they can be installed using the following cmd command:
```bash
pip install numpy pandas matplotlib scikit-learn seaborn
```

## File Descriptions

we have the following files:

### [`main.py`](./main.py)
The main.py file is used to analyze specific frames in a dataset, used for detailed, frame-by-frame analysis from a given date and time.

#### [`test_all.py`](./test_all.py)
The test_all.py file autonomously analyzes the entire dataset. It is designed to go through the dataset frame-by-frame, analysing each frame and only displaying the dashboard if any alerts are detected.

#### [`fun.py`](./fun.py)
The fun.py file contains utility functions that are used by both the main.py and test_all.py files. These functions handle data processing tasks, such as linear regression, temperature analysis, and other reusable operations that support the core functionality of the project.

---
## Contributors
* [Lison Van Asbrouck](https://github.com/username)
* [Nour-El Houda Teber](https://github.com/username)
* [Cyril Joliet](https://github.com/username)
* [Abdelali Dssam](https://github.com/username)

