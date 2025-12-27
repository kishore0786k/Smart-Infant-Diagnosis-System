# Smart Infant Diagnosis System

This project is based on the idea that an infant’s cry carries useful information about their condition. Since infants cannot communicate verbally, cry analysis can help in understanding possible discomfort or needs at an early stage.

In this project, infant cry audio signals are processed and analyzed using Python. Relevant audio features are extracted from the cry signals, and a machine learning model is used to classify different cry patterns. The system is intended to support observation and learning, not to replace medical diagnosis.

---

## Why this project?
Newborn monitoring mainly depends on human interpretation, which can vary from person to person. This project explores how data-driven techniques can assist in understanding infant cry behavior in a more consistent way.

---

## What this system does
- Takes infant cry audio as input  
- Preprocesses the audio and removes noise  
- Extracts important acoustic features  
- Uses a machine learning model to classify cry patterns  
- Displays classification results and basic performance metrics  

---

## Technologies used
- Python  
- NumPy, Pandas  
- Librosa for audio processing  
- Scikit-learn for machine learning  
- Matplotlib / Seaborn for visualization  

---

## Project structure
Smart-Infant-Diagnosis-System/
├── data/ # Dataset (not included if large)
├── src/ # Source code files
├── models/ # Trained model files
├── results/ # Output graphs and results
├── requirements.txt
└── README.md


---

## How to run the project
1. Clone the repository  
``'bash
git clone https://github.com/kishore0786k/Smart-Infant-Diagnosis-System.git

2.Install required libraries
pip install -r requirements.txt

3.Run the main program
python main.py

---

Output:
Classified infant cry patterns
Accuracy and evaluation graphs
Visual representation of results

Limitations:
The model depends on the quality and size of the dataset
Results are for learning and analysis purposes only
Not suitable for real-world medical decision making

Disclaimer:
This project is developed for educational and research purposes. It should not be considered as a medical diagnosis system.

Author
Kishore
