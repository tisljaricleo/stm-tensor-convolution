# stm-tensor-convolution :collision:  
This repo is in work-in-progress status.  
## Info
This code will be the result of the research related to training the Convolutional Neural Network (CNN) to classify the traffic states of the large traffic networks as "anomalous" or "normal" (binary classification). To represent the traffic state, a novel traffic data model called **Speed Transition Matrix (STM)** is used. More on STMs can be found [here](https://medium.com/analytics-vidhya/speed-transition-matrix-novel-road-traffic-data-modeling-technique-d37bd82398d1).  
### Here are the main steps of the method:
1. Preprocess GPS data (save in database as routes)
2. Create transitions from routes (needed for STM computation)
3. Compute STMs
4. Segment map using grid-based segmentation
5. For every cell in the grid extract all STMs and construct a tensor (in other words, every cell will be represented with one tensor)
6. Extract characteristic matrices for every cell using the tensor decomposition method
7. Label characteristic matrices using the center of mass for every STM (based on [this](https://www.researchgate.net/publication/344138884_Traffic_State_Estimation_and_Classification_on_Citywide_Scale_Using_Speed_Transition_Matrices) article)
8. Train CNN with labeled examples
9. TODO: validation
## More info
If you are interested in this topic please contact me:
[Leo Tisljaric](https://www.linkedin.com/in/leo-ti%C5%A1ljari%C4%87-28a56b123/)
## Requirements
1. Install Python (3.8 recommended) [Download link](https://www.python.org/downloads/).
2. Install required packages from `requirenments.txt` using [virtual environment](https://docs.python.org/3/tutorial/venv.html).
