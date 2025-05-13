# CPE462Project
## Recycle Sort: Image Classifier for Recycling Identification


### Project Goal
Within the area of image processing and coding, this project aims to explore the topic of image classification. Most people know to recycle items like bottles and cans, but beyond these basics, there’s a lack of awareness about what can and cannot be recycled. The goal of this project was to design a binary image classifier that can help people to determine whether or not an item is recyclable. This project was implemented in Python using TensorFlow and Keras to develop an image classification model, which was trained on a dataset consisting of 12,615 images. Then, a prediction script was written in Python in order to use the model. This script accepts user-uploaded images and returns a recyclability prediction for the item in the image based on the trained model. To provide an accessible user experience, a web interface was designed using Next.js framework with TypeScript in order to allow users to interact with the classification model. The model achieved an accuracy of 93% on the training set and an accuracy of 82% on the validation set, demonstrating its ability to classify items accurately. Therefore, this project resulted in a functional, working tool being developed to predict whether or not an item is recyclable. 

## How to Compile + Run the Code

The first step to compiling and running the code is to clone my github repository 

### Model Training + Evaluation: </br>

Since the model is already trained, the training script does not have to be run again as it will overwrite my trained model with a new version. Similarly, since the dataset has not changed, the results of the evaluation script will not change either making it unnecessary to run. However, I am still going to provide the instructions on how to run these scripts. 

Once the repository is cloned and opened in an IDE (I used VSCode as my IDE) the next step is to activate a python virtual environment and download my dataset.
1. To do this open your terminal and change directories (cd modelTraining) to the modelTraining folder of the project. If not already in the project folder you will first have to change directories into the project folder then use cd modelTraining.
2. Type the following command to create a new virtual environment: “python -m venv myenv”. You only have to create a new environment once. 
3. To activate your environment type: “source myenv/bin/activate”. 
4. Then go to the following link to download my dataset: https://www.kaggle.com/datasets/3d4ff694190d7a677ab09e96706aa76816e76d9ef59f357f233f837317fc5250. Once the dataset is downloaded, unzip the file and move the data folder into the modelTraining folder of the project. The structure of the dataset folders should be as followed: 

After the setup is accomplished, the training and evaluation scripts are ready to be run. 
1. Ensure your terminal is still in the modelTraining directory (use cd command to get there if not already there)
2. Type “pip install -r requirements.txt” in the terminal to download the necessary dependencies for the scripts
3. Run the desired script by typing “python training.py” in your terminal to run the training file or type “python evaluate.py” in your terminal to run the evaluation file. 
4. Once finished running, you can deactivate your virtual environment by typing deactivate in the terminal.

### Running the Application : </br>

To run my actual image classifier application you will need to run the frontend and backend at the same time in two separate terminals. 

To get the backend running follow these steps: 

1. In your terminal, change directories (cd) into the backend app folder of the project. If you are currently in the project directory (CPE462Project) you would type cd src/backend/app to get to the backend folder. If not in the project directory, change directories to that first then type cd src/backend/app.
2. Install necessary dependencies by typing “pip install -r requirements.txt” in the terminal.
3. Change directories into the backend folder. To do this type “cd ..”.
4. Run the backend server by typing “uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload”. You will get some terminal output that indicates the backend is running if done properly. 

</br>

While the backend is running, open another terminal to run the frontend. Follow these steps: 

1. In your terminal, change directories (cd) into the frontend folder of the project. If you are currently in the project directory (CPE462Project) you would type cd src/frontend to get to the frontend folder. If not in the project directory, change directories to that first then type cd src/frontend.
2. Install necessary dependencies by typing “npm install” in the terminal. Then, ensure your computer is using node version 20 or later by typing “nvm install 20” then “nvm use 20” in the terminal. 
3. Run the frontend server by typing “npm run dev” in the terminal. You will get some terminal output that indicates the frontend is running. Go to the local server link: http://localhost:3000 and you will be able to use my web interface for my project. You can upload images to test my image classification model. Note, that there is a folder in the repository called WebAppTestingImages that contains images of various items that can be used to test my image classifier.
</br>

When done with the website, you can close the tab. Then click Ctrl-C in both the frontend and backend terminals to stop them from running. 


## Project Report
Click the following link to view a more detailed report of my project: [Project Report](https://github.com/ardensentak/CPE462Project/blob/main/projectReport/CPE462ProjectReport.pdf)