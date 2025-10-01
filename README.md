# Explanation
This is a program to train and export a TensorFlow model that can translate one language into another (usually one of those being English). 
I am a beginner at training models and using TensorFlow (or any ML in Python) so it is loosely based on this tutorial from the TensorFlow documentation: https://www.tensorflow.org/text/tutorials/nmt_with_attention
I have added some features to make it possible to run fully locally and use any zip file of data you should want as well as some nicities like logging etc. It is also now fully OOP structured.
I got my data set from http://www.manythings.org/anki/

# Quick Start Guide
1. Ensure you have Python 3 installed on your system
2. Clone the project into your desired directory
3. Setup the virtual environment `python3 -m venv .venv`
4. Install dependencies from the requirements file `pip install -r requirements.txt`
5. Download on of the zips from http://www.manythings.org/anki/
6. Place it wherever you want but you probably want to put it in the root of the project. Take note of the relative path
7. Look at the contents of the archive using 7zip or the tool of your choice, check where the actual txt file is that has the data. Take not of the name
8. Create a .env file in the root of the project with the following structure
   ```
     PATH_TO_ZIP="./your-zip.zip"
     DATA_FILE_NAME="data file name"
   ```
9. Run using `python3 main.py`
10. Enjoy

#Project progess
- [x] Create a ShapeChecker class for handling shape incompatibility issues
- [x] Create a DataProcessor class to load the zip file and create a tf.dataset
- [ ] Add functions to DataProcessor to vectorise and process the data
- [ ] Build the code that will initialise and train the model
- [ ] Add Exporter class that will export the model for use in my language learning projects
- [ ] Integrate into japanese-with-text project 
