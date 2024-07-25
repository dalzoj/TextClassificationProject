
# Text Classification Project

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

This project is designed to classify text data into predefined categories using machine learning techniques. The primary objective is to preprocess the text data, build features, train a model, and evaluate its performance.

## Project Structure
```
.
├── data
│ ├── extends
│ │ └── words_to_ignore.txt
│ ├── processed
│ │ └── processed_data.csv
│ ├── raw
│ │ └── raw_data.csv
├── notebooks
│ ├── build_features.ipynb
│ ├── complete.ipynb
│ ├── data_processing.ipynb
│ ├── load_data.ipynb
│ ├── modeling.ipynb
│ ├── others
│ │ └── *.ipynb
├── src
│ ├── data
│ │ ├── load_data.py
│ │ └── preprocess.py
│ ├── features
│ │ └── build_features.py
│ ├── objects
│ │ └── *.pkl
│ ├── utils
│ │ ├── general_path.py
│ │ └── helper_functions.py
├── README.md
└── requirements.txt
```
  

## Installation

  

### Prerequisites
- Python 3.8+
-  `pip` package manager
 
### Steps
1.  **Clone the repository**
```
git clone https://github.com/dalzoj/TextClassificationProject.git
cd TextClassificationProject
```

2.  **Create a virtual environment**
```
python -m venv venv
source venv/bin/activate

# On Windows use
venv\Scripts\activate
```

3.  **Install the required packages**
```pip install -r requirements.txt```

4.  **Download NLTK stopwords**
```python -c "import nltk; nltk.download('stopwords')"```
  

## Usage

### Running the Preprocessing Script
* `--pt` specifies the type of preprocessing to apply (`normal` or `spellchecker`).

To preprocess the data with normal processing, execute:
```python src/data/preprocess.py --pt normal```
or
```python src/data/preprocess.py```
Then, to preprocess the data using spellchecker processing, execute:
```python src/data/preprocess.py --pt spellchecker```

### Running the Builder Features Script
* `--f` specifies the data name to process.

To build the features with normal processing, execute:
```python src/features/build_features.py```
or
```python src/features/build_features.py --f sc_pre_d_s28594```

### Running the Model Training
The script for training models accepts several arguments to customize the training process. Below are the descriptions and usage examples for each argument.
* `--tr` specifies the type of textual representation to use.
* `--f` specifies the name of the dataset to process.
* `--m` specifies the name of the model to train.
* `--pg` specifies the name of the YAML file containing the model hyperparameters.

**Example Usage**
To train a model with the default settings, execute:
```python src/train/train_model.py```
This uses:
* Textual Representation: `tfidf`
* Dataset: `n_pre_d_s28817`
* Model: `RandomForestClassifier`
* Hyperparameter Grid File: `classification_param_grid_small`


To customize the textual representation, dataset, model, or hyperparameter grid file, you can specify them as follows:
**Example with Custom Textual Representation and Dataset**
```python src/train/train_model.py --tr bow --f sc_pre_d_s28594```
This uses:
* Textual `Representation: bow`
* Dataset: `sc_pre_d_s28594`
* Model: `RandomForestClassifier` (default)
* Hyperparameter Grid File: `classification_param_grid_small` (default)


### Jupyter Notebooks
You can also explore the data and train models using the Jupyter notebooks provided in the notebooks directory. To start Jupyter Notebook, run:
```jupyter notebook```

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

### Steps to Contribute
1. Fork the repository
2. Create your feature branch ```git checkout -b feature/AmazingFeature```
3. Commit your changes ```git commit -m 'Add some AmazingFeature'```
4. Push to the branch ```git push origin feature/AmazingFeature```
5. Open a pull request

  
  

## License
This project is licensed under the MIT License - see the LICENSE file for details.

  

## Contact
For any questions or suggestions, feel free to reach out to:
* Diego Lizarazo
* Email: imdlizarazo@gmail.com.com
* GitHub: dalzoj