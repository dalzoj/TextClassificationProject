
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
├── notebooks
│ ├── .ipynb_checkpoints
│ └── others
│ ├── build_features.ipynb
│ ├── complete.ipynb
│ ├── data_processing.ipynb
│ ├── load_data.ipynb
│ └── modeling.ipynb
├── src
│ ├── data
│ │ ├── load_data.py
│ │ └── preprocess.py
├── .env
├── .gitignore
├── log.log
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
source venv/bin/activate # On Windows use `venv\Scripts\activate`
```

3.  **Install the required packages**
```
pip install -r requirements.txt
```

4.  **Download NLTK stopwords**
```
python -c "import nltk; nltk.download('stopwords')"
```
  

## Installation

### Configuration
Create a .env file in the root directory with the necessary environment variables. Refer to .env.example for the structure.

### Running the Preprocessing Script
To preprocess the data with normal processing, execute:
```
python src/data/preprocess.py --processing_type normal
```
or
```
python src/data/preprocess.py
```
Then, to preprocess the data using spellchecker processing, execute:
```
python src/data/preprocess.py --processing_type spellchecker
```


## Jupyter Notebooks
You can also explore the data and train models using the Jupyter notebooks provided in the notebooks directory. To start Jupyter Notebook, run:
```
jupyter notebook
```

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