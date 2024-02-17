# HTML file Search and Process Using TDF-IF

This project involves searching HTML documents for relevant information using TF-IDF and user-input queries.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Structure](#structure)



## Introduction

This project contains multiple python scripts for processing HTML documents and searching for releveant html information based on the tdf-if data used in these scripts. It makes use of multiple python libraries to tokenize, vectorize and parse html documents


## Requirements

- Python
- Libraries needed: BeautifulSoup, scikit-learn, nltk

## Installation

1. Install required libraries:

    ```bash
    pip install beautifulsoup4 
    pip install scikit-learn 
    pip install nltk
    ```

2. Run the files

## Usage

1. Make sure HTML files are stored in a folder.
2. Update the `videogames` variable in `main.py` with the path to your HTML files. (around line 89-91)
3. Ensure that both .py files are store together in the same directory
3. Run the script in dedicated terminal:

    ```bash
    main.py
    ```

4. Input search queries when prompted.

## Structure

- `Htmlread`: Function to read HTML files and extract title and content.
- `description`: Function to find the main paragraph in the text.
- `preprocess_query`: Function to preprocess user queries for query expansion.
- `search`: Function to perform search using TF-IDF.
- `processdata`: Function to read HTML files, create a dictionary for each, and return a list of dictionaries.
- Other necessary libraries and modules.

seperate file tokenizor.py

- `tokenize`:Function to tokenize and apply all the pre-processing techniques


