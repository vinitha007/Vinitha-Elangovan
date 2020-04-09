# Identification of Handwritten Letters Using Persistent Homology

This project uses methods from Topology to classify handwritten letters. The letter is put onto a 10x10 grid, and a 1 is marked if any part of the letter touches the grid square. This ultimately leaves us with a 1x101 array, where the first number is an index identifying the letter, and the next 100 are values from our 10x10 grid.

```
Sample: Data of letter ‘I’ from dataset. (letterIndex: 9)
9,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0
```

## Introduction
This project uses a method from Topology called **Persistent Homology** to classify letters. Persistent Homology is a nice technique for classifying because it requires no training and is relatively robust. For example, our model gets roughly ~70% accuracy on a noisy dataset of capital leters.

### Persistent Homology:
Persistent Homology is an algebraic method of calculating topological features from a given set of data. The theory itself is extremely mathematically complex, but for our purposes, it serves to think of it as 'pouring water' into areas of different elevation. At first, all water will fill in the lowest elevation areas, creating 'lakes.' As more 'water' is poured in, more 'lakes' are formed, and occasionally, two 'lakes' merge into a single entity.

We refer to these lakes as **classes**, the numerical value where a class starts forming is called 'time of birth.' When two classes merge, we consider them as a single class going forward, by convention, we keep the class that has lasted the longest (AKA the most persistent class) and we record the other class as 'dying,' the value where two classes merge like this is then called 'time of death' for the class we decided to terminate.

For more in-depth mathematical detail about Persistent Homology see [Persistent Homology - a Survey](https://www.maths.ed.ac.uk/~v1ranick/papers/edelhare.pdf) written by Herbert Edelsbrunner and John Harer.

### Scans:
'Scans' are the start of our model, although the name is a bit misleading. In general, all these scans do is transform our 1x101 arrays for each letter into a 10x10 array by ignoring our letter index and just taking the values from our 10x10 gridview of the letters. These 'scans' then change the values of our 10x10 array depending on how our letter should be scanned. Lower values represent 'low elevation.' So for a left-to-right scan (LRScan), low elevation values would be on the left, with higher values as you go across the columns of the array. Our model then uses the lower_star_img function from the Ripser library to calculate classes by 'pouring' one unit of water at a time, at the end it will return some number of 1x2 arrays, each array represents a class, where the first value is the 'time of birth' of that class, and the second value is the 'time of death.'

### Classification
Finally to classify our letters, we calculate the total number of classes and the average length of classes for a given scan and letter. These values are appended into a list of dictionaries, where listIndex = letterIndex. We consider these to be the 'vectors' for our handwritten letters. Then, as we get new test data we scan it, calculate the vector, and then calculate the minimum distance between our test vector and our original list of values. Our program then guesses a letter based on this minimum distande for classification.

## Getting Started

This is a standalone project, to get started all you need to do is downloade the Jupyter Notebook and run it on a Python 2 kernel. Functions in the notebook should be well documented enough to understand what they do.

### Prerequisites

To run the Jupyter Notebook you will need to install persim and ripser. This repository also contains a copy of our dataset, so we clone the repository as well. On top of installations, you'll need to import a number of libraries. Here's the code we currently use at the beginning of our program.

```
%%capture
!pip install ripser
!pip install persim
!git clone https://github.com/SWHulbert/TDA-Project.git
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rnd
from scipy import ndimage
import PIL
from persim import plot_diagrams
from ripser import ripser, lower_star_img
```

## Running the tests

Our program tests on data called dataT, (dataT is dataTest for short), dataT is an exact copy of our original data, but 'noise' is added by flipping bits, either from 0 to 1, or 1 to 0, with a certain probability (the percentage can be specified). Empirically, we've found 4% noise to be a good test case, where the original letters are still recognizable, but also clearly distinct from our original data.

Our program further 'denoises' the data, by removing any bit or 1, that is not connected to any other 1. It does this by simply checking all values around a specific bit and summing them, if the sum is > 0, then our bit is connected, otherwise it is unconnected and removed from the data.

To see how accurate our program is for a specific letter, you can run the perCorrect() function. This function require a letterIndex = a, noisePercent = b, numberofGuesses = c, and denoiser = TRUE or FALSE, by default noisePercent = 0.04, numberofGuesses = 10, denoiser = TRUE. After running, the perCorrect function will spit out a 2-tuple, (letterIndex, percent correct guesses/total number of guess)

To see how our model does on all capital letters in the English Alphabet, you can run code like this:

```
for y in range(26):
  print(y, perCorrect(y, 0.02, 50))
```

Here you can see we specificed the noisePercent = 0.02, and numberofGuesses = 50.

## Optimization of the Model

Originally, our model ran with 8 scans, and record the values of total number of classes, and average length of classes. However, as we added more scans, we noticed that accuracy, especially for specific letters such as A, D, O, and Y went down. Through the simple gumshoe move of testing each scan individually, then each pair of scans individually, each 3-tuple of scans, etc. we removed scans that were not contributing to our model.

To do this, we first picked the scan that classified the most letters. Then we compared a model with scan X, to a model without scan X, and calculated average accuracy over all letters in the alphabet. If average accuracy went up by a significant ammount (we used "Did average accuracy increase by 3% or more" as a cutoff question for our model). After running this test, we removed 2 scans that were below the cutoff, decreasing the overall runtime of our program.

Furthermore, we added a 'denoiser' functions, which removes any bit of text that's not connected to other text. It does this by going through each entry in our 1x101 array, taking any entry with the value '1' and then checking if any of the pixels around  that given pixel also have a value of '1.' If not, it replaces the pixel with a value of '0' and moves on.

## Built With

* [Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb) - The web framework used

## Authors

* **Seth Hulbert**
* **Yajun Fu**
* **Vinitha Elangovan**
* **Keerthi Padamata**
* **Nitheen Jammula**
* **Alexis Fiddemon**
