# TAD-Lactuca
TAD-Lactuca is a tool to predict Topologically Associated Domains(TAD) boundary using histone marks information. It was written in Python language, using Random Forests(RF) and Multilayer Perception(MLP).

## Overview
+ The `data` fold contains the origin data (expect the bigwig file, the detail can be found in paper) 
+ The `src` fold contains all scripts, such as data processing, modeling and plot.
+ The `cache` fold contains the main result and some processed the multiple resolution feature signal, such as the  `bin's` length range(`40kb,20kb,10kb`), the `bin's` number range(10,8,6). 
+ The `pic` fold contains the all the picture, such as the ROC Curve, the signal pattern of different feature and so on.

## Dependency

+ Python3.*
+ Numpy
+ Pandas
+ Tensorflow / Tensorflow-gpu
+ Keras
+ Scikit-Learn
+ Matplotlib


## Preprocessing
The `data` fold contains the original files, the signals of different type histone were calculate by `bedtool`, the details can be seen from `./src/utils/get_signal_plot.py`

## Usage
If you want to use the example data, you can run the script as:
```bash
python3 tad_lactuca.py
```

If you want to use your own data, please run the script as:
```bash
python3 tad_lactuca.py [the path of your data]  
```

### Input
1. The feature data of the locus you want to know is boundary or not
2. Only the base location you want to know is in the domain or boundary
### Output 
+ The probability of the input data belong to each category.
+ The signal of the locus.

