# Ditto

## Dataset

### Step 1: Raw Data

We use CodeTransOcean: https://github.com/WeixiangYAN/CodeTransOcean/tree/main (Can be downloaded as Zip file using Google Drive link). (MultiLingualTrans, NicheTrans, LLMTrans).

For each one of these folders, keep the json files in `/raw_data`.

Alternatively, click <a href=""></a> to directly download the `raw_data` zipfile.

### Step 2: Data

The raw json file contains codes from several languages. We need to extract paired and unpaired C and C++ languages from the data. For this we use `extract.py`. Please ensure that you have the folder `/data` along with empty files `paired_c.txt`, `paired_cpp.txt`, `unpaired_c.txt`, `unpaired_cpp.txt`. Running the `extract.py` script will populate these empty files.

Alternatively, click <a href=""></a> to directly download the `data` zipfile.

After this step, we obtain the following,

Paired C and C++ data: `2492`
Unpaired C data = `76020`
Unpaired C++ data = `81262`

However, we observe that the dataset contains several duplicates. We found the same code snippet repeat 9 times in the datset. After removing all the duplicates, we get the following:

Unique entries with both C and C++: `374`
Unique entries with only C: `882`
Unique entries with only C++: `773`

The steep reduction in dataset showed us the level of duplication in the raw data, and hence we decided to explore other datasets as well.

## Step 3: Cleaned Data

The data that we obtained from step 2 contains several C / C++ code snippets (paired and unpaired). However, not all of the codes can be executed. Some of the code snippets just represent some function defination, while others use some outdated libraries such as `graphics.h`. To avoid using such data, we need to perform cleaning. 

Cleaning step is a very expensive step as it involves going through each snippet of code and then compiling it to check if this code can be used as data or not. This process can take around 7 - 8 hours as we have approximately `150000` code snippets to evaluate.

For cleaning the data, run `preprocess.py`. Please ensure that you have the folder `/clean_data` along with empty files `paired_c.txt`, `paired_cpp.txt`, `unpaired_c.txt`, `unpaired_cpp.txt`. Running the `preprocess.py` script will populate these empty files.

In addition to this, we also require clang. Please follow the installation steps for installing clang.

Alternatively, click <a href=""></a> to directly download the `clean_data` zipfile.

After this step, we obtain the following,

Paired C and C++ data: ``
Unpaired C data = ``
Unpaired C++ data = ``

Here, our dataset shrinks a lot.

## Step 4: Tokenization 

The code snippet can be converted into a series of tokens using lexer and into an AST using parser. Hence, we use `libclang` to convert our code into tokens and AST. Once we have our `/clean_data` folder, we can run `tokenization.py`.

## Installation

Make sure that all the requirements in `requirements.txt` are installed

`sudo apt-get install clang`

`sudo apt-get install libclang-dev`

`sudo apt install libclang-dev python3-clang`

You will also need to change the path in `set_library_file()` in `tokenization.py` to point to `libclang.co`. The below command will return the path to `libclang.co`

``find /usr/lib -name "libclang.so*" 2>/dev/null`

For example, `/usr/lib/llvm-18/lib/libclang.so`