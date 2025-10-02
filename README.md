# Ditto

<p align="center">
  <img src="https://media.tenor.com/4wt81D8xUEwAAAAM/ditto-pokemon.gif" alt="ditto">
  <br>
  <small><i>Image source: https://tenor.com/search/ditto-pokemon-gifs</i></small>
</p>

## Introduction

Ditto is a transpiler that converts code between C++ and C languages. The name comes from the Pokémon Ditto, which can copy any other Pokémon exactly. Our goal is to build an AI transpiler that can translate code across different programming paradigms, no matter the language. For our project, we focus on C++ to C conversion and vice versa.

## Datasets

For Ditto, we require unpaired C, unpaired C++, and paired C and C++ code files. In order to obtain this, we use several existing datasets. 

## Setting up

Create a folder called `final_data` along with empty files `paired_c.txt`, `paired_cpp.txt`, `unpaired_c.txt`, `unpaired_cpp.txt` in it. All the indiidual datasets will append their data in these files. Now follow the steps below.

We also need to create a virtual enviroment. Steps to create the same are given here:

`python3 -m venv ditto`

`source ditto/bin/activate`

` pip install -r requirements.txt `


### Step 1: Individual data directories

#### Transcodeocean

We use CodeTransOcean: https://github.com/WeixiangYAN/CodeTransOcean/tree/main (Can be downloaded as Zip file using Google Drive link). (MultiLingualTrans, NicheTrans, LLMTrans).

For each one of these folders, keep the json files in `transcodeocean/raw_data`.

For each one of these folders, keep the json files in `transcodeocean/raw_data`.

The raw json file contains codes from several languages. We need to extract paired and unpaired C and C++ languages from the data. For this we use `extract.py`. Please ensure that you have the folder `/data` along with empty files `paired_c.txt`, `paired_cpp.txt`, `unpaired_c.txt`, `unpaired_cpp.txt`. Running the `extract.py` script will populate these empty files.

After this step, we obtain the following,

Paired C and C++ data: `2492`
Unpaired C data = `76020`
Unpaired C++ data = `81262`

However, we observe that the dataset contains several duplicates. We found the same code snippet repeat 9 times in the datset. After removing all the duplicates, we get the following:

Unique entries with both C and C++: `374`
Unique entries with only C: `882`
Unique entries with only C++: `773`

The steep reduction in dataset showed us the level of duplication in the raw data, and hence we decided to explore other datasets as well. 

#### Kaggle

Because the data obtained from transcodeocean is very less, we use other resources. Here, we use https://www.kaggle.com/datasets/dianavostrova/formai-v2-dataset-without-clones-7z dataset to obtain C code. 

Extract all the files inside `kaggle/raw_data` directory. There are a total of approximately 3 L code files here. Since our requirement in not that much, we will randomly sample the fles from this folder.

Here, we require `/data` directory with single `unpaired_c.txt` file.

Now, running extract.py will randomly sample N codefiles from the data and add them to data directory after processing. This dataset has no duplicates.

#### Hugging face directories

C++ data is fetched from [cpp_200k](https://huggingface.co/datasets/kloodia/cpp_200k) and [stack_edu_cpp](https://huggingface.co/datasets/hongliu9903/stack_edu_cpp). 

For each dataset, run `extract.py` file with the respective dataset name. For cpp_200k, the dataset name is kloodia/cpp_200k and for stack_edu_cpp, the dataset name is hongliu9903/stack_edu_cpp. 

This will extract all the cpp files from the dataset.

After extracting the files from the dataset, we need to filter out non-compilable files from the compilable ones. One common practice we noticed is that most of the c++ files include the header `bits/stdc++.h`. This is an amalgation of all the popular headers. However, a standard g++ compiler doesn't recognize this. And hence, we replaced this with the following popular headers in `extract.py` itself. 

```
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <numeric>
#include <queue>
#include <stack>
#include <deque>
#include <utility>
#include <functional>
#include <tuple>
#include <sstream> 
#include <iomanip>
```

Now, we will filter out all the non-compilable files. We wrote a bash script `filter_compilable_files.sh`. For each file, this will move the compilable cpp files into a folder `compiled_success_executables`.

#### Scraping OOPS C++ files from Github

We defined a list of keywords for identifying files with OOPS principles. 

```
    "class",
    "struct",
    "template",
    "namespace",
    "#include",
    "main()",
    "int main()",
    "void main()",
    "if",
    "for",
    "while",
    "switch",
    "try",
    "catch",
    "throw",
    "new",
    "delete",
    "const",
    "static",
    "virtual",
    "override",
    "public",
    "private",
    "protected",
    "return",
    "typedef",
    "enum",
    "union",
    "auto",
    "nullptr",
    "std::vector",
    "std::string",
    "std::map",
    "std::set",
    "std::shared_ptr",
    "std::unique_ptr",
    "std::thread",
    "std::mutex",
    "std::cout",
    "std::cin"
``` 
Now running the extract.py will extract all the files that are compilable. Please note that the search query will look something like - `extension:cpp std::mutex`. However, there is a API limit in Github in place which returns only 1000 results for each query. Hence, we could only scrape little data. Resulting in a total of 1618 cpp files in this method.

#### Scraping repositories for C++ files from Github

We scraped the [following repositories](https://docs.google.com/spreadsheets/d/1GKgE6r3UVirJbOs1KG2cyKQyNwJphTpj2JwjQkrZtrs/edit?gid=0#gid=0) to fetch data from Github. Clone the repository in the workspace using git clone. Then copy the `filter_compilable_files.sh` to the directory. Once we run this file, it will move all the compilable files from the repo to the folder - `compiled_success_executables`

### Step 2: Preprocessing and Data Collection

This step requires installing clang and libclang

#### Installation for clang and libclang

Make sure that all the requirements in `requirements.txt` are installed

`sudo apt-get install clang`

`sudo apt-get install libclang-dev`

`sudo apt install libclang-dev python3-clang`

You will also need to change the path in `set_library_file()` in `tokenization.py` to point to `libclang.co`. The below command will return the path to `libclang.co`

``find /usr/lib -name "libclang.so*" 2>/dev/null`

For example, `/usr/lib/llvm-18/lib/libclang.so`

#### Data Collection

Now, we have data folders in each dataset directory in the same format. This step involves combining all the individual files from `/data` directory of each dataset to `final_data`. However, we need an additional check which tests the files and appends them to final_data only if they are executable. For this we need to run `process_and_combine.py` script. PLease note that this step is a time consuming step as it involves executing each and every file.

## Step 3: Tokenization 

The code snippet can be converted into a series of tokens using lexer and into an AST using parser. Hence, we use `libclang` to convert our code into tokens and AST. Once we have our `/clean_data` folder, we can run `tokenization.py`.

## Step 4: Training

Code for model architecture, dataloaders, metrics, trainer in `model.ipynb`.

The training loop can be run by running `model.ipynb`. This requires `unpaired_c.txt` and `unpaired_cpp.txt` files. 

Link to checkpoints: https://iiithydresearch-my.sharepoint.com/:u:/g/personal/prit_kanadiya_research_iiit_ac_in/EdajC3Pzs09JsBsXNoh3-s0B6xgjuFSZRO_yuZR5PyTVyg?e=ScDc34 

Link to unpaired_c.txt: https://iiithydstudents-my.sharepoint.com/:t:/g/personal/kanakalatha_vemuru_students_iiit_ac_in/EWKOZlJCNOpBvsEX6ETG5fUB0ipKx7ocr7sb_wQYmIFmdA?e=KXTbmr

Link to unpaired_cpp.txt: https://iiithydstudents-my.sharepoint.com/:t:/g/personal/kanakalatha_vemuru_students_iiit_ac_in/Ec5mOFw_kfROoBL_6EKezn0BDQS-QDUI0i2gplq_WzV3LA?e=FgCrTZ