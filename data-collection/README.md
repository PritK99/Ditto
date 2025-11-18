## Data Collection

## Constraints

We added the following constraints to our dataset collection phase:

1) Compilability: We added a constraint that all the files used for training the transpiler should be compilable. That is, there should be no external dependencies in the file except for the standard libraries. We hope that this will ensure the model learns the correct syntactical nature of the code. This should also ensure that each file doesn’t have any external dependency. 

2) Object-oriented nature of C++: We added a constraint to collect C++ data that actively implements Object-Oriented Programming (OOP) principles, in the hopes that the transpiler will learn to do procedural-to-class translation and vice versa. 

3) Rich data focus: The transpiler should be capable of performing complex tasks such as function calls and handling different datatypes such as structs. For this, we collected data that incorporates complex functionality to ensure the transpiler can handle multiple interdepen- dent functions. 

4) Single file focus: We wanted the transpiler to emit compilable code. Hence, we added a constraint that the dataset should focus on entire files rather than individual functions. The compilability constraint will ensure that the dataset is indeed compilable

## Methodology

### Transcodeocean

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

### Kaggle

Because the data obtained from transcodeocean is very less, we use other resources. Here, we use https://www.kaggle.com/datasets/dianavostrova/formai-v2-dataset-without-clones-7z dataset to obtain C code. 

Extract all the files inside `kaggle/raw_data` directory. There are a total of approximately 3 L code files here. Since our requirement in not that much, we will randomly sample the fles from this folder.

Here, we require `/data` directory with single `unpaired_c.txt` file.

Now, running extract.py will randomly sample N codefiles from the data and add them to data directory after processing. This dataset has no duplicates.

### Hugging face directories

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

### Scraping OOPS C++ files from Github

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

### Scraping repositories for C++ files from Github

We scraped the [following repositories](https://docs.google.com/spreadsheets/d/1GKgE6r3UVirJbOs1KG2cyKQyNwJphTpj2JwjQkrZtrs/edit?gid=0#gid=0) to fetch data from Github. Clone the repository in the workspace using git clone. Then copy the `filter_compilable_files.sh` to the directory. Once we run this file, it will move all the compilable files from the repo to the folder - `compiled_success_executables`