# Ditto

<p align="center">
  <img src="https://media.tenor.com/4wt81D8xUEwAAAAM/ditto-pokemon.gif" alt="ditto">
  <br>
  <small><i>Image source: https://tenor.com/search/ditto-pokemon-gifs</i></small>
</p>

## Table of Contents

- [Ditto](#Ditto)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Data](#data)
  - [Methodology](#methodology)
      - [Data Collection](#data-collection)
      - [Preprocessing](#preprocessing)
  - [File Structure](#file-structure)
  - [Getting started](#Getting-Started)
  - [Future Goals](#future-goals)
  - [References](#references)
  - [License](#license)

## Introduction

Ditto is a transpiler that converts code between C++ and C languages. The name comes from the Pokémon Ditto, which can copy any other Pokémon exactly. Our goal is to build an AI transpiler that can translate code across different programming paradigms, no matter the language. For our project, we focus on C++ to C conversion and vice versa.

## Data

For C and C++,

Link to Raw Data: https://iiithydresearch-my.sharepoint.com/:f:/g/personal/prit_kanadiya_research_iiit_ac_in/IgDFA0seHhbKRonY97Qo3D8gAYFGnBhadsJY7dEPP4YQMfQ?e=ZQW5Hf 

Link to Tokenized Data (Without LCA): https://iiithydresearch-my.sharepoint.com/:f:/g/personal/prit_kanadiya_research_iiit_ac_in/IgCc4OcS1-l7SqNS3Gaqndb8AVB94cdZTJI3jy2aUukjXL4?e=sceeRY

Link to Tokenized Data (With LCA): N/A

## Methodology

The methodology comprises of 3 major sections: Data Collection, Preprocessing, Model Architecture.

### Data Collection

### Preprocessing

This section deals with conversion of code in raw form to tokens. 

For eg,

```c
#include <stdio.h>

void greet()
{
  printf("%s", "Ditto!");
}

int main()
{
  int x = 5;
  int y = 10;
  int z = x + y;

  return 0;
}
```

This should become

`
[
"#include", "<stdio.h>", "void", "func0", "(", ")", "{", "printf", "(", "lit0", ",", "lit1", ")", ";", "}", "int", "main", "(", ")", "{", "int", "var0", "=", "lit2", ";", "int", "var1", "=", "lit3", ";", "int", "var2", "=", "var1", "+", "var0", ";", "return", "lit3", ";", "}"
]
`

In addition to these tokens, we should recieve the mappings for obfuscated variables, functions, classes, structs, literals, etc. 

For more details on preprocessing section along with examples, please refer `preprocessing/README.md`. 