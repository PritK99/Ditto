# Ditto

<p align="center">
  <img src="https://media.tenor.com/4wt81D8xUEwAAAAM/ditto-pokemon.gif" alt="ditto">
  <br>
  <small><i>Image source: https://tenor.com/search/ditto-pokemon-gifs</i></small>
</p>

## Table of Contents

- [Ditto](#Ditto)
  - [Introduction](#introduction)
  - [Demo](#demo)
  - [Methodology](#methodology)
  - [File Structure](#file-structure)
  - [Getting started](#Getting-Started)
      - [Data](#data)
      - [Pretrained Models](#pretrained-models)
  - [References](#references)

## Introduction

Ditto is a transpiler that converts code between C++ and C languages. The name comes from the Pokémon Ditto, which can copy any other Pokémon exactly. Our goal is to build an AI transpiler that can translate code across different programming paradigms, no matter the language. For our project, we focus on C++ to C conversion and vice versa. Through this project, we aim to explore 2 hypotheses:

1) Can we build an AI system that is able to transpile code across programming paradigms? (For example, C → C++)

2) Can we integrate compiler domain knowledge to improve the performance of deep learning models?

## Demo

## Methodology

The methodology comprises of 3 major sections: Data Collection, Preprocessing, and Model Architecture. For detailed information on each stage, please refer to the `README.md` file in the respective directory.

## Getting Started

### Data

Link to the preprocessed data can be found <a href="">here</a>. 

**Note:** The dataset provided above has been obtained after several preprocessing steps. To access the raw data files, intermediate output files, or logs, click <a href="https://iiithydresearch-my.sharepoint.com/:f:/g/personal/prit_kanadiya_research_iiit_ac_in/IgDP3mtf5hr1RIuOD9ePrzljAZbwoiaikGDwoTUWWTzUiDE?e=EWenOJ">here</a>.

### Model

## References

```
@misc{lachaux2020unsupervisedtranslationprogramminglanguages,
      title={Unsupervised Translation of Programming Languages}, 
      author={Marie-Anne Lachaux and Baptiste Roziere and Lowik Chanussot and Guillaume Lample},
      year={2020},
      eprint={2006.03511},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2006.03511}, 
}

@misc{nachmani2024translatotron3speechspeech,
      title={Translatotron 3: Speech to Speech Translation with Monolingual Data}, 
      author={Eliya Nachmani and Alon Levkovitch and Yifan Ding and Chulayuth Asawaroengchai and Heiga Zen and Michelle Tadmor Ramanovich},
      year={2024},
      eprint={2305.17547},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2305.17547}, 
}
```