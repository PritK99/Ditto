# Ditto

<p align="center">
  <img src="https://media.tenor.com/4wt81D8xUEwAAAAM/ditto-pokemon.gif" alt="ditto">
  <br>
  <small><i>Image source: https://tenor.com/search/ditto-pokemon-gifs</i></small>
</p>

## Introduction

Ditto is a transpiler that converts code between C++ and C languages. The name comes from the Pokémon Ditto, which can copy any other Pokémon exactly. Our goal is to build an AI transpiler that can translate code across different programming paradigms, no matter the language. For our project, we focus on C++ to C conversion and vice versa.

# Preprocessing

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

### Requirements

1) Code Obfuscation: All the user defined entities i.e. variables, user defined functions, and literals, should be obfuscated like `var0`, `func0`, and `lit0`.

2) However, the obfuscation process should not affect standard functions such as `printf`, `main` and other keywords. 

3) The tokenization process should take care of escape characters.

4) Special cases need to be made for dealing with entities like `#define`, `printf("%d", ...)`, etc. 

5) The tokenization process should return:

    * Tokens
    * Raw Positions
    * LCA Positions
    * Variable Map
    * Literal Map
    * Function Map

    Hence the expected output is a CSV with all these fields.

6) Because we know we are dealing with programming languages, we know our vocabulory size can not be big. SImilarly, our token size can not be big. This is because there can only be a few keywords that can be used in a language, and the keywords are generally short enough.

