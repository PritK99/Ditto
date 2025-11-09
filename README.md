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

6) Because we know we are dealing with programming languages, we know our vocabulory size can not be big. Similarly, our token size can not be big. This is because there can only be a few keywords that can be used in a language, and the keywords are generally short enough.

### Results

For `simple.c` file, 

```
 ---- Transformed Code ----
['int', 'var0', '=', 'lit0', ';', 'struct', 'struct0', '{', 'char', '*', 'var1', ';', 'char', '*', 'var2', ';', 'int', 'var3', ';', 'int', 'var4', ';', '}', ';', 'int', 'func0', '(', 'int', 'var0', ')', '{', 'printf', '(', 'lit1', ',', 'lit2', ')', ';', 'return', '(', '(', 'var0', ')', '*', '(', 'var0', ')', ')', ';', '}', 'void', 'func1', '(', 'struct', 'struct0', 'var5', ')', '{', 'printf', '(', 'lit3', ',', 'var5', '.', 'var1', ',', 'var5', '.', 'var2', ',', 'var5', '.', 'var3', ',', 'var5', '.', 'var4', ')', ';', '}', 'int', 'main', '(', ')', '{', 'int', 'var6', '=', 'func0', '(', 'lit2', ')', ';', 'printf', '(', 'lit4', ',', '(', 'var6', ')', ')', ';', 'struct', 'struct0', 'var7', ';', 'var7', '.', 'var1', '=', 'lit5', ';', 'var7', '.', 'var2', '=', 'lit6', ';', 'var7', '.', 'var3', '=', 'lit7', ';', 'var7', '.', 'var4', '=', 'lit8', ';', 'func1', '(', 'var7', ')', ';', 'return', 'lit9', ';', '}']

---- Variable Dict ----
{'x': 'var0', 'number_plate': 'var1', 'model_name': 'var2', 'price': 'var3', 'year_of_manufacture': 'var4', 'c': 'var5', 'r': 'var6', 'car1': 'var7'}

---- Function Dict ----
{'square': 'func0', 'print_car': 'func1'}

---- Literal Dict ----
{'100': 'lit0', '"Squaring the number: %d\\n"': 'lit1', '5': 'lit2', '"Car: %s, Model: %s, Price: %d, Year: %d\\n"': 'lit3', '"%d\\n"': 'lit4', '"ABC123"': 'lit5', '"Tesla Model S"': 'lit6', '80000': 'lit7', '2022': 'lit8', '0': 'lit9'}

---- Struct Dict ----
{'Car': 'struct0'}

---- Class Dict ----
{}
Final output written to cleaned_simple.c
```

For `simple.cpp` file,

```
---- Transformed Code ----
['class', 'class0', '{', 'public', ':', 'std', '::', 'string', 'var0', ';', 'std', '::', 'string', 'var1', ';', 'int', 'var2', ';', 'int', 'var3', ';', 'class0', '(', 'const', 'std', '::', 'string', '&', 'var4', ',', 'const', 'std', '::', 'string', '&', 'var5', ',', 'int', 'var6', ',', 'int', 'var7', ')', ':', 'var0', '(', 'var4', ')', ',', 'var1', '(', 'var5', ')', ',', 'var2', '(', 'var6', ')', ',', 'var3', '(', 'var7', ')', '{', '}', 'virtual', 'void', 'func0', '(', ')', 'const', '{', 'std', '::', 'cout', '<<', 'lit0', '<<', 'var0', '<<', 'lit1', '<<', 'var1', '<<', 'lit2', '<<', 'var2', '<<', 'lit3', '<<', 'var3', '<<', 'lit4', ';', '}', 'virtual', '~', 'class0', '(', ')', '=', 'default', ';', '}', ';', 'class', 'class1', ':', 'public', 'class0', '{', 'public', ':', 'int', 'var8', ';', 'class1', '(', 'const', 'std', '::', 'string', '&', 'var4', ',', 'const', 'std', '::', 'string', '&', 'var5', ',', 'int', 'var6', ',', 'int', 'var7', ',', 'int', 'var9', ')', ':', 'class0', '(', 'var4', ',', 'var5', ',', 'var6', ',', 'var7', ')', ',', 'var8', '(', 'var9', ')', '{', '}', 'void', 'func0', '(', ')', 'const', 'override', '{', 'class0', '::', 'var10', '(', ')', ';', 'std', '::', 'cout', '<<', 'lit5', '<<', 'var8', '<<', 'lit4', ';', '}', '}', ';', 'int', 'func1', '(', 'int', 'var11', ')', '{', 'std', '::', 'cout', '<<', 'lit6', '<<', 'var11', '<<', 'lit4', ';', 'return', 'var11', '*', 'var11', ';', '}', 'int', 'main', '(', ')', '{', 'int', 'var12', '=', 'func1', '(', 'lit7', ')', ';', 'std', '::', 'cout', '<<', 'lit8', '<<', 'var12', '<<', 'lit4', ';', 'std', '::', 'vector', '<', 'class0', '*', '>', 'var13', ';', 'var13', '.', 'func2', '(', 'new', 'class1', '(', 'lit9', ',', 'lit10', ',', 'lit11', ',', 'lit12', ',', 'lit13', ')', ')', ';', 'var13', '.', 'func2', '(', 'new', 'class0', '(', 'lit14', ',', 'lit15', ',', 'lit16', ',', 'lit17', ')', ')', ';', 'for', '(', 'const', 'auto', '&', 'var14', ':', 'var13', ')', '{', 'var14', '->', 'var10', '(', ')', ';', 'std', '::', 'cout', '<<', 'lit18', ';', '}', 'for', '(', 'auto', 'var14', ':', 'var13', ')', 'delete', 'var14', ';', 'return', 'lit19', ';', '}']

---- Variable Dict ----
{'number_plate': 'var0', 'model_name': 'var1', 'price': 'var2', 'year_of_manufacture': 'var3', 'num': 'var4', 'model': 'var5', 'p': 'var6', 'year': 'var7', 'num_doors': 'var8', 'doors': 'var9', 'printInfo': 'var10', 'x': 'var11', 'r': 'var12', 'vehicles': 'var13', 'v': 'var14'}

---- Function Dict ----
{'printInfo': 'func0', 'square': 'func1', 'push_back': 'func2'}

---- Literal Dict ----
{'"Vehicle: "': 'lit0', '", Model: "': 'lit1', '", Price: $"': 'lit2', '", Year: "': 'lit3', '"\\n"': 'lit4', '"Number of doors: "': 'lit5', '"Squaring the number: "': 'lit6', '5': 'lit7', '"Result: "': 'lit8', '"ABC123"': 'lit9', '"Tesla Model S"': 'lit10', '80000': 'lit11', '2022': 'lit12', '4': 'lit13', '"XYZ789"': 'lit14', '"Generic Vehicle"': 'lit15', '30000': 'lit16', '2015': 'lit17', '"--------------------\\n"': 'lit18', '0': 'lit19'}

---- Struct Dict ----
{}

---- Class Dict ----
{'Vehicle': 'class0', 'Car': 'class1'}
Final output written to cleaned_simple.cpp
```