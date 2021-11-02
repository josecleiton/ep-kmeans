# kmeans

Projeto da disciplina de Engenharia de Programas. Especificação em /docs

## Codificação do KMeans

Está no arquivo `main.cpp`

Compile utilizando o seguinte comando: `g++ --std=c++17 -O1 -Wall main.cpp -lm`

## Análise quantitativa do KMeans

Distribuído no arquivo `main.cpp` através de comentários na função `kmeans`

Resultado:

```
(10, 0, 3) + N*(3, 1, 1) + K*(5, 1, 1) + X * (
    (4, 1, 4) + N * (3, 1, 2) + K * (9, 4, 3) + (N * K) * (21, 13, 4)
)
```

## Setup experimental

- CPU: Intel Xeon E-2276G (12) @ 4.9GHz
- Memória: 64GB
- Linguagem utilizada: C++17
- Compilador: g++ 10.3.0 com a flag -O1

Output com desvio padrão e média aritmética em: `/resources/setup_experimental.ods`

## Plano experimental

Todas as imagens devem rodar 100x e gerar um csv de resultados. As imagens se encontram em `/images`

| 	      imagem 		| 	    k    	|
| --------------------- |:-------------:|
|   casal_moreno01.jpg  |       5       |
| familia_branca01.jpg  |       6 	    |
| familia_morena01.jpg  |       6       |
|  mulher_morena03.jpg  |       3       |
|  senhor_branco01.jpg  |       3       |
|  senhor_branco02.jpg  |       3       |
|  senhor_branco03.jpg  |       3       |
|  senhor_branco04.jpg  |       3       |