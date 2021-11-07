# kmeans

Projeto da disciplina de Engenharia de Programas. Especificação em /docs

## Codificação do KMeans

Está no arquivo `main.cpp`

Compile utilizando o seguinte comando: `g++ --std=c++17 -O1 -Wall main.cpp -lm`
Rode: `./a.out <image> <k> <repeat>`

## Análise quantitativa do KMeans

Distribuído no arquivo `main.cpp` através de comentários na função `kmeans`

Resultado:

```
(10, 0, 2) + K * (5, 1, 1) + N * (1, 0, 0) + X * (
   (4, 2, 4) + N * (6, 1 ,3) + K * (12, 4, 3) + (N * K) * (25, 15, 4)
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


## Planejamento do domínio de testes
 Do total de 46 imagens, foram selecionadas, inicialmente, 20 imagens com o Ns mais disperso para definir os seus respectivos Ks. 

O Output, em ordem decrescente da diferença percentual entre as imagens, para encontrar as que possuiam o Ns com maior dispersão: `measurement/ranking-diff-images.json`.

As 20 imagens escolhidas e seus respectivos K em : `experimental`

## Medição do tempo
Tempo de execução da inicialização e por iteração. `outputs/outputs.7z`

