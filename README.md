# Solving Captcha With Noisy Upper Case Characters
---

CAPTCHA é um acrônimo de "Completely Automated Public Turing test to tell Computers and Humans Apart" e como o nome diz,
é utilizado para diferenciar humanos e máquinas. Porém, com o avanço da computação, principalmente no campo da visão
computacional, essa tarefa fica cada vez mais difícil. Neste repositório é apresentada uma solução para resolver captchas
de 7 letras maiúsculas com ruídos utilizando técnicas de visão computacional clássica para tratamento e segmentação das imagens
e uma Rede Neural Convolucional para classificar os caracteres individualmente.

Abaixo é apresentada uma breve descrição sobre o conteúdo de cada diretório deste repositório:

- **captcha_labeling**: este diretório contém um notebook jupyter para facilitar a rotulação de novos captchas.
- **dataset**: neste diretório estão as imagens dos caracteres pré-processados, prontos para serem utilizados no treinamento do modelo.
- **model_weights**: os pesos da rede foram salvos no arquito *best_model.pth* (o estado do otimizador também foi salvo para possibilitar
que o treinamento continue a partir deste ponto).
- **src**: este diretório contém os módulos que possibilitam a criação e treinamento do modelo, bem como funções úteis para manipular o *dataset*.
- **test_captchas**: neste diretório estão as imagens dos captchas rotulados sem pré-processamento que foram utilizados para validar a solução.

O arquivo **Documentação.ipynb** apresenta a documentação detalhada da implementação e um exemplo de como utilizar os módulos implementados
para reproduzir os resultados.

## Resumo dos Resultados:

- Taxa de acerto dos caracteres individualmente (média): 94%.
- Taxa de resolução dos captchas: 72%.
- Taxa de erro de classificação dos captchas de 28%, sendo que:
  - 62% de erro por 1 caractere.
  - 24% de erro por 2 caracteres.
  - 10% de erro por 3 caracteres.
  
## Pontos de melhoria em trabalhos futuros:
- Estimar o tamanho de cada letra para minimizar os erros de segmentação quando há intersecção entre as letras.
- Implementar um emsemble com modelos distintos para tentar diminuir os erros de classificação entre D, O e Q e H, K e X.
- Avaliar outras técnicas de visão computacional para aplicar ao pré-processamento.
- Expandir para solucionar letras minúsculas e números.
