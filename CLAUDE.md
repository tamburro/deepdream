# Dream Canvas

Reproduz o DeepDream original do Google (Mordvintsev, 2015), não o tutorial em
TensorFlow. O objetivo do projeto é fidelidade ao resultado de 2015.

## Ambiente
- Rodar sempre pelo venv: `.venv/bin/python` (Python 3.12 do Homebrew).
  O `python3` do sistema é 3.9 e não deve ser usado.
- Device padrão é MPS (Apple Silicon). `pick_device()` cai para CPU se não houver.

## Modelos
O backend padrão `bvlc` carrega os pesos `bvlc_googlenet` originais do Caffe via
`pytorch-caffe-models`, com a arquitetura fiel: LRN, `ceil_mode=True` nos poolings
e **sem batchnorm**. É ele que produz as "puppy slugs" clássicas — não trocar o
padrão. O backend `torchvision` é outro treinamento (com batchnorm) e existe como
opção estética, não como referência.

## Fidelidade ao original
Estes detalhes vêm da receita em Caffe e mudam bastante o resultado se alterados:
- Pré-processamento Caffe: BGR, médias [104, 117, 123], escala 0-255.
- Objetivo `l2` (soma dos quadrados das ativações), não a média.
- Passo normalizado pela média absoluta do gradiente, em unidades de pixel 0-255
  (por isso `step_size=1.5`, dividido por 255 internamente). A equivalência com o
  espaço [0, 1] usado internamente é exata, porque o pré-processamento é afim.
- Jitter (roll aleatório) antes de cada passo, desfeito depois.
- Redimensionamento com `scipy.ndimage.zoom(order=1)`, o mesmo do original —
  não trocar por `F.interpolate`, que interpola de forma ligeiramente diferente.
- Pirâmide de octaves com **reinjeção de detalhe**: o que a octave menor gerou é
  ampliado e somado à base da octave seguinte. Sem isso o resultado borra.
- A soma `octave_base + detail` não é clipada; o clip acontece dentro do passo,
  como no original.

## Reprodutibilidade
`--seed` só dá resultado bit-exato em `--device cpu`. No MPS os kernels não são
determinísticos e o DeepDream amplifica a diferença até ela ficar visível
(medido: diferença média de 8/255 entre duas execuções com a mesma seed).
Para comparar parâmetros de forma controlada, fixar seed **e** usar CPU.

## ZeroGPU
Rodando no Hugging Face Space, `spaces` é importável e `ZEROGPU` fica verdadeiro.
Duas regras vêm da documentação deles e não devem ser revertidas:
- O modelo é carregado em `cuda` **na importação** (`preload`), não dentro da
  função decorada. Fora do `@spaces.GPU` existe um modo de emulação CUDA que
  permite isso, e as transferências são otimizadas na inicialização.
- A duração declarada em `@spaces.GPU` é **dinâmica** (`estimate_duration`).
  Declarar perto do real melhora a prioridade do visitante na fila; pedir muito
  mais que o necessário custa prioridade à toa.

A cota de GPU é debitada de quem acessa, não do dono do Space — por isso vale
economizar segundos por chamada.

## Objetivo guiado
`_guided_objective` reproduz o `objective_guide` do Caffe: para cada posição,
escolhe o vetor da guia com maior produto escalar e usa ele como gradiente. O
truque é que o gradiente de `(x * melhor).sum()` em relação a x é exatamente
`melhor` — era o que o Caffe escrevia direto no `dst.diff`. O argmax roda sob
`no_grad` porque a escolha é constante, não diferenciável.

As ativações da guia são calculadas uma vez, antes do laço, e reusadas em todos
os passos e octaves. Em vídeo e zoom, `_prepare_guide` as calcula uma vez para
a sequência inteira e passa a lista pronta — por isso `dream(guide=...)` aceita
tanto uma PIL.Image quanto ativações já calculadas.

## Convenções
- Nomes de camada são canônicos no formato Caffe (`inception_4c/output`) e cada
  backend traduz para o nome real do seu módulo via a terceira entrada de `MODELS`.
- Imagens circulam como tensores `(3, H, W)` em `[0, 1]`; o pré-processamento de
  cada modelo acontece dentro do `FeatureExtractor.forward`.
- Imagens maiores que `tile_size` usam o gradiente por ladrilhos. Esse caminho
  não existe no original e afasta um pouco do resultado de referência: para
  fidelidade máxima, manter a imagem abaixo de `tile_size`.
