---
title: Dream Canvas
emoji: 🐕
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.24.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Reprodução fiel do DeepDream original do Google, de 2015
---

# Dream Canvas

O estado do projeto, decisões e próximos passos estão em [HANDOFF.md](HANDOFF.md).

O design system está em [DESIGN.md](DESIGN.md) e a implementação em Gradio, em
[theme.py](theme.py). O DESIGN.md é a fonte: mexa nele antes de mexer nos tokens.


Reprodução do DeepDream original do Google (Mordvintsev, 2015) em PyTorch, com
os pesos `bvlc_googlenet` do Caffe — não o InceptionV3 do tutorial em TensorFlow.
É esse modelo que produz as "puppy slugs" clássicas.

## Rodar local

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py
```

Abre em <http://localhost:7860>. Em Apple Silicon usa a GPU via MPS automaticamente.

## App de desktop

A mesma interface numa janela nativa, sem navegador:

```bash
.venv/bin/pip install -r requirements-desktop.txt
.venv/bin/python local.py
```

No macOS dá para deixar o `Dream Canvas.command` na mesa e abrir com dois cliques.

Para gerar um link público temporário (72 h), tunelando para a sua máquina:

```bash
DEEPDREAM_SHARE=1 .venv/bin/python app.py
```

## Linha de comando

```bash
.venv/bin/python deepdream.py foto.jpg -l inception_4c/output -o saida.png
```

Modelos: `bvlc` (padrão), `places365`, `places205`, `torchvision`.
Camadas: `inception_3a` a `inception_5b`.

## Duas linhagens do original

O preset **deepdream.c (2021)** reproduz os padrões da reescrita do próprio
Mordvintsev ([znah/deepdream_c](https://github.com/znah/deepdream_c)): 7 octaves,
20 passos, jitter determinístico e pirâmide que **amplia e continua** em vez de
reinjetar detalhe. O resultado é bem mais agressivo que o **Clássico 2015**,
que segue o notebook em Caffe.

Os dois controles estão em Ajuste fino:

- **Padrão do tremor** — `Aleatório` (Caffe) ou `Sequência fixa`, os
  deslocamentos por múltiplos de 79 e 127 do `deepdream.c`.
- **Pirâmide** — `Reinjetar detalhe` (Caffe) ou `Ampliar e continuar`
  (`deepdream.c`).

Combinar `Sequência fixa` + `Ampliar e continuar` remove toda aleatoriedade: em
CPU o resultado é bit-exato **sem precisar de seed**.

## Guiar por outra imagem

O objetivo `objective_guide` do notebook original: em vez de amplificar as
próprias características, a imagem é empurrada na direção das características
de uma segunda imagem. Para cada região da sua imagem, o algoritmo acha a
região da guia que melhor combina e puxa naquela direção — a guia empresta o
vocabulário visual.

```bash
.venv/bin/python deepdream.py foto.jpg -g flores.jpg -o saida.png
```

Funciona nas três abas — Imagem, Vídeo e Zoom — no accordion **Guiar por outra
imagem**, e nas CLIs de `deepdream.py` e `video.py` pela flag `-g`. Com guia, o
parâmetro `--objective` é ignorado.

Em vídeo e zoom as características da guia são calculadas **uma vez** e valem
para a sequência inteira, então o custo por quadro não muda.

## Sonhar em direção a um texto

Usa CLIP: em vez de amplificar camadas, a imagem é otimizada até o CLIP
considerá-la parecida com uma descrição escrita.

```bash
.venv/bin/pip install -r requirements-clip.txt
.venv/bin/python deepdream.py foto.jpg -t "um campo de girassóis" -n 20 --step-size 3
```

O que faz funcionar são os **recortes aleatórios**: otimizar a imagem inteira
contra o texto produz ruído adversarial, que o CLIP reconhece e nós não.
Pontuando dezenas de recortes de tamanhos variados por passo, a descrição
precisa valer em muitas escalas, e o resultado vira forma.

**Roda em CPU de propósito.** Medido num M4: o backward do ViT do CLIP leva
39,5 s em MPS contra 0,38 s em CPU — cem vezes mais lento, de forma consistente.
É limitação do backend MPS. Uma imagem de 384px leva cerca de 25 s.

Com texto, o modelo, as camadas e a imagem-guia são ignorados.

## Vídeo

Aplica o efeito a um vídeo inteiro, com coerência temporal:

```bash
.venv/bin/pip install -r requirements-video.txt
.venv/bin/python video.py entrada.mp4 -o saida.mp4
```

Exige o `ffmpeg` no sistema (`brew install ffmpeg`). O áudio do original é
preservado. A interface (`app.py` e `local.py`) ganha uma aba **Vídeo** quando
o opencv e o ffmpeg estão disponíveis — ela não aparece no Hugging Face Spaces,
porque um vídeo estoura o limite de duração da GPU do ZeroGPU.

Sonhar cada quadro isoladamente pisca muito: o processo é caótico e dois quadros
quase idênticos divergem por completo. Aqui o quadro sonhado anterior é deformado
pelo fluxo óptico até a posição do quadro atual e misturado a ele antes de sonhar,
o que faz os padrões grudarem nos objetos e acompanharem o movimento. A mesma
seed em todos os quadros mantém o padrão de jitter idêntico, o que remove parte
do flicker residual.

Medido num teste com zoom, o flicker com movimento compensado (menor é melhor):

| | Flicker |
| --- | --- |
| Vídeo de entrada (piso do ruído) | 1,86 |
| Quadro a quadro, sem realimentação | 6,69 |
| Realimentado, sem fluxo óptico | 8,32 |
| Realimentado + fluxo óptico | **4,58** |

Note que realimentar **sem** fluxo é pior que não realimentar: o padrão fica
parado enquanto a cena se move. É por isso que o fluxo óptico é o padrão.

Opções úteis: `--blend` (quanto do quadro anterior é realimentado; mais alto dá
mais estabilidade e mais rastro), `--start` e `--duration` para recortar um
trecho, `--max-dim` para a resolução, e `--no-flow` para ganhar velocidade
abrindo mão da estabilidade.

## Zoom infinito

Gera um vídeo a partir de **uma** imagem: sonha um quadro, aproxima um pouco em
direção a um ponto escolhido, e repete. Como cada quadro parte do anterior já
aproximado, detalhe novo nasce no centro sem parar.

Na interface (aba **Zoom infinito**) você clica na imagem para mirar o ponto e
escolhe a duração. Por código:

```python
from PIL import Image
import video

video.zoom_video(Image.open("foto.jpg"), "zoom.mp4",
                 center=(0.28, 0.22), duration=5, zoom=0.025)
```

`center` é normalizado (0 a 1) e `zoom` é o quanto a imagem avança por quadro.
Conte ~0,4 s por quadro a 512px: 5 s a 20 fps são 100 quadros, cerca de 40 s.

## Pulsar com uma música

No zoom infinito, uma faixa de áudio pode dirigir o efeito: a energia de cada
quadro acelera o avanço do zoom e reforça a força do passo, e o áudio entra no
vídeo final.

```bash
.venv/bin/python video.py --help   # --audio e --reactivity valem no modo zoom
```

Na interface, no accordion **Pulsar com uma música** da aba Zoom. Sem duração
explícita, a duração passa a ser a da faixa.

A envoltória é o RMS por quadro, normalizado pelo percentil 95 em vez do máximo
— assim um único pico não achata o resto da música. Medido num teste com batida
a 2 Hz: a variação do movimento entre quadros sobe de 0,76 para 2,63 com
reatividade 1,5.

## Montar um conjunto para treinar

`dataset.py` baixa imagens do Wikimedia Commons já organizadas em pastas que o
`ImageFolder` do torchvision lê como classes.

```bash
.venv/bin/python dataset.py --category "Botanical illustrations by family" \
  --per-class 300 --min-per-class 40 --strict-licenses -o dataset/botanical
```

Por que a Commons e não uma raspagem de busca de imagens: a licença de cada
arquivo vem declarada, a árvore de categorias já dá os **rótulos**, e a API
devolve a imagem redimensionada no servidor — não é preciso baixar originais de
20 MB para treinar a 224px.

`Botanical illustrations by family` tem **396 famílias**. A maioria é pequena
demais e cai no `--min-per-class`; sobram as que valem. Uma execução completa
leva dezenas de minutos, então vale rodar em segundo plano. Use
`--max-classes` para testar antes.

`--strict-licenses` restringe a domínio público e CC0, deixando de fora CC BY-SA
— copyleft cujo compartilhamento nas mesmas condições pode alcançar o que se
deriva dele, inclusive um modelo treinado.

O `manifest.csv` guarda origem, licença e autoria de cada arquivo.

## Treinar no seu próprio conjunto

Treinar o classificador num domínio novo reescreve as características das
camadas intermediárias — e são elas que o DeepDream amplifica. Treinado em
ilustração botânica, o modelo passa a fazer brotar pétalas em vez de focinhos.

```bash
.venv/bin/python train.py dataset/botanical -o modelos/botanico.pt
.venv/bin/python deepdream.py foto.jpg -m modelos/botanico.pt
```

O checkpoint é aceito onde quer que um modelo seja pedido, inclusive em vídeo e
zoom. Por padrão o treino congela tudo até `inception_3b`: as primeiras camadas
aprendem bordas e cor, que não mudam entre domínios, e o DeepDream trabalha de
`inception_4c` para frente.

O pré-processamento do treino é **o mesmo** do `dream()` — Caffe, BGR, médias
104/117/123. Treinar com a normalização da ImageNet e sonhar com a do Caffe
produziria características inconsistentes com o que o motor espera.

Medido num M4: 0,16 s por lote de 16 a 224px em MPS, contra 0,75 s em CPU.
Diferente do CLIP, a convnet não sofre no backward do MPS.

## Plugin do Figma

Exporta a camada selecionada, aplica o efeito e insere o resultado de volta no
documento. Veja [figma-plugin/README.md](figma-plugin/README.md).

Ele conversa com a API HTTP, que também serve para qualquer outra integração:

```bash
.venv/bin/python server.py
```

## Docker

```bash
docker build -t deepdream .
docker run -p 7860:7860 deepdream
```

O Dockerfile serve para auto-hospedagem, e usa wheels de CPU do torch. No Hugging
Face Spaces o app usa o SDK `gradio` com hardware ZeroGPU — ambos gratuitos; o SDK
`docker` e o CPU Basic são pagos. Serverless não serve em nenhum caso: as
dependências passam de 1 GB e o Gradio precisa de um servidor de longa duração.

No ZeroGPU a GPU só existe dentro da função decorada com `@spaces.GPU`, por isso o
device é resolvido a cada execução e os pesos ficam em cache na CPU, movidos por
cópia para a GPU a cada chamada.

## Variáveis de ambiente

| Variável | Padrão | Para quê |
| --- | --- | --- |
| `DEEPDREAM_DEVICE` | melhor disponível | `mps`, `cuda` ou `cpu` |
| `DEEPDREAM_MAX_DIM` | 1536 (768 em CPU) | Teto de resolução |
| `DEEPDREAM_SHARE` | — | `1` gera link público temporário |
| `PORT` | 7860 | Porta do servidor |
| `DEEPDREAM_OUTPUT_DIR` | `$TMPDIR/deepdream` | Onde as saídas são gravadas |
| `DEEPDREAM_OUTPUT_MAX_AGE_H` | 24 | Horas até uma saída ser apagada |

A interface tem uma seção **Manutenção** com o uso atual e um botão para limpar.

### Onde ficam os arquivos

As imagens e vídeos gerados pela interface vão para uma pasta única
(`$TMPDIR/deepdream` por padrão, mostrada no topo da própria interface) com nome
`AAAAMMDD-HHMMSS-xxxxxx`. Saídas com mais de 24 h são apagadas quando o app
sobe. **Baixe o que quiser guardar** — ou aponte `DEEPDREAM_OUTPUT_DIR` para
uma pasta permanente.

A CLI é diferente: ela escreve onde você mandar, e nada é apagado.

Além disso, o Gradio mantém cache próprio de uploads em `$TMPDIR/gradio`
(controlado por `GRADIO_TEMP_DIR`), e os pesos dos modelos ficam em
`~/.cache/torch`, que são permanentes de propósito — apagá-los só força novo
download.
