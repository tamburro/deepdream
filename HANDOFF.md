# Dream Canvas — handoff

Estado do projeto em 22/08/2026. Complementa o [CLAUDE.md](CLAUDE.md), que tem as
regras de código, e o [DESIGN.md](DESIGN.md), que tem o design system.

## O que é

Reprodução do DeepDream original do Google (Mordvintsev, 2015) em PyTorch, com
os pesos `bvlc_googlenet` do Caffe. **Não** é o tutorial em TensorFlow — essa
distinção é o ponto central do projeto e determinou várias decisões.

## Onde está

| | Endereço |
| --- | --- |
| GitHub | https://github.com/tamburro/deepdream |
| Space | https://huggingface.co/spaces/pedrothumbs/2015dream |
| App direto | https://pedrothumbs-2015dream.hf.space |

O remote `space` já está configurado no git local. `git push space main` publica.

## Mapa dos arquivos

| Arquivo | Papel |
| --- | --- |
| `deepdream.py` | Motor. Não sabe que existe interface. CLI própria. |
| `video.py` | Vídeo com coerência temporal + zoom infinito. CLI própria. |
| `server.py` | API HTTP (`/health`, `/dream`) para integrações. |
| `app.py` | Interface Gradio, três abas. |
| `theme.py` | Tradução do DESIGN.md para tokens do Gradio. |
| `local.py` | Mesma interface numa janela nativa (pywebview). |
| `clipguide.py` | Objetivo por texto via CLIP. Roda em CPU de propósito. |
| `dataset.py` | Baixa conjuntos do Wikimedia Commons para treinar. |
| `train.py` | Fine-tuning do GoogLeNet. Gera checkpoint que o motor aceita. |
| `figma-plugin/` | Plugin que fala com o `server.py`. |

Dependências separadas de propósito: `requirements.txt` (núcleo + web),
`requirements-video.txt` (opencv), `requirements-desktop.txt` (pywebview).
O Space não instala os dois últimos, e é assim que as abas de Vídeo e Zoom
somem sozinhas lá.

## Como rodar

```bash
.venv/bin/python app.py      # web em 0.0.0.0:7860
.venv/bin/python local.py    # janela nativa, só 127.0.0.1
.venv/bin/python server.py   # API em 8000, necessária para o plugin
.venv/bin/python deepdream.py foto.jpg -l inception_4c/output
.venv/bin/python video.py entrada.mp4 --duration 3
```

## Decisões que não são óbvias no código

**O modelo é o `bvlc`, não o torchvision.** O GoogLeNet do torchvision tem
batchnorm e é outro treinamento; produz um resultado bem mais fraco. O port
`pytorch-caffe-models` tem a arquitetura fiel (LRN, `ceil_mode`, sem batchnorm)
e os pesos originais. Testado lado a lado: a diferença é grande.

**Fidelidade custou detalhes específicos.** Objetivo L2 (não a média), passo
normalizado pela média absoluta do gradiente em unidades 0-255, jitter por
passo, `scipy.ndimage.zoom(order=1)` em vez de `F.interpolate`, e reinjeção de
detalhe entre octaves. Cada um desses mudou o resultado de forma visível.

**Cache de modelo por `(nome, device)`, sem cópia por chamada.** Houve uma
versão com `deepcopy` por receio do ZeroGPU derrubar o contexto CUDA. A
documentação deles diz o contrário: carregar em `cuda` na importação é o
recomendado e é mais eficiente. O log do Space confirma
(`ZeroGPU tensors packing: 100%`).

**Duração dinâmica no `@spaces.GPU`.** Declarar perto do real melhora a
prioridade do visitante na fila. Era 120 s fixos; agora 40 s no caso comum.

**Dropdowns usam tuplas `(rótulo, valor)`.** Antes havia `model.split(" — ")[0]`
em três callbacks — a copy estava acoplada à lógica e mexer no texto quebrava a
chamada.

## Medições

| | |
| --- | --- |
| Imagem 700px, MPS | 1,14 s |
| Imagem 1920×1080, MPS | 10,8 s |
| Imagem 700px, CPU do M4 | ~2,5 s |
| Vídeo, por quadro a 640px | ~0,7 s |
| Zoom infinito, por quadro a 512px | ~0,4 s |
| Build do Space | ~2 min |
| Treino, lote de 16 a 224px, MPS | 0,16 s |
| Treino, época de 23,5k imagens, MPS | ~178 s |
| Treino, lote de 16 a 224px, CPU | 0,75 s |
| CLIP, forward+backward 16 recortes, MPS | 39,5 s |
| CLIP, forward+backward 16 recortes, CPU | 0,38 s |

Flicker em vídeo, com movimento compensado (menor é melhor): entrada 1,86 ·
quadro a quadro 6,69 · realimentado sem fluxo 8,32 · **realimentado + fluxo
óptico 4,58**. Note que realimentar sem fluxo é *pior* que não realimentar.

## Reprodutibilidade

`--seed` só é bit-exato em `--device cpu`. No MPS os kernels não são
determinísticos e o DeepDream amplifica a diferença até ficar visível — medido:
diferença média de 8/255 entre duas execuções com a mesma seed.

## Custos

Hospedar no Space é **grátis** e continua grátis. A cota de GPU é debitada de
**quem acessa**: 2 min/dia anônimo, 5 min conta grátis, 40 min PRO. Contas
gratuitas podem manter 2 Spaces ZeroGPU.

Domínio próprio em Space é recurso **pago** (PRO ou Team/Enterprise). Renomear
o Space é grátis e muda a URL.

## Limitações conhecidas

- **Disco do Space é efêmero.** Galeria, histórico ou favoritos não funcionam lá
  sem armazenamento pago ou banco externo.
- **O plugin do Figma só funciona na máquina que roda o `server.py`**, porque
  aponta para `localhost:8000`.
- Vídeo e Zoom não existem no Space, por decisão — estourariam o teto de GPU.
- Tema claro não existe. É decisão, não omissão.

## Referência do autor

[znah/deepdream_c](https://github.com/znah/deepdream_c) é a reescrita do próprio
Mordvintsev, em C89 sem dependências (2021). Serve de oráculo de fidelidade.
Confere com o nosso: camada `inception_4c_output`, objetivo L2
(`grad[i] = val[i]`), passo `val += 1.5 * grad / mean(|grad|)` e octave 1.4
bilinear. As três divergências dele — 7 octaves e 20 passos, pirâmide que cresce
sem reinjetar detalhe, e jitter determinístico — estão implementadas como o
preset **deepdream.c (2021)** e como os parâmetros `jitter_mode` e
`pyramid_mode`.

Medido: o jitter determinístico **não** resolve a não-determinância em MPS
(diferença média de 5/255), porque ela vem dos kernels. Mas em CPU,
`sequence` + `grow` é bit-exato sem seed.

## O modelo botânico

`dataset/botanical` e `modelos/botanico.pt` estão fora do git (`.gitignore`).
Para refazer:

```bash
.venv/bin/python dataset.py --category "Botanical illustrations by family" \
  --per-class 250 --min-per-class 40 --width 512 --all-licenses -o dataset/botanical
.venv/bin/python train.py dataset/botanical -o modelos/botanico.pt -e 12
```

O download leva 2 a 3 horas com 2 workers — não aumente, ver abaixo.

Resultado: 49,8% de acerto em 181 classes, 12 épocas de ~3 min. Por camada,
`4c` dá rosetas de pétalas, `5a` formas bulbosas com nervura e `5b` espirais de
capítulo floral.

**Mais épocas não deram dreams melhores.** Com 20 épocas o acerto sobe para
52,0%, mas a perda de treino cai a 0,141 e o resultado visual se afasta do
botânico: em `5b` aparecem cabeças de pássaro no lugar das espirais florais.
Rede mais discriminativa amplifica objeto, não textura. O acerto do
classificador é indicador fraco do que interessa aqui — o padrão continua
`botanico.pt`, de 12 épocas.

`app.py` varre `modelos/*.pt` e monta uma opção de modelo **e um preset** para
cada. O preset é necessário: sem ele, escolher o modelo em Ajuste fino e clicar
num chip de estilo devolveria tudo para o `bvlc`.

## Armadilhas que custaram tempo

- `allowedDomains` do Figma **rejeita IP numérico**. Tem de ser `localhost`.
- O Chrome barra chamadas de página pública para rede local: exige
  `allow_private_network=True` no CORSMiddleware. Sem isso o preflight volta 400
  com "Disallowed CORS private-network", mesmo com CORS todo liberado.
- Com `-shortest`, o ffmpeg fecha a entrada quando a trilha mais curta acaba e
  os últimos `write` estouram `BrokenPipeError`. É fim normal: `_write_frame`
  trata e encerra o laço.
- A Wikimedia limita a taxa de download por concorrência, e o erro chega como
  HTTPError. Medido em 40 imagens, sem retentativa: 1 worker acerta 34, 2
  acertam 30, **4 acertam só 5**. Com 2 workers e respeitando o `Retry-After`,
  o acerto vai a 58/60. Não aumentar `--workers`.
- `short_description` do Space tem limite de **60 caracteres**; o push é
  rejeitado se passar.
- No Gradio 6, `theme` e `css` vão no `launch()`, não no construtor de `Blocks`.
- O primeiro push para o Space precisa de `--force`: ele nasce com um commit
  próprio e o histórico é independente.

## Em aberto

**Decidido não fazer por enquanto:** montar o `/dream` junto do Gradio no Space
para o plugin funcionar fora da máquina local. É viável (`gr.mount_gradio_app`
existe), custa zero, mas a alocação de GPU numa rota não-Gradio é incerta.

**Próximos passos sugeridos, em ordem:**
1. **DeepDream guiado** — feito.
2. **CLIP com texto** — sonhar em direção a uma descrição escrita.
3. **Preset deepdream.c** — feito.
4. **Áudio-reativo** no zoom — feito.
5. **Fine-tuning** — feito e rodado. `dataset/botanical` tem 27.717 imagens em
   181 famílias (8,7 GB), e `modelos/botanico.pt` chegou a 49,8% de acerto em
   12 épocas, ~3 min cada. O acerto ainda subia no fim: mais épocas devem
   render. O resultado visual é inequívoco — some o focinho de cachorro,
   aparecem rosetas e frondes.

**Antes de monetizar:** confirmar a licença dos pesos `places365` e `places205`
(CSAIL/MIT, alguns termos são só para pesquisa). O `bvlc` diz "released for
unrestricted use" e o torchvision é BSD.
