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
bilinear. Diverge em três pontos: ele usa 7 octaves e 20 passos (nós, 4 e 10),
**não** faz reinjeção de detalhe — a imagem cresce e continua — e o jitter dele
é determinístico (`sx = passo*79, sy = passo*127`) em vez de aleatório.

## Armadilhas que custaram tempo

- `allowedDomains` do Figma **rejeita IP numérico**. Tem de ser `localhost`.
- O Chrome barra chamadas de página pública para rede local: exige
  `allow_private_network=True` no CORSMiddleware. Sem isso o preflight volta 400
  com "Disallowed CORS private-network", mesmo com CORS todo liberado.
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
1. **DeepDream guiado** — empurrar a imagem na direção das características de
   uma segunda imagem-guia. Está no notebook original, é pouca mudança no
   `_objective`, e é o maior desbloqueio criativo por linha de código.
2. **CLIP com texto** — sonhar em direção a uma descrição escrita.
3. **Áudio-reativo** no vídeo, só local.
4. **Fine-tuning** do GoogLeNet num conjunto próprio, para dreams autorais.

**Antes de monetizar:** confirmar a licença dos pesos `places365` e `places205`
(CSAIL/MIT, alguns termos são só para pesquisa). O `bvlc` diz "released for
unrestricted use" e o torchvision é BSD.
