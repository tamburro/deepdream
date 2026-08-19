---
version: alpha
name: Dream-Canvas
description: "Um palco quase preto para imagens extremamente saturadas. O sistema resolve um único conflito: o artwork gerado é caótico, ornamentado e cheio de cor, então a interface tem de ser acromática e recuar por completo. Herda de Runway o palco cinematográfico escuro e as faixas branco-papel para blocos de leitura, e de Linear a escada de superfícies com fios de 1px, os controles densos e precisos, e um único acento lavanda (#5E6AD2) usado com parcimônia em ação primária e foco. Nenhuma segunda cor cromática, nenhum gradiente atmosférico, nenhuma sombra colorida — a cor entra na tela pela imagem, nunca pelo chrome."

colors:
  primary: "#5E6AD2"
  on-primary: "#FFFFFF"
  primary-hover: "#828FFF"
  primary-pressed: "#4C55B8"
  canvas: "#0A0A0B"
  stage: "#000000"
  surface-1: "#141416"
  surface-2: "#1A1A1D"
  surface-3: "#212125"
  hairline: "rgba(255,255,255,0.08)"
  hairline-strong: "rgba(255,255,255,0.14)"
  ink: "#F4F4F5"
  ink-muted: "#A1A1AA"
  ink-subtle: "#71717A"
  ink-disabled: "#52525B"
  reading-paper: "#FAFAF8"
  reading-ink: "#18181B"
  semantic-danger: "#DC2626"
  semantic-warning: "#D97706"
  semantic-overlay: "rgba(0,0,0,0.72)"

typography:
  display:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: 600
    lineHeight: 1.15
    letterSpacing: -0.8px
  title:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: 600
    lineHeight: 1.25
    letterSpacing: -0.3px
  body:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: 400
    lineHeight: 1.55
    letterSpacing: 0
  label:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: 500
    lineHeight: 1.3
    letterSpacing: 0
  caption:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: 400
    lineHeight: 1.45
    color: "{colors.ink-subtle}"
  numeric:
    fontFamily: ui-monospace
    fontSize: 12px
    fontWeight: 500
    fontVariantNumeric: tabular-nums

spacing:
  scale: [4, 8, 12, 16, 24, 32, 48, 64]
  rail-width: 320px
  gutter: 24px

rounded:
  xs: 4px
  sm: 6px
  md: 8px
  lg: 12px
  pill: 999px

motion:
  fast: 120ms
  base: 180ms
  easing: cubic-bezier(0.2, 0, 0, 1)
---

## Overview

Dream Canvas gera imagens que já são barulhentas: espirais, olhos, focinhos, cor saturada em cada pixel. Qualquer cor na interface entra em competição direta com o resultado e perde. Por isso o sistema é **acromático por regra**, com exatamente um acento.

A tela se divide em duas zonas de função oposta. O **palco** (`{colors.stage}`, preto puro) é onde a imagem vive — sem moldura, sem sombra, sem cantos arredondados grandes que criem uma borda concorrente. O **trilho de controles** (`{colors.surface-1}`, 320px à esquerda) é uma coluna densa e de baixo contraste, feita para ser varrida com os olhos e esquecida.

O acento lavanda (`{colors.primary}`) aparece em três lugares e em nenhum outro: o botão de ação primária, o anel de foco, e o preenchimento do slider à esquerda do polegar. Não decora, não separa seções, não pinta ícones.

Blocos de texto longo — guias, explicações de camada — saem do palco escuro e entram numa **faixa branco-papel** (`{colors.reading-paper}` sobre `{colors.reading-ink}`), herdada de Runway. Texto corrido em cinza sobre preto cansa; a inversão sinaliza "isto é para ler, não para operar".

**Características-chave:**
- **Acromático por regra.** Um único acento (`{colors.primary}`). A cor da tela vem da imagem gerada.
- **Palco preto puro** (`{colors.stage}`) para o artwork; canvas quase preto (`{colors.canvas}`) para o resto.
- **Escada de três superfícies** com fios de 1px em vez de sombra para criar hierarquia.
- **Trilho de controles fixo de 320px** — densidade alta, contraste baixo, tipografia de 13px.
- **Faixas branco-papel** para qualquer bloco com mais de duas frases.
- **Números sempre em tabular-nums.** Valores de slider não podem dançar enquanto arrastam.
- Sem gradiente, sem glow, sem sombra colorida, sem ícone decorativo.

## Colors

### Brand & Accent
- **Lavanda** (`{colors.primary}`): botão primário, anel de foco, trilha preenchida do slider. Nada mais.
- **Lavanda Hover** (`{colors.primary-hover}`): estado hover do botão primário.
- **Lavanda Pressed** (`{colors.primary-pressed}`): estado pressionado.

### Surface
- **Stage** (`{colors.stage}`): preto puro, exclusivo da área da imagem e do vídeo.
- **Canvas** (`{colors.canvas}`): fundo geral da aplicação.
- **Surface 1** (`{colors.surface-1}`): trilho de controles, painéis.
- **Surface 2** (`{colors.surface-2}`): campos, chips não selecionados, accordions.
- **Surface 3** (`{colors.surface-3}`): hover de campo, chip selecionado.
- **Hairline** (`{colors.hairline}`): todas as divisórias e bordas de campo.
- **Hairline Strong** (`{colors.hairline-strong}`): borda de campo em hover.

### Text
- **Ink** (`{colors.ink}`): títulos e valores.
- **Ink Muted** (`{colors.ink-muted}`): corpo e rótulos de controle.
- **Ink Subtle** (`{colors.ink-subtle}`): microcopy de apoio, unidades, estimativas de tempo.
- **Ink Disabled** (`{colors.ink-disabled}`): controles inativos.

### Reading
- **Reading Paper** (`{colors.reading-paper}`) sobre **Reading Ink** (`{colors.reading-ink}`): faixas de leitura invertidas.

### Semantic
- **Danger** (`{colors.semantic-danger}`): apenas erros e a ação destrutiva de limpar cache.
- **Warning** (`{colors.semantic-warning}`): apenas avisos de custo alto (vídeos longos).

Não existe verde de sucesso. Sucesso é a imagem aparecer.

## Typography

Uma família: **Inter**, com `-apple-system, system-ui, Segoe UI, Roboto` de fallback. Nada de serifa, nada de display alternativo — o app não precisa de voz tipográfica própria, precisa sumir.

Números — valores de slider, contadores de quadro, dimensões, tempos — usam `{typography.numeric}` com `tabular-nums`. Sem isso, um contador de progresso muda de largura a cada quadro e a linha inteira treme.

Rótulos de camada (`inception_4c/output`) também vão em mono: são identificadores técnicos, não prosa.

## Layout

Duas colunas fixas. Trilho de 320px à esquerda, palco ocupando o resto. Abaixo de 900px o trilho vira uma folha inferior recolhível e o palco assume a largura toda.

Escala de espaçamento em passos de 4px. Dentro do trilho, o ritmo é 12px entre controles de um mesmo grupo e 24px entre grupos.

O palco nunca tem padding maior que 24px — a imagem é o conteúdo, não um cartão dentro de uma página.

## Elevation & Depth

Nenhuma sombra. Hierarquia é feita pela escada de superfícies e por fios de 1px. Um painel "acima" de outro é mais claro, não mais sombreado.

A única exceção é o scrim de modal (`{colors.semantic-overlay}`).

## Shapes

- Campos, chips e botões: `{rounded.md}`.
- Painéis e accordions: `{rounded.lg}`.
- Chips de preset: `{rounded.pill}`.
- **A imagem no palco: `{rounded.xs}` ou nada.** Cantos arredondados grandes numa imagem gerada cortam o artwork e criam uma borda que compete com o conteúdo.

## Components

**`stage`** — Área da imagem ou vídeo.
- Fundo `{colors.stage}`, sem borda, padding máximo 24px, `{rounded.xs}`.
- Estado vazio: ícone de 24px em `{colors.ink-disabled}` centralizado, mais uma linha em `{typography.caption}`. Nunca uma ilustração.

**`control-rail`** — Coluna de controles.
- Fundo `{colors.surface-1}`, largura `{spacing.rail-width}`, fio de 1px `{colors.hairline}` na borda direita.

**`slider`** — Controle numérico principal.
- Trilha 4px em `{colors.surface-3}`; parte preenchida em `{colors.primary}`.
- Rótulo à esquerda em `{typography.label}` `{colors.ink-muted}`; valor à direita em `{typography.numeric}` `{colors.ink}`.
- Microcopy de consequência abaixo, em `{typography.caption}`, só quando o parâmetro não é autoexplicativo.

**`preset-chip`** — Seleção de estilo.
- Padrão: `{colors.surface-2}`, texto `{colors.ink-muted}`, `{rounded.pill}`, padding 6px 14px.
- Selecionado: `{colors.surface-3}`, texto `{colors.ink}`, fio `{colors.hairline-strong}`. Seleção é elevação de superfície, não cor.

**`button-primary`** — Ação única de cada aba.
- Fundo `{colors.primary}`, texto `{colors.on-primary}`, `{rounded.md}`, padding 10px 16px, largura total do trilho.
- Existe **um** por aba. Se aparecer um segundo botão primário, a aba está fazendo duas coisas.

**`button-ghost`** — Ações secundárias (atualizar, limpar).
- Fundo transparente, fio `{colors.hairline}`, texto `{colors.ink-muted}`.

**`reading-band`** — Bloco de texto longo.
- Fundo `{colors.reading-paper}`, texto `{colors.reading-ink}`, `{rounded.lg}`, padding 24px.

**`progress`** — Andamento de geração.
- Barra 2px em `{colors.primary}` sobre `{colors.surface-3}`.
- Sempre acompanhada de contagem absoluta em `{typography.numeric}`: `Quadro 34/100`. Nunca só porcentagem, nunca só spinner.

## Voice & Copy

A pessoa que usa isto está fazendo arte, não configurando software. A copy segue daí.

**Rotule o resultado, não o parâmetro.** `Iterações por octave` é o nome do número no código; `Intensidade` é o que a pessoa quer mudar. Onde o termo técnico é o nome real e pesquisável — `inception_4c`, `octaves` — mantenha, e explique ao lado.

**Toda espera tem número.** Nunca "Processando…" sozinho. Sempre `Quadro 34/100` ou `cerca de 40 s`. O app tem operações de minutos; esconder isso é hostil.

**Microcopy declara a consequência, não a definição.** Não: "Controla a taxa de ampliação por quadro." Sim: "Mais alto: mergulho mais rápido, criaturas com menos tempo de se formar."

**Erros dizem o que fazer.** Não: "Invalid input." Sim: "Escolha uma imagem." — sujeito implícito, verbo no imperativo, uma frase.

**Sem entusiasmo fabricado.** Nada de exclamação, "Pronto!", "Incrível!", emoji em rótulo. O resultado na tela é o entusiasmo.

**Português direto, segunda pessoa implícita.** "Clique na imagem para mirar", não "O usuário deve clicar".

**Números com unidade colada ao valor**, não ao rótulo: rótulo `Duração`, valor `5 s`.

## Do's and Don'ts

### Do
- Deixe o palco preto e vazio quando não há imagem.
- Use a escada de superfícies para hierarquia.
- Coloque o custo em tempo ao lado de todo controle que aumenta o custo.
- Mantenha um único botão primário por aba.
- Use tabular-nums em tudo que muda durante o processamento.

### Don't
- Não introduza uma segunda cor cromática — nem para "categorizar" abas ou modelos.
- Não coloque gradiente, glow ou sombra colorida em nada.
- Não arredonde a imagem gerada além de 4px.
- Não use verde de sucesso.
- Não escreva microcopy que repita o rótulo com outras palavras.
- Não anime nada acima de `{motion.base}`. A espera real já é longa; animação lenta a piora.

## Responsive Behavior

- **≥1200px**: trilho 320px + palco.
- **900–1199px**: trilho 280px, microcopy dos sliders recolhe para tooltip.
- **<900px**: trilho vira folha inferior; palco ocupa a largura toda; alvos de toque sobem para 44px.

## Iteration Guide

Ao acrescentar uma aba nova, a pergunta é sempre: qual é a **única** ação primária? Se houver duas, são duas abas.

Ao acrescentar um controle, ele entra recolhido em "Ajustes" por padrão. Só sobe para o trilho principal se mudar o resultado de forma que a pessoa perceba sem comparar lado a lado.

## Known Gaps

- Tema claro não está definido. O app é dark-only por decisão, não por omissão: imagem saturada sobre fundo claro perde contraste.
- Não há estados de vazio ilustrados nem onboarding.
- Acessibilidade: o contraste de `{colors.ink-subtle}` sobre `{colors.surface-1}` fica em 4.6:1, acima de AA para texto pequeno, mas apertado. Não escureça mais.
