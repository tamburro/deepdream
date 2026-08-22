# Dream Canvas — plugin do Figma

O painel acompanha o tema claro ou escuro do Figma: o manifesto declara
`themeColors: true` e o CSS usa as variáveis que o Figma injeta. O único
elemento fora desse esquema é o lavanda da ação primária, que é o acento do
[design system](../DESIGN.md).

Plugins do Figma rodam JavaScript num sandbox — não há como executar PyTorch lá
dentro. Então o plugin exporta o nó selecionado como PNG, manda para o DeepDream
rodando na sua máquina e insere o resultado de volta no documento.

## Instalar

1. Suba a API (é ela que faz o trabalho pesado):

   ```bash
   .venv/bin/python server.py
   ```

2. No Figma **desktop**, menu Plugins → Development → Import plugin from
   manifest… → escolha `figma-plugin/manifest.json`.

3. Selecione uma camada, abra o plugin em Plugins → Development → DeepDream.

O resultado entra como um retângulo novo ao lado do original, com a imagem como
preenchimento. Nada é sobrescrito.

## Tamanho da saída

Por padrão a imagem é reduzida para 1024px no maior lado. Marque **Manter o
tamanho original da camada** para desligar o teto — uma base 1920×1080 sai
1920×1080, levando ~11s em vez de ~4s.

O plugin exporta a camada em 1x, então o "tamanho original" é o tamanho do nó
no Figma, não o da imagem que você importou. Uma foto de 4000px colocada num
frame de 800px é exportada com 800px.

## Detalhes que custaram caro

**O manifesto não aceita IP numérico.** `http://127.0.0.1:8000` é recusado com
"must be a valid URL"; tem de ser `http://localhost:8000`. Por isso o `ui.html`
também chama `localhost`.

**O Chrome barra rede local por padrão.** No Figma web, a página é pública e a
API é local — o Private Network Access exige que o servidor responda
`Access-Control-Allow-Private-Network: true` no preflight. O `server.py` faz
isso via `allow_private_network=True` no CORSMiddleware; sem essa flag o
preflight volta 400 com "Disallowed CORS private-network", mesmo com o CORS
todo liberado.

Com os dois resolvidos, o plugin funciona tanto no app de desktop quanto no
Figma web.

## A API

Independente do plugin — serve para qualquer integração.

```bash
curl -s localhost:8000/health
```

`POST /dream` recebe `{"image": "<png em base64>", "layers": [...], "model": ...}`
e devolve `{"image": "<png em base64>", "width": ..., "height": ...}`. Os demais
parâmetros são os mesmos da CLI e todos têm padrão.
