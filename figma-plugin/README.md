# Plugin do Figma

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

## Por que só no app de desktop

O Figma no navegador roda em HTTPS, e uma página HTTPS não pode chamar
`http://127.0.0.1`. O app de desktop não tem essa restrição. Se precisar do
plugin no navegador, a API teria que estar atrás de HTTPS.

## A API

Independente do plugin — serve para qualquer integração.

```bash
curl -s localhost:8000/health
```

`POST /dream` recebe `{"image": "<png em base64>", "layers": [...], "model": ...}`
e devolve `{"image": "<png em base64>", "width": ..., "height": ...}`. Os demais
parâmetros são os mesmos da CLI e todos têm padrão.
