// Roda no sandbox do Figma: tem acesso ao documento, mas não à rede.
// Quem faz as chamadas HTTP é o ui.html, que roda num iframe.

figma.showUI(__html__, { width: 320, height: 560 });

function selectedRenderable() {
  return figma.currentPage.selection.filter((node) => node.exportAsync);
}

function reportSelection() {
  const nodes = selectedRenderable();
  figma.ui.postMessage({
    type: "selection",
    count: nodes.length,
    name: nodes.length === 1 ? nodes[0].name : null,
  });
}

figma.on("selectionchange", reportSelection);
reportSelection();

figma.ui.onmessage = async (message) => {
  if (message.type === "export") {
    const nodes = selectedRenderable();
    if (nodes.length === 0) {
      figma.ui.postMessage({ type: "error", message: "Selecione uma camada." });
      return;
    }

    try {
      // O DeepDream reescala internamente, então exportar em 1x já basta.
      const bytes = await nodes[0].exportAsync({ format: "PNG" });
      figma.ui.postMessage({ type: "exported", bytes });
    } catch (error) {
      figma.ui.postMessage({ type: "error", message: String(error) });
    }
    return;
  }

  if (message.type === "result") {
    const nodes = selectedRenderable();
    const source = nodes[0];
    const image = figma.createImage(new Uint8Array(message.bytes));

    // Um retângulo novo ao lado do original preserva o que já estava lá.
    const rect = figma.createRectangle();
    rect.name = `${source ? source.name : "Dream Canvas"} — dream`;
    rect.resize(message.width, message.height);
    if (source) {
      rect.x = source.x + source.width + 40;
      rect.y = source.y;
    }
    rect.fills = [{ type: "IMAGE", scaleMode: "FILL", imageHash: image.hash }];

    figma.currentPage.appendChild(rect);
    figma.currentPage.selection = [rect];
    figma.viewport.scrollAndZoomIntoView([rect]);
    figma.ui.postMessage({ type: "done" });
    return;
  }

  if (message.type === "close") {
    figma.closePlugin();
  }
};
