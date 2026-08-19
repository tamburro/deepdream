"""Tema e CSS do Dream Canvas — a implementação do DESIGN.md.

Os valores aqui são os tokens do DESIGN.md. Ao mexer em cor, raio ou espaçamento,
mexa lá primeiro: o DESIGN.md é a fonte, este arquivo é a tradução para Gradio.
"""

import gradio as gr

PRIMARY = "#5E6AD2"
PRIMARY_HOVER = "#828FFF"
PRIMARY_PRESSED = "#4C55B8"

STAGE = "#000000"
CANVAS = "#0A0A0B"
SURFACE_1 = "#141416"
SURFACE_2 = "#1A1A1D"
SURFACE_3 = "#212125"

HAIRLINE = "rgba(255,255,255,0.08)"
HAIRLINE_STRONG = "rgba(255,255,255,0.14)"

INK = "#F4F4F5"
INK_MUTED = "#A1A1AA"
INK_SUBTLE = "#71717A"
INK_DISABLED = "#52525B"

READING_PAPER = "#FAFAF8"
READING_INK = "#18181B"

DANGER = "#DC2626"


def build_theme():
    return gr.themes.Base(
        primary_hue=gr.themes.colors.indigo,
        neutral_hue=gr.themes.colors.zinc,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
        radius_size=gr.themes.sizes.radius_sm,
    ).set(
        body_background_fill=CANVAS,
        body_text_color=INK_MUTED,
        body_text_color_subdued=INK_SUBTLE,
        background_fill_primary=SURFACE_1,
        background_fill_secondary=SURFACE_2,
        block_background_fill=SURFACE_1,
        block_border_color=HAIRLINE,
        block_border_width="1px",
        block_label_background_fill="transparent",
        block_label_text_color=INK_SUBTLE,
        block_label_text_weight="500",
        block_title_text_color=INK_MUTED,
        block_title_text_weight="500",
        border_color_primary=HAIRLINE,
        input_background_fill=SURFACE_2,
        input_border_color=HAIRLINE,
        input_border_color_focus=PRIMARY,
        button_primary_background_fill=PRIMARY,
        button_primary_background_fill_hover=PRIMARY_HOVER,
        button_primary_text_color="#FFFFFF",
        button_primary_border_color=PRIMARY,
        button_secondary_background_fill="transparent",
        button_secondary_border_color=HAIRLINE,
        button_secondary_text_color=INK_MUTED,
        slider_color=PRIMARY,
        checkbox_background_color=SURFACE_2,
        checkbox_background_color_selected=PRIMARY,
        checkbox_border_color=HAIRLINE_STRONG,
        checkbox_border_color_selected=PRIMARY,
        checkbox_label_background_fill=SURFACE_2,
        checkbox_label_background_fill_selected=SURFACE_3,
        checkbox_label_border_color=HAIRLINE,
        checkbox_label_text_color=INK_MUTED,
        panel_background_fill=SURFACE_1,
    )


CSS = f"""
/* ---- Base ---------------------------------------------------------- */
.gradio-container {{ max-width: 100% !important; }}
footer {{ display: none !important; }}

/* Números que mudam durante o processamento não podem alterar a largura
   da linha, ou a interface treme a cada quadro. */
input[type=number], .dc-numeric, .dc-rail .wrap span:last-child {{
  font-variant-numeric: tabular-nums;
}}

/* ---- Cabeçalho ----------------------------------------------------- */
#dc-header {{
  padding: 20px 24px 12px;
  border-bottom: 1px solid {HAIRLINE};
}}
#dc-header h1 {{
  font-size: 20px; font-weight: 600; letter-spacing: -0.3px;
  color: {INK}; margin: 0;
}}
#dc-header p {{
  font-size: 12px; color: {INK_SUBTLE}; margin: 4px 0 0;
}}
#dc-header code {{
  background: {SURFACE_2}; padding: 1px 5px; border-radius: 4px;
  font-size: 11px; color: {INK_MUTED};
}}

/* ---- Trilho de controles ------------------------------------------- */
.dc-rail {{
  background: {SURFACE_1};
  border: 1px solid {HAIRLINE};
  border-radius: 12px;
  padding: 16px;
}}
.dc-rail .block, .dc-rail .form, .dc-rail fieldset, .dc-rail .wrap {{
  border: none !important; background: transparent !important; box-shadow: none !important;
}}
.dc-rail label span {{ font-size: 13px !important; font-weight: 500 !important; }}

/* ---- Palco --------------------------------------------------------- */
/* Preto puro e sem moldura: a imagem gerada é o conteúdo, não um cartão. */
.dc-stage, .dc-stage .block {{
  background: {STAGE} !important;
  border-color: {HAIRLINE} !important;
}}
.dc-stage img, .dc-stage video {{ border-radius: 4px !important; }}

/* ---- Faixa de leitura ---------------------------------------------- */
/* Texto corrido em cinza sobre preto cansa; a inversão sinaliza
   "isto é para ler, não para operar". */
.dc-reading {{
  background: {READING_PAPER} !important;
  border-radius: 12px; padding: 20px 24px !important;
}}
.dc-reading * {{ color: {READING_INK} !important; }}
.dc-reading table {{ font-size: 13px; }}
.dc-reading th {{ border-bottom: 1px solid rgba(0,0,0,.15) !important; }}
.dc-reading td {{ border-bottom: 1px solid rgba(0,0,0,.06) !important; }}
.dc-reading code {{ background: rgba(0,0,0,.06); padding: 1px 5px; border-radius: 4px; }}

/* ---- Chips de preset ------------------------------------------------ */
/* Seleção é elevação de superfície, nunca cor. */
.dc-preset label {{
  border-radius: 999px !important;
  background: {SURFACE_2} !important;
  border: 1px solid {HAIRLINE} !important;
  padding: 6px 14px !important;
}}
.dc-preset label.selected {{
  background: {SURFACE_3} !important;
  border-color: {HAIRLINE_STRONG} !important;
  color: {INK} !important;
}}
.dc-preset input {{ display: none !important; }}

/* ---- Microcopy ------------------------------------------------------ */
.dc-hint {{ font-size: 12px !important; color: {INK_SUBTLE} !important; }}
.dc-hint p {{ margin: 2px 0 !important; }}

/* ---- Ação destrutiva ------------------------------------------------ */
button.stop {{
  background: transparent !important;
  border: 1px solid {DANGER} !important;
  color: {DANGER} !important;
}}

/* ---- Abas ----------------------------------------------------------- */
.tab-nav button {{ font-size: 13px !important; font-weight: 500 !important; }}
.tab-nav button.selected {{ color: {INK} !important; }}

/* ---- Movimento ------------------------------------------------------ */
/* A espera real já é longa; animação lenta a piora. */
* {{ transition-duration: 120ms !important; }}

/* ---- Responsivo ----------------------------------------------------- */
@media (max-width: 900px) {{
  .dc-rail {{ padding: 12px; }}
  button {{ min-height: 44px; }}
}}
"""
