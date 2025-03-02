#!/usr/bin/env python3
#
# patterns.py

CHINESE_BOX_1 = Box(
    "╭日月╮\n"
    "┆   ┆\n"
    "├火水┤\n"
    "┆   ┆\n"
    "├金木┤\n"
    "├风雷┤\n"
    "┆   ┆\n"
    "╰山川╯\n"
)

CHINESE_BOX_2 = Box(
    "╭星辰╮\n"
    "┆   ┆\n"
    "├龙凤┤\n"
    "┆   ┆\n"
    "├虎鹤┤\n"
    "├剑刀┤\n"
    "┆   ┆\n"
    "╰琴棋╯\n"
)

CHINESE_BOX_3 = Box(
    "╭青龙╮\n"
    "┆   ┆\n"
    "├白虎┤\n"
    "┆   ┆\n"
    "├朱雀┤\n"
    "├玄武┤\n"
    "┆   ┆\n"
    "╰神仙╯\n"
)

CHINESE_BOX_4 = Box(
    "╭乾坤╮\n"
    "┆   ┆\n"
    "├阴阳┤\n"
    "┆   ┆\n"
    "├天地┤\n"
    "├日夜┤\n"
    "┆   ┆\n"
    "╰光影╯\n"
)

DASHED_ROUNDED = Box(
    "╭┄┄┄╮\n"
    "┆   ┆\n"
    "├┄┄┄┤\n"
    "┆   ┆\n"
    "├┄┄┄┤\n"
    "├┄┄┄┤\n"
    "┆   ┆\n"
    "╰┄┄┄╯\n"
)
DIAMOND_BORDER = Box(
    "◆━◆━◆\n"
    "◇   ◇\n"
    "◆━◆━◆\n"
    "◇   ◇\n"
    "◆━◆━◆\n"
    "◆━◆━◆\n"
    "◇   ◇\n"
    "◆━◆━◆\n"
)
ZIGZAG = Box(
    "╱─╮╭╲\n"
    "│   │\n"
    "╰─┼─╯\n"
    "│   │\n"
    "╱─┼─╲\n"
    "│   │\n"
    "╰─┼─╯\n"
    "╲─╯╰╱\n"
)
TRIPLE_LINE = Box(
    "┏━━━┳┓\n"
    "┃   ┃┃\n"
    "┣━━━╋┫\n"
    "┃   ┃┃\n"
    "┣━━━╋┫\n"
    "┣━━━╋┫\n"
    "┃   ┃┃\n"
    "┗━━━┻┛\n"
)
THICK_DOTTED = Box(
    "●●●●●\n"
    "●   ●\n"
    "●●●●●\n"
    "●   ●\n"
    "●●●●●\n"
    "●●●●●\n"
    "●   ●\n"
    "●●●●●\n"
)
MIXED_STYLE = Box(
    "╭─┬─╮\n"
    "│   │\n"
    "├─┼─┤\n"
    "│   │\n"
    "├─┼─┤\n"
    "├─┼─┤\n"
    "│   │\n"
    "╰─┴─╯\n"
)
BRAILLE = Box(
    "⠿⠿⠿⠿⠿\n"
    "⠿   ⠿\n"
    "⠿⠿⠿⠿⠿\n"
    "⠿   ⠿\n"
    "⠿⠿⠿⠿⠿\n"
    "⠿⠿⠿⠿⠿\n"
    "⠿   ⠿\n"
    "⠿⠿⠿⠿⠿\n"
)
CURVED_BRACKET = Box(
    "╭───╮\n"
    "│   │\n"
    "╞═══╡\n"
    "│   │\n"
    "╞═══╡\n"
    "╞═══╡\n"
    "│   │\n"
    "╰───╯\n"
)
LIGHTNING = Box(
    "⚡⚡⚡⚡⚡\n"
    "⚡   ⚡\n"
    "⚡⚡⚡⚡⚡\n"
    "⚡   ⚡\n"
    "⚡⚡⚡⚡⚡\n"
    "⚡⚡⚡⚡⚡\n"
    "⚡   ⚡\n"
    "⚡⚡⚡⚡⚡\n"
)
STAR_BORDER = Box(
    "★ ★ ★ ★ ★\n"
    "★       ★\n"
    "★ ★ ★ ★ ★\n"
    "★       ★\n"
    "★ ★ ★ ★ ★\n"
    "★ ★ ★ ★ ★\n"
    "★       ★\n"
    "★ ★ ★ ★ ★\n"
)

CHINESE_BASIC = Box(
    "田田田田田\n"
    "口  口  口\n"
    "田田田田田\n"
    "口  口  口\n"
    "田田田田田\n"
    "田田田田田\n"
    "口  口  口\n"
    "田田田田田\n"
)

CHINESE_ROUNDED = Box(
    "囗囗囗囗囗\n"
    "口  口  口\n"
    "囗囗囗囗囗\n"
    "口  口  口\n"
    "囗囗囗囗囗\n"
    "囗囗囗囗囗\n"
    "口  口  口\n"
    "囗囗囗囗囗\n"
)

UNICODE_CIRCLE = Box(
    "●●●●●\n"
    "○  ○○\n"
    "●●●●●\n"
    "○  ○○\n"
    "●●●●●\n"
    "●●●●●\n"
    "○  ○○\n"
    "●●●●●\n"
)

UNICODE_DIAMOND = Box(
    "◆◆◆◆◆\n"
    "◇  ◇◇\n"
    "◆◆◆◆◆\n"
    "◇  ◇◇\n"
    "◆◆◆◆◆\n"
    "◆◆◆◆◆\n"
    "◇  ◇◇\n"
    "◆◆◆◆◆\n"
)
SOLID_BLOCK = Box(
    "█████\n"
    "█   █\n"
    "█████\n"
    "█   █\n"
    "█████\n"
    "█████\n"
    "█   █\n"
    "█████\n"
)
SHADED_BLOCK = Box(
    "▓▓▓▓▓\n"
    "▓   ▓\n"
    "▓▓▓▓▓\n"
    "▓   ▓\n"
    "▓▓▓▓▓\n"
    "▓▓▓▓▓\n"
    "▓   ▓\n"
    "▓▓▓▓▓\n"
)
DOTTED_LINE = Box(
    "┈┈┈┈┈\n"
    "┊   ┊\n"
    "┈┈┈┈┈\n"
    "┊   ┊\n"
    "┈┈┈┈┈\n"
    "┈┈┈┈┈\n"
    "┊   ┊\n"
    "┈┈┈┈┈\n"
)
DASHED_LINE = Box(
    "┉┉┉┉┉\n"
    "┋   ┋\n"
    "┉┉┉┉┉\n"
    "┋   ┋\n"
    "┉┉┉┉┉\n"
    "┉┉┉┉┉\n"
    "┋   ┋\n"
    "┉┉┉┉┉\n"
)
MIXED_ROUNDED = Box(
    "╭○○○╮\n"
    "│   │\n"
    "├○○○┤\n"
    "│   │\n"
    "├○○○┤\n"
    "├○○○┤\n"
    "│   │\n"
    "╰○○○╯\n"
)
MIXED_HEAVY_THIN = Box(
    "┏━┳━┓\n"
    "┃ ┃ ┃\n"
    "┣━╋━┫\n"
    "┃ ┃ ┃\n"
    "┣━╋━┫\n"
    "┣━╋━┫\n"
    "┃ ┃ ┃\n"
    "┗━┻━┛\n"
)

HEBREW_BOX = Box(
    "ש══ש══ש\n"
    "ל       ל\n"
    "ש══ש══ש\n"
    "ל       ל\n"
    "ש══ש══ש\n"
    "ש══ש══ש\n"
    "ל       ל\n"
    "ש══ש══ש\n"
)

TAMIL_BOX = Box(
    "ஃ══ஃ══ஃ\n"
    "ம       ம\n"
    "ஃ══ஃ══ஃ\n"
    "ம       ம\n"
    "ஃ══ஃ══ஃ\n"
    "ஃ══ஃ══ஃ\n"
    "ம       ம\n"
    "ஃ══ஃ══ஃ\n"
)
BENGALI_BOX = Box(
    "ঙ━━ঙ━━ঙ\n"
    "ম       ম\n"
    "ঙ━━ঙ━━ঙ\n"
    "ম       ম\n"
    "ঙ━━ঙ━━ঙ\n"
    "ঙ━━ঙ━━ঙ\n"
    "ম       ম\n"
    "ঙ━━ঙ━━ঙ\n"
)
JAPANESE_BOX = Box(
    "フ━━フ━━フ\n"
    "コ       コ\n"
    "フ━━フ━━フ\n"
    "コ       コ\n"
    "フ━━フ━━フ\n"
    "フ━━フ━━フ\n"
    "コ       コ\n"
    "フ━━フ━━フ\n"
)
KOREAN_BOX = Box(
    "ㅂ━━ㅂ━━ㅂ\n"
    "ㅁ       ㅁ\n"
    "ㅂ━━ㅂ━━ㅂ\n"
    "ㅁ       ㅁ\n"
    "ㅂ━━ㅂ━━ㅂ\n"
    "ㅂ━━ㅂ━━ㅂ\n"
    "ㅁ       ㅁ\n"
    "ㅂ━━ㅂ━━ㅂ\n"
)
THAI_BOX = Box(
    "ธ══ธ══ธ\n"
    "ษ       ษ\n"
    "ธ══ธ══ธ\n"
    "ษ       ษ\n"
    "ธ══ธ══ธ\n"
    "ธ══ธ══ธ\n"
    "ษ       ษ\n"
    "ธ══ธ══ธ\n"
)
ORNAMENTAL_BOX = Box(
    "✶══✶══✶\n"
    "⚑       ⚑\n"
    "✶══✶══✶\n"
    "⚑       ⚑\n"
    "✶══✶══✶\n"
    "✶══✶══✶\n"
    "⚑       ⚑\n"
    "✶══✶══✶\n"
)
MATH_BOX = Box(
    "∑══∑══∑\n"
    "∣       ∣\n"
    "∑══∑══∑\n"
    "∣       ∣\n"
    "∑══∑══∑\n"
    "∑══∑══∑\n"
    "∣       ∣\n"
    "∑══∑══∑\n"
)


ASCII: Box = Box(
    "+--+\n"
    "| ||\n"
    "|-+|\n"
    "| ||\n"
    "|-+|\n"
    "|-+|\n"
    "| ||\n"
    "+--+\n",
    ascii=True,
)

ASCII2: Box = Box(
    "+-++\n"
    "| ||\n"
    "+-++\n"
    "| ||\n"
    "+-++\n"
    "+-++\n"
    "| ||\n"
    "+-++\n",
    ascii=True,
)

ASCII_DOUBLE_HEAD: Box = Box(
    "+-++\n"
    "| ||\n"
    "+=++\n"
    "| ||\n"
    "+-++\n"
    "+-++\n"
    "| ||\n"
    "+-++\n",
    ascii=True,
)

SQUARE: Box = Box(
    "┌─┬┐\n"
    "│ ││\n"
    "├─┼┤\n"
    "│ ││\n"
    "├─┼┤\n"
    "├─┼┤\n"
    "│ ││\n"
    "└─┴┘\n"
)

SQUARE_DOUBLE_HEAD: Box = Box(
    "┌─┬┐\n"
    "│ ││\n"
    "╞═╪╡\n"
    "│ ││\n"
    "├─┼┤\n"
    "├─┼┤\n"
    "│ ││\n"
    "└─┴┘\n"
)

MINIMAL: Box = Box(
    "  ╷ \n"
    "  │ \n"
    "╶─┼╴\n"
    "  │ \n"
    "╶─┼╴\n"
    "╶─┼╴\n"
    "  │ \n"
    "  ╵ \n"
)


MINIMAL_HEAVY_HEAD: Box = Box(
    "  ╷ \n"
    "  │ \n"
    "╺━┿╸\n"
    "  │ \n"
    "╶─┼╴\n"
    "╶─┼╴\n"
    "  │ \n"
    "  ╵ \n"
)

MINIMAL_DOUBLE_HEAD: Box = Box(
    "  ╷ \n"
    "  │ \n"
    " ═╪ \n"
    "  │ \n"
    " ─┼ \n"
    " ─┼ \n"
    "  │ \n"
    "  ╵ \n"
)


SIMPLE: Box = Box(
    "    \n"
    "    \n"
    " ── \n"
    "    \n"
    "    \n"
    " ── \n"
    "    \n"
    "    \n"
)

SIMPLE_HEAD: Box = Box(
    "    \n"
    "    \n"
    " ── \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
)


SIMPLE_HEAVY: Box = Box(
    "    \n"
    "    \n"
    " ━━ \n"
    "    \n"
    "    \n"
    " ━━ \n"
    "    \n"
    "    \n"
)


HORIZONTALS: Box = Box(
    " ── \n"
    "    \n"
    " ── \n"
    "    \n"
    " ── \n"
    " ── \n"
    "    \n"
    " ── \n"
)

ROUNDED: Box = Box(
    "╭─┬╮\n"
    "│ ││\n"
    "├─┼┤\n"
    "│ ││\n"
    "├─┼┤\n"
    "├─┼┤\n"
    "│ ││\n"
    "╰─┴╯\n"
)

HEAVY: Box = Box(
    "┏━┳┓\n"
    "┃ ┃┃\n"
    "┣━╋┫\n"
    "┃ ┃┃\n"
    "┣━╋┫\n"
    "┣━╋┫\n"
    "┃ ┃┃\n"
    "┗━┻┛\n"
)

HEAVY_EDGE: Box = Box(
    "┏━┯┓\n"
    "┃ │┃\n"
    "┠─┼┨\n"
    "┃ │┃\n"
    "┠─┼┨\n"
    "┠─┼┨\n"
    "┃ │┃\n"
    "┗━┷┛\n"
)

HEAVY_HEAD: Box = Box(
    "┏━┳┓\n"
    "┃ ┃┃\n"
    "┡━╇┩\n"
    "│ ││\n"
    "├─┼┤\n"
    "├─┼┤\n"
    "│ ││\n"
    "└─┴┘\n"
)

DOUBLE: Box = Box(
    "╔═╦╗\n"
    "║ ║║\n"
    "╠═╬╣\n"
    "║ ║║\n"
    "╠═╬╣\n"
    "╠═╬╣\n"
    "║ ║║\n"
    "╚═╩╝\n"
)

DOUBLE_EDGE: Box = Box(
    "╔═╤╗\n"
    "║ │║\n"
    "╟─┼╢\n"
    "║ │║\n"
    "╟─┼╢\n"
    "╟─┼╢\n"
    "║ │║\n"
    "╚═╧╝\n"
)

MARKDOWN: Box = Box(
    "    \n"
    "| ||\n"
    "|-||\n"
    "| ||\n"
    "|-||\n"
    "|-||\n"
    "| ||\n"
    "    \n",
    ascii=True,
)
from ascii_tui import Box

# Standard ASCII Box
ASCII_BOX = Box(
    "+--+\n"
    "|  |\n"
    "|--|\n"
    "|  |\n"
    "|--|\n"
    "|--|\n"
    "|  |\n"
    "+--+\n",
    ascii=True,
)

# Double Line Box
DOUBLE_LINE_BOX = Box(
    "╔═╦╗\n"
    "║ ║║\n"
    "╠═╬╣\n"
    "║ ║║\n"
    "╠═╬╣\n"
    "╠═╬╣\n"
    "║ ║║\n"
    "╚═╩╝\n"
)

# Rounded Corners Box
ROUNDED_BOX = Box(
    "╭─┬╮\n"
    "│ ││\n"
    "├─┼┤\n"
    "│ ││\n"
    "├─┼┤\n"
    "├─┼┤\n"
    "│ ││\n"
    "╰─┴╯\n"
)

# Heavy Borders Box
HEAVY_BOX = Box(
    "┏━┳┓\n"
    "┃ ┃┃\n"
    "┣━╋┫\n"
    "┃ ┃┃\n"
    "┣━╋┫\n"
    "┣━╋┫\n"
    "┃ ┃┃\n"
    "┗━┻┛\n"
)

# Minimal Box
MINIMAL_BOX = Box(
    "  ╷ \n"
    "  │ \n"
    "╶─┼╴\n"
    "  │ \n"
    "╶─┼╴\n"
    "╶─┼╴\n"
    "  │ \n"
    "  ╵ \n"
)

# Minimal Heavy Head Box
MINIMAL_HEAVY_HEAD_BOX = Box(
    "  ╷ \n"
    "  │ \n"
    "╺━┿╸\n"
    "  │ \n"
    "╶─┼╴\n"
    "╶─┼╴\n"
    "  │ \n"
    "  ╵ \n"
)

# Minimal Double Head Box
MINIMAL_DOUBLE_HEAD_BOX = Box(
    "  ╷ \n"
    "  │ \n"
    " ═╪ \n"
    "  │ \n"
    " ─┼ \n"
    " ─┼ \n"
    "  │ \n"
    "  ╵ \n"
)

# Simple Box
SIMPLE_BOX = Box(
    "    \n"
    "    \n"
    " ── \n"
    "    \n"
    "    \n"
    " ── \n"
    "    \n"
    "    \n",
    ascii=True,
)

# Simple Head Box
SIMPLE_HEAD_BOX = Box(
    "    \n"
    "    \n"
    " ── \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n",
    ascii=True,
)

# Simple Heavy Box
SIMPLE_HEAVY_BOX = Box(
    "    \n"
    "    \n"
    " ━━ \n"
    "    \n"
    "    \n"
    " ━━ \n"
    "    \n"
    "    \n",
    ascii=True,
)

# Horizontals Box
HORIZONTALS_BOX = Box(
    " ── \n"
    "    \n"
    " ── \n"
    "    \n"
    " ── \n"
    " ── \n"
    "    \n"
    " ── \n"
)

# Markdown Box
MARKDOWN_BOX = Box(
    "    \n"
    "|  |\n"
    "|- |\n"
    "|  |\n"
    "|- |\n"
    "|- |\n"
    "|  |\n"
    "    \n",
    ascii=True,
)
from ascii_tui import Box

# Chinese Symbols Box
CHINESE_SYMBOLS_BOX = Box(
    "田田田田田田田田田田\n"
    "口        口\n"
    "口        口\n"
    "口        口\n"
    "口        口\n"
    "口        口\n"
    "口        口\n"
    "田田田田田田田田田田\n"
)

# Chinese Punctuation Box
CHINESE_PUNCTUATION_BOX = Box(
    "【――――――――――】\n"
    "｜          ｜\n"
    "｜          ｜\n"
    "｜          ｜\n"
    "｜          ｜\n"
    "｜          ｜\n"
    "｜          ｜\n"
    "【――――――――――】\n"
)

# Hybrid Box (Mixed Chinese and Box-Drawing Characters)
HYBRID_BOX = Box(
    "田田田田田田田田田田\n"
    "┃        ┃\n"
    "┃        ┃\n"
    "┃        ┃\n"
    "┃        ┃\n"
    "┃        ┃\n"
    "┃        ┃\n"
    "田田田田田田田田田田\n"
)

# CJK Compatibility Forms Box
CJK_COMPATIBILITY_BOX = Box(
    "【―――┬―――】\n"
    "｜     │     ｜\n"
    "╟─────┼─────╢\n"
    "｜     │     ｜\n"
    "╟─────┼─────╢\n"
    "╟─────┼─────╢\n"
    "｜     │     ｜\n"
    "【―――┴―――】\n"
)

