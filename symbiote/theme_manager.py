#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: theme_manager.py
# Author: Wadih Khairallah
# Description: 
# Created: 2024-11-26 15:51:35
# Modified: 2024-11-27 03:14:24
#!/usr/bin/env python3

import random
from prompt_toolkit.styles import Style
from InquirerPy import inquirer

class ThemeManager:
    def __init__(self):
        self.themes = {
            "toolbar_off": {
                "description": "Disable the toolbar theme.",
                "colors": {
                    '': '#f95393',
                    'prompt': '#06AC6C',
                }
            },
            "original": {
                "description": "Original symbiote theme",
                "colors": {
                    '': '#f95393',
                    'prompt': '#06AC6C',
                    'bottom-toolbar': 'bg:#FFFFFF #6e757c',
                    'bottom-toolbar.off': 'bg:#e5e5e5 #9A9A9A'
                }
            },
            "liveshell": {
                "description": "Colors for when running in shell mode.",
                "colors": {
                    '': '#FF0000',
                    'prompt': '#FFA500',
                    'bottom-toolbar': 'bg:#000000 #FF4500',
                    'bottom-toolbar.off': 'bg:#2E2E2E #B22222'
                }
            },
            "experiment": {
                "description": "Developed with the help of AI.",
                "colors": {
                    "": "#afd7ff",                  
                    "body": "blue",                 
                    "prompt": "#00d75f",            
                    "rprompt": "#ffd700",           
                    "bottom-toolbar": "bg:#ffd700 #121212", 
                    "search": "bg:#005f5f #ffffff", 
                    "search.current": "bg:#00d75f #000000", 
                    "selected": "bg:#444444 #ffffff", 
                    "scrollbar": "bg:#262626",      
                    "scrollbar.button": "#87d7d7", 
                    "menu.completion": "bg:#005f87 #ffffff", 
                    "menu.completion.current": "bg:#87afd7 #000000", 
                    "menu.border": "fg:#00d75f",
                    "scrollbar": "bg:#262626",
                    "scrollbar.button": "bg:#87d7d7",
                    "validation-toolbar": "bg:#ff0000 #ffffff", 
                    "line-number": "bg:#303030 #afd7ff", 
                    "line-number.current": "bg:#444444 #ffd700", 
                    "tab-bar": "bg:#1c1c1c #afd7ff",  
                    "tab-bar.tab": "bg:#3a3a3a #ffffff", 
                    "tab-bar.tab.selected": "bg:#005f87 #ffd700" 
                }
            },
            "primary": {
                "description": "Developed with the help of ai.",
                "colors": {
                    '': '#afd7ff',
                    "body": "blue",
                    'prompt': '#00d75f', 
                    'rprompt': '#ffd700',
                    'bottom-toolbar': 'bg:#ffd700 #121212',
                }
            },
            "default": {
                "description": "Default symbiote theme",
                "colors": {
                    '': '#DED300',
                    'prompt': '#02788E',
                    'rprompt': "#ffd700",
                    #'bottom-toolbar': 'bg:#FFFFFF #6e757c',
                    'bottom-toolbar': 'bg:#ffffff #121212',
                    'bottom-toolbar.off': 'bg:#e5e5e5 #9A9A9A'
                }
            },
            "ocean": {
                "description": "A calming theme inspired by the colors of the ocean.",
                "colors": {
                    '': '#00CED1',  # typed text color - DarkTurquoise
                    'prompt': '#FF7F50',  # prompt color - Coral
                    'bottom-toolbar': 'bg:#4682B4 #F0E68C',  # Bottom toolbar style - SteelBlue background and Khaki text
                    'bottom-toolbar.off': 'bg:#708090 #778899'  # Bottom toolbar off style - SlateGray background and LightSlateGray text
                }
            },
            "sunset": {
                "description": "A warm theme inspired by the colors of a sunset.",
                "colors": {
                    '': '#FF6347',  # typed text color - Tomato
                    'prompt': '#8A2BE2',  # prompt color - BlueViolet
                    'bottom-toolbar': 'bg:#FF8C00 #008080',  # Bottom toolbar style - DarkOrange background and Teal text
                    'bottom-toolbar.off': 'bg:#BDB76B #696969'  # Bottom toolbar off style - DarkKhaki background and DimGray text
                }
            },
            "forest": {
                "description": "A refreshing theme inspired by the colors of a forest.",
                "colors": {
                    '': '#228B22',  # typed text color - ForestGreen
                    'prompt': '#8B4513',  # prompt color - SaddleBrown
                    'bottom-toolbar': 'bg:#32CD32 #BA55D3',  # Bottom toolbar style - LimeGreen background and MediumOrchid text
                    'bottom-toolbar.off': 'bg:#556B2F #808080'  # Bottom toolbar off style - DarkOliveGreen background and Gray text
                }
            },
            "desert": {
                "description": "A theme inspired by the colors of a desert.",
                "colors": {
                    '': '#D2B48C',  # typed text color - Tan
                    'prompt': '#FF00FF',  # prompt color - Magenta
                    'bottom-toolbar': 'bg:#CD853F #00FFFF',  # Bottom toolbar style - Peru background and Cyan text
                    'bottom-toolbar.off': 'bg:#8B4513 #2F4F4F'  # Bottom toolbar off style - SaddleBrown background and DarkSlateGray text
                }
            },
            "sky": {
                "description": "A cool theme inspired by the colors of the sky.",
                "colors": {
                    '': '#87CEEB',  # typed text color - SkyBlue
                    'prompt': '#FFD700',  # prompt color - Gold
                    'bottom-toolbar': 'bg:#1E90FF #FF69B4',  # Bottom toolbar style - DodgerBlue background and HotPink text
                    'bottom-toolbar.off': 'bg:#6A5ACD #2F4F4F'  # Bottom toolbar off style - SlateBlue background and DarkSlateGray text
                }
            },
            "rose": {
                "description": "A romantic theme inspired by the colors of a rose.",
                "colors": {
                    '': '#FFB6C1',  # typed text color - LightPink
                    'prompt': '#008080',  # prompt color - Teal
                    'bottom-toolbar': 'bg:#FF1493 #FFDAB9',  # Bottom toolbar style - DeepPink background and PeachPuff text
                    'bottom-toolbar.off': 'bg:#C71585 #708090'  # Bottom toolbar off style - MediumVioletRed background and SlateGray text
                }
            },
            "earth": {
                "description": "A grounded theme inspired by the colors of the earth.",
                "colors": {
                    '': '#6B8E23',  # typed text color - OliveDrab
                    'prompt': '#DA70D6',  # prompt color - Orchid
                    'bottom-toolbar': 'bg:#8B4513 #ADD8E6',  # Bottom toolbar style - SaddleBrown background and LightBlue text
                    'bottom-toolbar.off': 'bg:#556B2F #696969'  # Bottom toolbar off style - DarkOliveGreen background and DimGray text
                }
            },
            "night": {
                "description": "A dark theme inspired by the colors of the night.",
                "colors": {
                    '': '#483D8B',  # typed text color - DarkSlateBlue
                    'prompt': '#FF4500',  # prompt color - OrangeRed
                    'bottom-toolbar': 'bg:#2F4F4F #FFDEAD',  # Bottom toolbar style - DarkSlateGray background and NavajoWhite text
                    'bottom-toolbar.off': 'bg:#696969 #DCDCDC'  # Bottom toolbar off style - DimGray background and Gainsboro text
                }
            },
            "spring": {
                "description": "A lively theme inspired by the colors of spring.",
                "colors": {
                    '': '#ADFF2F',  # typed text color - GreenYellow
                    'prompt': '#DB7093',  # prompt color - PaleVioletRed
                    'bottom-toolbar': 'bg:#90EE90 #A52A2A',  # Bottom toolbar style - LightGreen background and Brown text
                    'bottom-toolbar.off': 'bg:#7CFC00 #A9A9A9'  # Bottom toolbar off style - LawnGreen background and DarkGray text
                }
            },
            "winter": {
                "description": "A cool theme inspired by the colors of winter.",
                "colors": {
                    '': '#00BFFF',  # typed text color - DeepSkyBlue
                    'prompt': '#FF6347',  # prompt color - Tomato
                    'bottom-toolbar': 'bg:#4169E1 #D2691E',  # Bottom toolbar style - RoyalBlue background and Chocolate text
                    'bottom-toolbar.off': 'bg:#4682B4 #D3D3D3'  # Bottom toolbar off style - SteelBlue background and LightGray text
                }
            },
            "autumn": {
                "description": "A warm theme inspired by the colors of autumn.",
                "colors": {
                    '': '#FFD700',  # typed text color - Gold
                    'prompt': '#7B68EE',  # prompt color - MediumSlateBlue
                    'bottom-toolbar': 'bg:#FF8C00 #20B2AA',  # Bottom toolbar style - DarkOrange background and LightSeaGreen text
                    'bottom-toolbar.off': 'bg:#DAA520 #808080'  # Bottom toolbar off style - GoldenRod background and Gray text
                }
            },
            "summer": {
                "description": "A bright theme inspired by the colors of summer.",
                "colors": {
                    '': '#FFFF00',  # typed text color - Yellow
                    'prompt': '#6A5ACD',  # prompt color - SlateBlue
                    'bottom-toolbar': 'bg:#FFA500 #5F9EA0',  # Bottom toolbar style - Orange background and CadetBlue text
                    'bottom-toolbar.off': 'bg:#FFD700 #778899'  # Bottom toolbar off style - Gold background and LightSlateGray text
                }
            },
            "rainbow": {
                "description": "A vibrant theme inspired by the colors of a rainbow.",
                "colors": {
                    '': '#EE82EE',  # typed text color - Violet
                    'prompt': '#3CB371',  # prompt color - MediumSeaGreen
                    'bottom-toolbar': 'bg:#8B0000 #00FA9A',  # Bottom toolbar style - DarkRed background and MediumSpringGreen text
                    'bottom-toolbar.off': 'bg:#FF4500 #708090'  # Bottom toolbar off style - OrangeRed background and SlateGray text
                }
            },
            "pastel": {
                "description": "A soft theme inspired by pastel colors.",
                "colors": {
                    '': '#FFB6C1',  # typed text color - LightPink
                    'prompt': '#9370DB',  # prompt color - MediumPurple
                    'bottom-toolbar': 'bg:#D8BFD8 #008B8B',  # Bottom toolbar style - Thistle background and DarkCyan text
                    'bottom-toolbar.off': 'bg:#DDA0DD #708090'  # Bottom toolbar off style - Plum background and SlateGray text
                }
            },
            "neon": {
                "description": "A flashy theme inspired by neon colors.",
                "colors": {
                    '': '#00FF00',  # typed text color - Lime
                    'prompt': '#FF1493',  # prompt color - DeepPink
                    'bottom-toolbar': 'bg:#00FFFF #DC143C',  # Bottom toolbar style - Cyan background and Crimson text
                    'bottom-toolbar.off': 'bg:#7FFF00 #2F4F4F'  # Bottom toolbar off style - Chartreuse background and DarkSlateGray text
                }
            },
            "monochrome": {
                "description": "A classic theme with shades of grey.",
                "colors": {
                    '': '#808080',  # typed text color - Gray
                    'prompt': '#ffffff',  # prompt color - White
                    'bottom-toolbar': 'bg:#C0C0C0 #000000',  # Bottom toolbar style - Silver background and Black text
                    'bottom-toolbar.off': 'bg:#696969 #A9A9A9'  # Bottom toolbar off style - DimGray background and DarkGray text
                }
            },
            "lavender": {
                "description": "A soothing theme inspired by the color of lavender.",
                "colors": {
                    '': '#E6E6FA',  # typed text color - Lavender
                    'prompt': '#FF00FF',  # prompt color - Fuchsia
                    'bottom-toolbar': 'bg:#9370DB #48D1CC',  # Bottom toolbar style - MediumPurple background and MediumTurquoise text
                    'bottom-toolbar.off': 'bg:#8A2BE2 #D3D3D3'  # Bottom toolbar off style - BlueViolet background and LightGray text
                }
            },
            "citrus": {
                "description": "A refreshing theme inspired by the colors of citrus fruits.",
                "colors": {
                    '': '#FFA500',  # typed text color - Orange
                    'prompt': '#32CD32',  # prompt color - LimeGreen
                    'bottom-toolbar': 'bg:#FFFF00 #FF4500',  # Bottom toolbar style - Yellow background and OrangeRed text
                    'bottom-toolbar.off': 'bg:#ADFF2F #8B4513'  # Bottom toolbar off style - GreenYellow background and SaddleBrown text
                }
            },
            "berry": {
                "description": "A sweet theme inspired by the colors of berries.",
                "colors": {
                    '': '#DC143C',  # typed text color - Crimson
                    'prompt': '#8B008B',  # prompt color - DarkMagenta
                    'bottom-toolbar': 'bg:#FF69B4 #FFFF00',  # Bottom toolbar style - HotPink background and Yellow text
                    'bottom-toolbar.off': 'bg:#C71585 #808080'  # Bottom toolbar off style - MediumVioletRed background and Gray text
                }
            },
            "coffee": {
                "description": "A comforting theme inspired by the colors of coffee.",
                "colors": {
                    '': '#8B4513',  # typed text color - SaddleBrown
                    'prompt': '#D2B48C',  # prompt color - Tan
                    'bottom-toolbar': 'bg:#A0522D #FFFACD',  # Bottom toolbar style - Sienna background and LemonChiffon text
                    'bottom-toolbar.off': 'bg:#CD853F #2F4F4F'  # Bottom toolbar off style - Peru background and DarkSlateGray text
                }
            }
        }


    def get_theme(self, theme_name):
        self.generate_random_theme()
        if theme_name in self.themes:
            theme_colors = self.themes[theme_name]['colors']
            return Style.from_dict(theme_colors)
        else:
            print("Theme not found.")
            return None

    def select_theme(self):
        self.generate_random_theme()
        theme_list = list(self.themes.keys())

        theme_name = inquirer.select(
            message="Select a theme:",
            choices=theme_list,
        ).execute()

        theme_colors = self.themes[theme_name]['colors']

        return theme_name, Style.from_dict(theme_colors)

    def generate_random_theme(self):
        def random_color():
            return f'#{random.randint(0, 255):02x}{random.randint(0, 255):02x}{random.randint(0, 255):02x}'

        def contrast_color(color):
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            yiq = ((r * 299) + (g * 587) + (b * 114)) / 1000
            return '#000000' if yiq >= 128 else '#ffffff'

        def color_difference(color1, color2):
            r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
            r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)
            return abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2)

        colors = []
        for _ in range(6):
            attempts = 0
            new_color = random_color()
            while any(color_difference(new_color, color) < 100 for color in colors) and attempts < 100:
                new_color = random_color()
                attempts += 1
            colors.append(new_color)

        self.themes['random'] = {
            "description": "Randomly generated theme.",
            "colors": {
                '': colors[0],
                'prompt': colors[1],
                'bottom-toolbar': f'bg:{colors[2]} {contrast_color(colors[2])}',
                'bottom-toolbar.off': f'bg:{colors[3]} {contrast_color(colors[3])}',
            }
        }

        return


# usage
'''
from themes import ThemeManager

theme_manager = ThemeManager()

# Change to selected theme
prompt_style = theme_manager.select_theme()
'''
