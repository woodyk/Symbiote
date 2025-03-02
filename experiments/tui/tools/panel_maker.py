#!/usr/bin/env python3
#
# panel_maker.py

import re
import unicodedata
from wcwidth import wcwidth


class BoxGenerator:
    """
    A library for generating box patterns and plain boxes with a given character.
    """

    @staticmethod
    def _visual_width(text: str) -> int:
        """Calculate the visual width of a string."""
        return sum(max(wcwidth(char), 1) for char in text)

    @staticmethod
    def _sanity_check(box: str, width: int) -> bool:
        """
        Validates that all lines in the box have consistent visual length.

        Args:
            box (str): The box string to validate.
            width (int): Expected visual width of each line.

        Returns:
            bool: True if the box passes the sanity check, False otherwise.
        """
        lines = box.splitlines()
        for idx, line in enumerate(lines):
            visual_length = BoxGenerator._visual_width(line)
            if visual_length != width:
                print(f"Sanity Check Failed on Line {idx + 1}: Line Visual Length={visual_length}, Expected={width}")
                return False
        return True

    @staticmethod
    def _is_combining_character(char: str) -> bool:
        """Check if a character is a combining character."""
        return unicodedata.combining(char) != 0

    @staticmethod
    def create_plain_box(char: str, width: int = 20, height: int = 10) -> str:
        """
        Create a plain box of given dimensions using a single character.

        Args:
            char (str): Character to use for the box.
            width (int): Width of the box.
            height (int): Height of the box.

        Returns:
            str: The plain box as a string.
        """
        if BoxGenerator._is_combining_character(char):
            raise ValueError(f"Character '{char}' is a combining character and cannot be used for a box.")

        char_width = max(wcwidth(char), 1)
        horizontal_count = width // char_width
        internal_space_width = width - (2 * char_width)

        if internal_space_width < 0:
            raise ValueError(f"Width too small for character '{char}'.")

        top = char * horizontal_count
        middle = char + " " * internal_space_width + char

        box = f"{top}\n" + f"{middle}\n" * (height - 2) + f"{top}\n"

        if not BoxGenerator._sanity_check(box, width):
            raise ValueError(f"Generated box failed sanity check:\n{box}")

        return box

    @staticmethod
    def create_box_with_title(char: str, width: int = 20, height: int = 10, title: str = "test") -> str:
        """
        Create a box with a title in the top bar, starting at the third character.

        Args:
            char (str): Character to use for the box.
            width (int): Width of the box.
            height (int): Height of the box.
            title (str): Title to display in the top bar.

        Returns:
            str: The box with the title as a string.
        """
        if BoxGenerator._is_combining_character(char):
            raise ValueError(f"Character '{char}' is a combining character and cannot be used for a box.")

        char_width = max(wcwidth(char), 1)
        horizontal_count = width // char_width

        if BoxGenerator._visual_width(title) > width - (2 * char_width):
            raise ValueError("Title is too long to fit in the box.")

        # Append a space to the title if the character is full-width
        if char_width == 2:
            title += " "

        # Calculate padding for the title
        title_visual_length = BoxGenerator._visual_width(title)
        total_padding = width - title_visual_length
        left_padding = (total_padding // 2) // char_width
        right_padding = horizontal_count - left_padding - (title_visual_length // char_width)

        # Construct title row
        title_row = (
            char * left_padding
            + title
            + char * right_padding
        )

        # Adjust internal space width
        internal_space_width = width - (2 * char_width)
        if internal_space_width < 0:
            raise ValueError(f"Width too small for character '{char}'.")

        middle = char + " " * internal_space_width + char
        bottom_row = char * horizontal_count

        box = f"{title_row}\n" + f"{middle}\n" * (height - 2) + f"{bottom_row}\n"

        if not BoxGenerator._sanity_check(box, width):
            raise ValueError(f"Generated box failed sanity check:\n{box}")

        return box


# Example Usage
if __name__ == "__main__":
    generator = BoxGenerator()

    # Test with half-width character
    print("Plain Box with '*':")
    print(generator.create_plain_box(char="*", width=40))

    # Test with full-width character
    print("\nPlain Box with 'ゎ':")
    print(generator.create_plain_box(char="ゎ", width=40))

    # Test with full-width character and title
    print("\nBox with Title using 'ゎ':")
    print(generator.create_box_with_title(char="ゎ", width=40, title="Welcome"))

    # Testing a string of characters.
    test_chars = "ノ⣶"
    for char in test_chars:
        print(f"\nPlain box test. {char}")
        test_plain = generator.create_plain_box(char=char, width=40)
        print(test_plain)

        print(f"\nTitle box test. {char}")
        test_title = generator.create_box_with_title(char=char, title="Welcome", width=40)
        print(test_title)

# Example Usage
if __name__ == "__main__":
    generator = BoxGenerator()
    # Test with half-width character
    plain_box_ascii = generator.create_plain_box(char="*")
    print("Plain Box with '*':")
    print(plain_box_ascii)

    # Test with full-width character
    plain_box_fullwidth = generator.create_plain_box(char="ゎ")
    print("Plain Box with 'ゎ':")
    print(plain_box_fullwidth)

    # Test box with title and full-width character
    box_with_title = generator.create_box_with_title(char="ゎ", title="Welcome")
    print("Box with Title using 'ゎ':")
    print(box_with_title)

    # Testing a string of characters.
    test_chars = """
」 『 』 【 】 〒 〓 〔 〕 〖 〗 〘 〙 〚 〛 〜 〝 〞 〟 〠 〡 〢 〣 〤 〥 〦 〧 〨 〩 〪 
〫 〬 〭 〮 〯 〰 〱 〲 〳 〴 〵 〶 〷 〸 〹 〺 〻 〼 〽 〾 〿 ぀ ぁ あ ぃ い ぅ う ぇ え 
ぉ お か が き ぎ く ぐ け げ こ ご さ ざ し じ す ず せ ぜ そ ぞ た だ ち ぢ っ つ づ て 
で と ど な に ぬ ね の は ば ぱ ひ び ぴ ふ ぶ ぷ へ べ ぺ ほ ぼ ぽ ま み む め も ゃ や 
ゅ ゆ ょ よ ら り る れ ろ ゎ わ ゐ ゑ を ん ゔ ゕ ゖ ゗ ゘ ゙ ゚ ゛ ゜ ゝ ゞ ゟ ゠ ァ ア 
ィ イ ゥ ウ ェ エ ォ オ カ ガ キ ギ ク グ ケ ゲ コ ゴ サ ザ シ ジ ス ズ セ ゼ ソ ゾ タ ダ 
チ ヂ ッ ツ ヅ テ デ ト ド ナ ニ ヌ ネ ノ ハ ⣶
"""
    test_chars = re.sub(r" |\n", "", test_chars)
    for char in test_chars:
        try:
            print(f"Plain box test. {char}")
            test_fullwidth = generator.create_plain_box(char=char, width=40)
            print(test_fullwidth)

            print(f"Title box test. {char}")
            test_title = generator.create_box_with_title(char=char, title="Welcome", width=40)
            print(test_title)
        except Exception as e:
            print(e)
            continue
