# --- Color Class ---
class Color:
    # Reset
    RESET = '\033[0m'

    # User's Original Bright Text Colors (Foreground) - Preserved to maintain compatibility
    RED = '\033[91m' # Original RED was bright red
    GREEN = '\033[92m' # Original GREEN was bright green
    YELLOW = '\033[93m' # Original YELLOW was bright yellow
    BLUE = '\033[94m' # Original BLUE was bright blue
    PURPLE = '\033[95m' # Original PURPLE was bright magenta

    # Standard (Normal) Text Colors (Foreground) - New names for these
    NORMAL_BLACK = '\033[30m'
    NORMAL_RED = '\033[31m'
    NORMAL_GREEN = '\033[32m'
    NORMAL_YELLOW = '\033[33m'
    NORMAL_BLUE = '\033[34m'
    NORMAL_MAGENTA = '\033[35m' # Standard magenta (different from user's original PURPLE)
    NORMAL_CYAN = '\033[36m'
    NORMAL_WHITE = '\033[37m'
    NORMAL_LIGHT_GRAY = '\033[37m' # Alias for NORMAL_WHITE

    # Bright Text Colors (Foreground) - Added for completeness beyond original user colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'

    # Background Colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'

    # Bright Background Colors
    BG_BRIGHT_BLACK = '\033[100m'
    BG_BRIGHT_RED = '\033[101m'
    BG_BRIGHT_GREEN = '\033[102m'
    BG_BRIGHT_YELLOW = '\033[103m'
    BG_BRIGHT_BLUE = '\033[104m'
    BG_BRIGHT_MAGENTA = '\033[105m'
    BG_BRIGHT_CYAN = '\033[106m'
    BG_BRIGHT_WHITE = '\033[107m'

    # Text Effects
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m' # May not work in all terminals
    REVERSE = '\033[7m' # Swap foreground and background
    HIDDEN = '\033[8m' # Hidden text (useful for passwords)
    STRIKETHROUGH = '\033[9m' # May not work in all terminals

    # Disable specific effects (or return to default)
    NO_BOLD_OR_DIM = '\033[22m'
    NO_ITALIC = '\033[23m'
    NO_UNDERLINE = '\033[24m'
    NO_BLINK = '\033[25m'
    NO_REVERSE = '\033[27m'
    NO_HIDDEN = '\033[28m'
    NO_STRIKETHROUGH = '\033[29m'

def format_text(text, *colors_and_effects):
    """
    Formats the given text with multiple ANSI color codes and effects.

    Args:
        text (str): The text to format.
        *colors_and_effects: One or more Color class attributes (e.g., Color.RED, Color.BOLD).

    Returns:
        str: The formatted string with ANSI escape codes and then reset.
    """
    applied_codes = "".join(colors_and_effects)
    return f"{applied_codes}{text}{Color.RESET}"

def pformat_text(text, *colors_and_effects, **kargs):
    """
    Prints the given text with the specified ANSI color codes and effects.
    Accepts multiple color/effect arguments.
    """
    applied_codes = "".join(colors_and_effects)
    print (f"{applied_codes}{text}{Color.RESET}", **kargs)

# --- End Color Class ---
