import curses

def main(stdscr):
    curses.curs_set(0)  # Hide the cursor
    curses.start_color()
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)
    curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)

    # Get screen dimensions
    height, width = stdscr.getmaxyx()

    # Define window dimensions
    sidebar_width = width // 4
    sidebar = curses.newwin(height, sidebar_width, 0, 0)  # Left sidebar
    main_content = curses.newwin(height, width - sidebar_width, 0, sidebar_width)  # Right area

    # Render sidebar
    sidebar.addstr(0, 0, "Menu", curses.color_pair(1) | curses.A_BOLD)
    menu_items = ["Home", "Settings", "Exit"]
    for i, item in enumerate(menu_items):
        sidebar.addstr(i + 2, 1, item, curses.color_pair(2 if i == 0 else 0))

    # Main content
    main_content.addstr(0, 0, "Welcome to the main content area!", curses.color_pair(1))
    main_content.refresh()

    while True:
        sidebar.refresh()
        key = sidebar.getch()
        if key == ord('q'):
            break

if __name__ == "__main__":
    curses.wrapper(main)

