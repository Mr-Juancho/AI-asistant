# jarvis/console.py
import threading
RESET="\x1b[0m"; BOLD="\x1b[1m"; DIM="\x1b[2m"
RED="\x1b[31m"; BLUE="\x1b[34m"; GRAY="\x1b[90m"
print_lock = threading.Lock()

def c(text, color): return f"{color}{text}{RESET}"
def tag(role, color, dim=False):
    style = (DIM if dim else BOLD)
    return f"{style}{color}[{role}]{RESET}"

def clear_partial_line(width=140):
    with print_lock:
        print("\r"+" "*width+"\r", end="", flush=True)

def sprint(msg="", end="\n"):
    with print_lock: print(msg, end=end, flush=True)

def render_user_partial(state, text, width=120):
    if state.suppress_user_partials or state.assistant_printing:
        return
    with print_lock:
        lbl = tag("TÚ", RED)
        print(f"\r{lbl}: {text:<{width}}", end="", flush=True)
