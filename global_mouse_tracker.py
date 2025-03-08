from pynput import mouse
from pythonosc import udp_client

# Set your TouchDesigner IP and port (usually 127.0.0.1 and a port you choose, e.g., 8000)
OSC_IP = "127.0.0.1"
OSC_PORT = 10000

client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)

# These global variables can help if you want to accumulate scroll data
scroll_value = 0

def on_click(x, y, button, pressed):
    # Only process left-click events
    if button == mouse.Button.left:
        # Send 1 for press and 0 for release
        client.send_message("/globalLeft", 1 if pressed else 0)
        print("Left click:", "Pressed" if pressed else "Released")

def on_scroll(x, y, dx, dy):
    # You can accumulate or send dy directly; here we send dy as the scroll delta.
    client.send_message("/globalScroll", dy)
    print("Scrolled:", dy)

# Optional: Track mouse movement if needed
def on_move(x, y):
    # For example, you might want to send position as well
    # client.send_message("/globalPos", [x, y])
    pass

# Start the global mouse listener
with mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll) as listener:
    listener.join()
