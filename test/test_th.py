from pynput import keyboard
def on_press(key):
    if key == keyboard.Key.esc:
        return False  # stop listener
    try:
        k = key.char  # single-char keys
        print('aaaaaa'+k)
    except:
        k = key.name  # other keys
        
def main():
    while True:
        cmd = input('waiting for cmd: ')
        print(cmd)
        if cmd == 's':
            print('start for keyboard ctrl')
            with keyboard.Listener(on_press=on_press) as listener:
            #listener.start()
                listener.join()
        if cmd == 'x':
            print("finished!")
            break

main()