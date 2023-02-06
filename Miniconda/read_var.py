import os
import time

if __name__ == "__main__":
    var_name = "UPDATE_P"
    print("reading...")
    while True:
        # read global variable
        var = os.getenv(var_name)
        print(time.time(),":",var_name,":",var)
        # sleep
        time.sleep(0.1)
        # Clearing the Screen
        os.system('clear')
        