import twophase.start_server as ss
from threading import Thread
bg = Thread(target=ss.start, args=(8080, 20, 2))
bg.start()