from src.chess_bot.lichess_api import check_events, create_seek
from src.chess_bot.redis_client import redis_client
import threading
import argparse
import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Chess Bot',
                    description='',
                    epilog='Use --help to see all commands')

    parser.add_argument('-s', '--seek', action='store_true')
    args = parser.parse_args()

    threading.Thread(target=check_events).start()

    while True:
        try:
            if not redis_client(key="current_game", method="get"):
                resp = create_seek()
                if resp:
                    redis_client(key="current_game", value="True", method="set")

        except KeyboardInterrupt:
            exit(-1)
