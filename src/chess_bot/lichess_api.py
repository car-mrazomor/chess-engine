import requests
import json
import threading
import os
from dotenv import load_dotenv
from datetime import datetime
from src.chess_bot.db_client import db_client
from src.chess_bot.redis_client import redis_client
from src.chess_bot.__chess_engine import check_turn, _get_move, _first_to_play


load_dotenv()

username = os.environ.get("USERNAME")
api_token = os.environ.get("API_TOKEN")


def _url(path: str) -> str:
    return f"https://lichess.org{path}"

def _request(method: str, path: str, stream: bool = False, data: dict = None, headers: dict = None) -> requests.Response:
    if not headers:
        headers: dict = {"Authorization": f"Bearer {api_token}"}
    response = requests.request(method, _url(path), headers=headers, stream=stream, data=data)
    if response.status_code == 200:
        return response
    error = response.json().get("error", None)
    if response.status_code == 400:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{now} Error occured on {response} '{response.json()}'")
    if error and error == "This game cannot be played with the Board API.":
        return "not_board"
    return False


def start_event(json_data: dict):
    _type = json_data["type"]
    gameId = json_data[_type if _type == "challenge" else "game"]["id"]
    threading.Thread(target=_start_game, args=[gameId, _type]).start()

def finish_event(json_data: dict):
    gameId = json_data["game"]["id"]
    status = json_data["game"]["status"]["name"]
    winner = json_data["game"].get("winner", None)
    fen = json_data["game"]["fen"]
    my_color = json_data["game"]["color"]

    db_client(
        "INSERT INTO games (gameId, botColor, winner, fen, status) VALUES (%s, %s, %s, %s, %s)",
        (gameId, my_color, winner if winner else "none", fen, status)
    )

    if gameId in redis_client(f"sessionId:{gameId}", method="get"):
        redis_client(key="current_game", method="delete")
        redis_client(key=f"sessionId:{gameId}", method="delete")


FUNCS:dict = {
    "gameStart":start_event,
    "challenge":start_event,
    "gameFinish":finish_event
}


def check_events() -> None:
    resp = _request("GET", "/api/stream/event", stream=True)
    if not resp:
        return
    
    for line in resp.iter_lines():
        try:
            json_data = json.loads(line.decode("utf-8"))
            _type = json_data["type"]
            if _type in ["challenge", "gameStart", "gameFinish"]:
                FUNCS[_type](json_data)
        except Exception as e:
            pass


def _start_game(gameId, _type: str = None):
    if _type == "challenge":
        accept_challenge(gameId)
    if _type == "gameStart":
        redis_client(key=f"sessionId:{gameId}", value=gameId, method="set")
    check_game_state(gameId)


def accept_challenge(gameId: str) -> dict:
    resp = _request("POST", f"/api/challenge/{gameId}/accept")
    if not resp:
        return False
    return resp.json()

def make_move(gameId: str, move: str) -> dict:
    resp = _request("POST", f"/api/board/game/{gameId}/move/{move}")
    if not resp:
        return False
    return resp.json()

def abort_game(gameId: str) -> dict:
    resp = _request("POST", f"/api/board/game/{gameId}/abort")
    return resp.json()

def check_game_state(gameId) -> None:
    resp = _request("GET", f"/api/board/game/stream/{gameId}", stream=True)
    if resp == "not_board":
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{now} Game can not be played gameId '{gameId}'")
        return
    turn: str = "white"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{now} Game started gameId '{gameId}'")
    for line in resp.iter_lines():
        try:
            json_data = json.loads(line.decode("utf-8"))
            initialFen = json_data.get("initialFen", None)
            white = json_data.get("white", None)

            if initialFen == "startpos" and white:
                if white["id"] == "montymoomoo":
                    move = _first_to_play()
                    make_move(gameId, move)
                    continue
                else:
                    turn = "black"
            
            game_status = json_data.get("status", None)
            if game_status and game_status == "started":
                moves = json_data["moves"]
                my_turn = check_turn(moves, turn)
                if my_turn:
                    move = _get_move(moves=moves)
                    make_move(gameId, move)
                    continue

            if game_status and game_status != "started":
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"{now} Game done gameId '{gameId}'")
                return
        except Exception as e:
            pass


def create_seek():
    data:dict = {
        "rated":"true",
        "variant":"standard",
        "ratingRange":"",
        "time":"10",
        "increment":"0",
        "color":"random"
    }
    headers:dict = {
        "Content-Type":"application/x-www-form-urlencoded",
        "Authorization":f"Bearer {api_token}"
    }
    resp = _request("POST", f"/api/board/seek", data=data, headers=headers)
    if resp:
        return resp
    return False
