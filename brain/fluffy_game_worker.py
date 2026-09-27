"""Outbound desktop worker for Dave's Agent Tools' private tic-tac-toe page.

Run from the repository root: python -m brain.fluffy_game_worker
Board-only by default. --robot-api explicitly enables fixed robot reactions.
"""

import argparse
import os
import time
from urllib.parse import urlsplit

import requests

from brain.fluffy_tictactoe import Game
from fluffy_action_broker import FluffyActionBroker


def answer(state):
    """Rebuild locally from human moves, verifying the relay's mirrored state."""
    if not state or state.get('stopped') or state['expires'] <= time.time():
        return None
    pending = state.get('pending')
    if not pending or time.time() - pending['created'] > 30:
        return None
    moves = state['moves']
    if not isinstance(moves, list) or len(moves) > 5:
        raise ValueError('Invalid game history')
    game = Game()
    for square in moves:
        game.play(square, game.revision)
    if game.revision != state['revision'] or list(game.board) != state['board']:
        raise ValueError('Relay history does not match local game')
    snapshot, reaction = game.play(pending['square'], game.revision)
    return {'op': 'ack', 'id': state['id'], 'revision': state['revision'],
            'board': snapshot['board'], 'outcome': snapshot['outcome'],
            'message': reaction.speech}, reaction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', default='https://davesagenttools.com')
    parser.add_argument('--robot-api', help='Enable reactions using this local/Tailscale bridge')
    parser.add_argument('--allow-local-http', action='store_true', help='Local relay tests only')
    args = parser.parse_args()
    url = urlsplit(args.site)
    if url.username or url.password or url.query or url.fragment or not url.hostname:
        parser.error('site must be a plain site URL')
    if url.scheme != 'https' and not (args.allow_local_http and url.scheme == 'http'
            and url.hostname in ('localhost', '127.0.0.1', '::1')):
        parser.error('site requires HTTPS (except explicit loopback tests)')
    secret = os.environ.get('FLUFFY_GAME_SECRET', '')
    if len(secret) < 32:
        parser.error('set FLUFFY_GAME_SECRET to the secret configured on Lightsail (32+ characters)')
    relay = requests.Session()
    relay.headers['X-Fluffy-Secret'] = secret
    robot = requests.Session()
    if os.environ.get('NOX_API_TOKEN'):
        robot.headers['Authorization'] = 'Bearer ' + os.environ['NOX_API_TOKEN']
    broker = FluffyActionBroker(args.robot_api, session=robot) if args.robot_api else None

    def post(payload):
        response = relay.post(args.site.rstrip('/') + '/fluffy-game-api.php',
                              json=payload, timeout=(3, 8), allow_redirects=False)
        if response.is_redirect:
            raise ValueError('Use the canonical HTTPS site URL; redirects are disabled')
        response.raise_for_status()
        return response.json()

    print('Fluffy game worker running; ' + ('robot reactions enabled.' if broker else 'board-only mode.'), flush=True)
    try:
        while True:
            try:
                state = post({'op': 'poll'})['state']
                result = answer(state)
                if result:
                    payload, reaction = result
                    acknowledged = post(payload)
                    print(reaction.speech, flush=True)
                    # Ack before effects. A lost ack response suppresses effects,
                    # rather than replaying physical actions on a network retry.
                    if broker:
                        fresh = post({'op': 'poll'})['state']
                        if (fresh['id'] == state['id'] and not fresh['stopped']
                                and fresh['expires'] > time.time()
                                and fresh['revision'] == acknowledged['game']['revision']):
                            broker.execute(reaction.action, {})
                            response = robot.post(args.robot_api.rstrip('/') + '/speak',
                                                  json={'text': reaction.speech}, timeout=(1, 8))
                            response.raise_for_status()
            except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
                print(f'Waiting/retrying: {type(exc).__name__}: {exc}', flush=True)
                time.sleep(3)
            time.sleep(1)
    except KeyboardInterrupt:
        print('Worker stopped.')
    finally:
        relay.close()
        robot.close()


if __name__ == '__main__':
    main()
