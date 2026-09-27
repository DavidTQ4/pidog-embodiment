# Fluffy's private phone game

The website is a PHP relay on Dave's Agent Tools' existing Lightsail instance.
The Windows desktop polls outbound over HTTPS, runs minimax, and publishes the
board. No inbound desktop port or public PiDog port is required.

## Website installation

The matching website changes are in the separate `readkeyboard` checkout at
`tmp/readkeyboard`, branch `codex/fluffy-tic-tac-toe`. Keep this checkout until
its changes have been committed/pushed; the parent PiDog repository ignores tmp.

Files: `fluffy-game.php`, `fluffy-game-api.php`, `fluffy-game.js`,
`fluffy-game.css`, `lib/fluffy-game.php`, and the two `tests/fluffy-*.php` tests.
The existing deployment workflow publishes its default branch to Lightsail.
Deploy these changes through that repository's normal review/merge process.

On Lightsail, add a separate `fluffy_game_secret` entry (at least 32 random
characters) to the existing `config.php` array. Do not replace the configuration
or reuse the existing PiDog control secret. The file is excluded from deployment.
For example, generate the value locally with:

```powershell
python -c "import secrets; print(secrets.token_hex(32))"
```

The phone uses the site's existing admin username/password. The game uses the
existing private storage directory outside the web root. PHP must be able to
write there, as it already does for the other private site tools. Use HTTPS.

Open `https://davesagenttools.com/fluffy-game.php` on the phone. It is unlinked,
omitted from the sitemap, and sends noindex/nofollow headers. Authentication is
the access control; the hidden URL is not a secret.

## Desktop

From the PiDog repository, with the existing virtual environment:

```powershell
$env:FLUFFY_GAME_SECRET = 'the same secret configured on Lightsail'
venv\Scripts\python.exe -m brain.fluffy_game_worker
```

This runs board-only. On the phone, tap **New game**, wait for the desktop to
connect, and choose a square. X always starts; Fluffy's O is unbeatable.

After testing the board, stop the worker with Ctrl+C. Close the vision/tracking
desktop controller and stop other robot-control sessions before enabling this
standalone worker's reactions. It cannot see another process's tracking-armed
flags. Run only one game worker for this robot.

```powershell
venv\Scripts\python.exe -m brain.fluffy_game_worker --robot-api http://127.0.0.1:18888
```

Replace that example URL with the bridge address/SSH tunnel you actually use.
If the bridge requires authentication, set `NOX_API_TOKEN` in this shell.
Reactions use fixed `think`, `nod`, and `wag_tail` gestures through
`move_if_idle`. Speech uses the bridge's `/speak` endpoint and requires working
Pi-side TTS. This initial worker does not use the GPU vision program's Windows
speech synthesis. No physical commands are sent unless `--robot-api` is supplied.

## Recovery and limits

- One private game at a time, expiring after two hours. Stop before replacing an
  unfinished game. The desktop reconstructs the game from accepted human moves.
- One pending move; revision checks reject double taps and duplicate responses.
  Pending moves expire after 30 seconds. Stop/start if one expires.
- A failed acknowledgement is retried through polling. Robot reactions happen
  only after a confirmed acknowledgement, so a lost response can skip a reaction
  rather than replay one. Website restarts preserve state on disk.
- **Stop game** prevents further turns and clears pending moves. It cannot
  cancel a gesture or speech already submitted to PiDog; use the robot's local
  stop control for immediate physical stopping.
- Game results remain playable even if a robot reaction fails. Read the worker
  console for connection/robot errors. No robot hardware is needed for tests.

## Tests

```powershell
venv\Scripts\python.exe -m pytest tests/test_fluffy_tictactoe.py tests/test_fluffy_game_worker.py
```

From the website checkout with PHP installed: `php tests/fluffy-game.php`.
Run `php tests/fluffy-http.php` for an isolated HTTP/authentication smoke test.
