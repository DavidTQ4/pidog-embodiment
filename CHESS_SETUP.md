# Fluffy chess setup

Fluffy's chess rules and legal-move handling run on the desktop. Stockfish
chooses Fluffy's moves; the Raspberry Pi only performs speech and body actions.

## Windows desktop

From the `pidog-embodiment` folder with the normal virtual environment active:

```powershell
python -m pip install python-chess
winget install --id Stockfish.Stockfish --exact
```

Open a new PowerShell window and confirm that Stockfish is visible:

```powershell
stockfish
```

Type `quit` to exit it. If `stockfish` is not on `PATH`, point Fluffy at the
downloaded executable before launching the controller:

```powershell
$env:STOCKFISH_PATH = "C:\path\to\stockfish-windows-x86-64-avx2.exe"
python .\pidog_yolo_vlm.py
```

To persist that path for future PowerShell windows:

```powershell
[Environment]::SetEnvironmentVariable(
  "STOCKFISH_PATH",
  "C:\path\to\stockfish-windows-x86-64-avx2.exe",
  "User"
)
```

## Playing

1. Start the normal Pi services and desktop controller.
2. Open `https://davesagenttools.com/fluffy.php` on the phone.
3. Choose Puppy, Friendly, Clever, or Strong and press **Play chess**.
4. Wait for Fluffy to become idle. The human always plays White.
5. Tap a piece and then a highlighted legal destination.

The phone remains a display and input device. The desktop validates every move,
owns the authoritative board, and runs Stockfish. Only one phone/player lease is
accepted at a time, as with tic-tac-toe.

