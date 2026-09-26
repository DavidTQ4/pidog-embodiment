# PiDog offline voice commands

This adds offline command recognition to the PiDog V2 microphone. The
Raspberry Pi captures 16 kHz audio and runs a small Vosk command grammar. The
desktop continues to run YOLO, face identity and movement decisions.

## Behaviour

| Spoken phrase | Result |
| --- | --- |
| `Fluffy, select me` | Selects the only recognised visible person. It refuses if none or multiple recognised people are visible. |
| `Fluffy, arm head` | Arms head tracking for the selected person. |
| `Fluffy, follow me` | Arms identity-locked body following if the selected identity and head tracking are ready. |
| `Fluffy, stop` | Locally clears queued walking frames, stabilises in `stand`, and disarms desktop following. |
| `Fluffy, lie down` | Locally clears queued walking frames, lies down at controlled speed, and disarms desktop following. |
| `Fluffy, stop and lie down` | Same safe controlled lie-down path. |
| `Fluffy, sit` / `stand up` | Changes posture locally when the legs are idle. |
| `Fluffy, paw` | Runs the SDK's `hand_shake` action. |
| `Fluffy, high five` | Runs `high_five` locally. |
| `Fluffy, bark` / `howl` / `pant` | Runs the corresponding local preset. |
| `Fluffy, wag your tail` | Runs `wag_tail` locally. |
| `Fluffy, stretch` / `scratch` | Runs the corresponding local trick. |
| `Fluffy, nod` / `shake your head` | Runs the corresponding local head action. |
| `Fluffy, go to sleep` | Runs `doze_off` when the legs are idle. |

All commands require the `Fluffy` wake word by default, including `stop`. This
reduces false emergency stops caused by ambient speech. Keyboard controls and
the ultrasonic emergency halt remain available.

This first version does **speech recognition**, not biometric speaker
verification. `select me` therefore succeeds only when exactly one recognised
face is visible. It will not guess between David and Joss.

## Files

Pi files in `/home/david/pidog-embodiment/body`:

- `nox_daemon.py`
- `nox_brain_bridge.py`
- `nox_voice.py`
- `nox-voice.env`

Desktop file:

- `pidog_yolo_vlm.py`

System service:

- `/etc/systemd/system/nox-voice.service`

## Pi installation

After copying the updated Pi files into the `body` directory, create an
isolated environment for Vosk:

```bash
cd /home/david/pidog-embodiment
python3 -m venv --system-site-packages .voice-venv
.voice-venv/bin/python -m pip install --upgrade pip
.voice-venv/bin/python -m pip install vosk
```

Locate the model used by SunFounder's working example:

```bash
find /home/david -maxdepth 6 -type d -name 'vosk-model*' -print
```

If that prints nothing, install the standard small English model:

```bash
sudo apt install -y wget unzip
mkdir -p /home/david/.local/share/vosk
cd /home/david/.local/share/vosk
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
rm vosk-model-small-en-us-0.15.zip
```

Copy the environment template and, if necessary, set `NOX_VOSK_MODEL` to the
reported model directory:

```bash
cd /home/david/pidog-embodiment/body
cp nox-voice.env.example nox-voice.env
nano nox-voice.env
```

Install and start the service:

```bash
sudo cp /home/david/pidog-embodiment/body/nox-voice.service \
  /etc/systemd/system/nox-voice.service

sudo systemctl daemon-reload
sudo systemctl enable --now nox-body nox-bridge nox-voice
```

Check all three services:

```bash
sudo systemctl status nox-body nox-bridge nox-voice --no-pager -l
sudo journalctl -u nox-voice -f
```

Expected startup includes:

```text
[voice] Loading Vosk model: ...
[voice] Listening on plughw:2,0; say 'Fluffy' followed by a command
```

## Test local safety before walking

Keep PiDog on the floor with clearance. First test while it is stationary:

```text
Fluffy stop
Fluffy lie down
```

The Pi log should show `ACCEPTED stop` or `ACCEPTED lie_down`, while the body
daemon log should show `HALT requested` or `LIE DOWN requested`:

```bash
sudo journalctl -u nox-voice -u nox-body -f
```

The `stop` path does not require the desktop or SSH tunnel. If the HTTP bridge
is unavailable, `nox_voice.py` falls back to the daemon's local TCP port 9999.

## Desktop test

Start the usual SSH forwards, then run the updated tracker normally. Its
startup should include:

```text
Voice commands enabled: select me | arm head | follow me | stop | lie down
```

Stand alone in view and speak the commands in order:

```text
Fluffy select me
Fluffy arm head
Fluffy follow me
```

The terminal prints every accepted command and the reason for any refusal.
`select me` deliberately refuses when both David and Joss are recognised; true
speaker verification can be added later without weakening this safety rule.

To temporarily disable desktop voice control:

```powershell
python pidog_yolo_vlm.py --disable-voice-commands
```

## Tuning

The microphone already records at useful distance, so begin with gain `1.0`.
If recognition is consistently too quiet, edit `nox-voice.env` and try:

```text
NOX_VOICE_GAIN=2.0
```

Then restart only the voice service:

```bash
sudo systemctl restart nox-voice
```

Avoid excessive gain because clipping and amplified servo noise reduce speech
recognition accuracy.
