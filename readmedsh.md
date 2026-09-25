# PiDog desktop AI: fresh-boot runbook

This runbook starts the complete PiDog V2 system after both the Raspberry Pi
and Windows PC have been rebooted.

The Raspberry Pi handles hardware, camera capture, ultrasonic safety, offline
voice recognition and immediate stop/lie commands. The Windows PC handles
YOLO, pose estimation, face identity, Qwen VLM inference and following logic.

## System layout

| Component | Address/port | Purpose |
| --- | --- | --- |
| `nox-body.service` | Pi `127.0.0.1:9999` | Servos, gait, camera, ultrasonic and safety latch |
| `nox-bridge.service` | Pi `127.0.0.1:8888` | HTTP API and voice-command inbox |
| Vilib camera | Pi `127.0.0.1:9000` | MJPEG camera stream; starts on first camera request |
| `nox-voice.service` | No network port | Microphone, Vosk and local voice safety commands |
| SSH forward | PC `127.0.0.1:18888` | PC to Pi HTTP API port 8888 |
| SSH forward | PC `127.0.0.1:19000` | PC to Pi camera port 9000 |

No static IP address is required. The Windows PC connects using
`raspberrypi.local`, and all robot ports remain bound to the Pi's loopback
interface.

## Files and expected locations

### Raspberry Pi

These files should be in `/home/david/pidog-embodiment/body`:

```text
nox_daemon.py
nox_brain_bridge.py
nox_voice.py
nox-voice.env
```

The service units should be installed as:

```text
/etc/systemd/system/nox-body.service
/etc/systemd/system/nox-bridge.service
/etc/systemd/system/nox-voice.service
```

The voice Python environment should exist at:

```text
/home/david/pidog-embodiment/.voice-venv
```

### Windows PC

The project directory is expected to be:

```text
C:\Users\d_s_h\Documents\pidog-embodiment
```

Important contents include:

```text
pidog_yolo_vlm.py
pidog_face_identity.py
face_profiles\David.npz
face_profiles\Joss.npz
venv\Scripts\Activate.ps1
```

If the environment directory is called `.venv` rather than `venv`, substitute
`.venv` in the activation command below.

---

# Normal startup after a fresh boot

## 1. Prepare PiDog physically

1. Put PiDog on the floor with clear space around it.
2. Check that the battery is adequately charged.
3. Check that no cables can enter the legs or camera mechanism.
4. Switch PiDog on and allow approximately 60 seconds for the Pi to boot and
   join Wi-Fi.
5. Do not arm body following while PiDog is on a desk.

## 2. Connect to the Pi

Open PowerShell on Windows:

```powershell
ssh david@raspberrypi.local
```

If Windows reports that the host key changed after reinstalling the Pi, remove
only the old key for this hostname and reconnect:

```powershell
ssh-keygen -R raspberrypi.local
ssh david@raspberrypi.local
```

Confirm the three services started automatically:

```bash
systemctl is-active nox-body nox-bridge nox-voice
```

Expected result:

```text
active
active
active
```

If any service is inactive, start all three in dependency order:

```bash
sudo systemctl restart nox-body
sudo systemctl restart nox-bridge
sudo systemctl restart nox-voice
```

Inspect detailed status:

```bash
sudo systemctl status nox-body nox-bridge nox-voice --no-pager -l
```

Check the initially available TCP listeners:

```bash
sudo ss -ltnp | grep -E ':(8888|9000|9999)\b'
```

Ports `8888` and `9999` should be present. Port `9000` may not appear until
the camera has been warmed in step 4.

Leave the services running, then exit this ordinary SSH session if desired:

```bash
exit
```

## 3. Open the persistent SSH tunnels

Open a dedicated PowerShell window and run this as one line:

```powershell
ssh -N -T -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -L 127.0.0.1:18888:127.0.0.1:8888 -L 127.0.0.1:19000:127.0.0.1:9000 david@raspberrypi.local
```

Enter the Pi password. A successful tunnel normally displays no further
output and keeps the PowerShell window occupied. Leave this window open for
the whole experiment.

Do not launch a second copy of the same tunnel. If PowerShell reports that a
local address is already in use, find the existing SSH process first:

```powershell
Get-Process ssh
```

## 4. Verify the API and warm the camera

Open a second PowerShell window. Check the HTTP bridge through the tunnel:

```powershell
curl.exe --max-time 10 http://127.0.0.1:18888/status
```

The response should contain:

```json
{"ok": true}
```

Trigger the first camera capture. This also starts the persistent MJPEG server
on Pi port 9000:

```powershell
curl.exe --max-time 25 http://127.0.0.1:18888/frame.jpg --output "$env:TEMP\pidog-warmup.jpg"
```

Confirm that a non-empty image was created:

```powershell
Get-Item "$env:TEMP\pidog-warmup.jpg" | Select-Object Name,Length
```

Optionally open it:

```powershell
Start-Process "$env:TEMP\pidog-warmup.jpg"
```

If `/frame.jpg` returns HTTP 503, use the camera troubleshooting section
before starting the desktop tracker.

## 5. Verify the Pi voice service

Say:

```text
Nox stop
```

Bare `stop` or `halt` also works by design. It clears queued locomotion,
stabilises PiDog in a standing posture and closes the Pi-side locomotion latch.

To watch voice recognition live, open another SSH session:

```powershell
ssh david@raspberrypi.local
```

Then:

```bash
sudo journalctl -u nox-voice -u nox-body -f
```

Expected messages include:

```text
[voice] ACCEPTED stop
[nox] HALT requested
```

Press `Ctrl+C` to leave the log view; this does not stop the services.

## 6. Activate the desktop environment

In the second PowerShell window:

```powershell
Set-Location C:\Users\d_s_h\Documents\pidog-embodiment
.\venv\Scripts\Activate.ps1
```

If the environment is named `.venv`:

```powershell
.\.venv\Scripts\Activate.ps1
```

Confirm that CUDA is visible:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Confirm that the identity profiles exist:

```powershell
Get-ChildItem .\face_profiles\*.npz
```

## 7. Start YOLO, face recognition and the VLM

The current normal launch, retaining the requested 30 cm following distance,
is:

```powershell
python .\pidog_yolo_vlm.py --follow-distance 30 --max-new-tokens 160
```

The application should:

1. Load `yolo11n.pt`.
2. Load `yolo11n-pose.pt`.
3. Load the David and Joss face profiles.
4. Load `Qwen/Qwen3-VL-4B-Instruct`.
5. Connect to the warmed MJPEG stream.
6. Centre the head once automatically.
7. Leave head tracking and body following disarmed.
8. Start polling the voice-command inbox.

Expected terminal messages include:

```text
Camera stream ready
Head centred automatically; tracking and body following remain disarmed
Voice commands enabled: select me | arm head | follow me | stop | lie down
```

## 8. Begin tracking by voice

Stand alone in view for the first test and speak the commands separately,
allowing tracking to stabilise between them:

```text
Nox select me
Nox arm head
Nox follow me
```

The intended sequence is:

1. `select me` selects the only recognised visible identity and leaves all
   movement disarmed.
2. `arm head` enables bounded head tracking.
3. `follow me` performs the slow stand, explicitly opens the Pi locomotion
   latch, and then enables ultrasonic-protected walking and turning.

If both David and Joss are recognised in the frame, `select me` deliberately
refuses to guess which person spoke. Use keyboard `1` or `2`, or move to a
single-person view. Biometric speaker verification is not part of this first
voice version.

## 9. Stop safely

At any time say either:

```text
Nox stop
```

or simply:

```text
stop
```

This happens locally on the Pi and does not depend on the Windows application
or SSH tunnel. It:

1. Clears queued locomotion frames.
2. Queues a controlled stabilising stand.
3. Latches further locomotion off on the Pi.
4. Relays the command to Windows so head and body following are disarmed.

To stop and lie down:

```text
Nox stop and lie down
```

or:

```text
Nox lie down
```

## 10. End the experiment

1. Say `Nox stop and lie down`.
2. Confirm in the desktop terminal that following is disarmed.
3. Press `Q` or `Esc` in the PiDog vision window.
4. Wait for `PiDog YOLO + VLM person tracker stopped`.
5. Return to the SSH tunnel PowerShell window and press `Ctrl+C`.

The three Pi services can remain running. If PiDog itself is being switched
off, shut the Pi down cleanly first:

```powershell
ssh -t david@raspberrypi.local "sudo poweroff"
```

Wait for shutdown before switching off the battery supply.

---

# Keyboard fallback controls

| Key | Function |
| --- | --- |
| `1`–`9` | Select visible person from left to right |
| `0` | Clear selection and disarm tracking/following |
| `C` | Centre head manually |
| `M` | Toggle head tracking |
| `T` | Toggle identity-locked body following |
| `A` | Toggle fallback vertical aiming point |
| `H` | Toggle facial-keypoint head aiming |
| `V` or `Space` | Run one VLM scene analysis |
| `Y` | Toggle YOLO processing |
| `Q` or `Esc` | Quit the desktop program |

Keyboard `0` disarms the desktop controller but is not a substitute for the
Pi-local spoken `stop` when the robot is currently walking.

---

# Emergency PowerShell halt

If speech recognition is unavailable but the HTTP tunnel still works, send a
local halt through the bridge:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:18888/command `
  -ContentType application/json `
  -Body '{"cmd":"halt","speed":40}'
```

For the existing hard emergency action, which clears all action buffers and
lies down rapidly:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:18888/emergency_stop `
  -ContentType application/json `
  -Body '{}'
```

Use the hard emergency action only when its faster lie-down is preferable to a
controlled halt.

---

# Troubleshooting

## SSH cannot resolve `raspberrypi.local`

Try the last known address only as a temporary diagnostic:

```powershell
ping raspberrypi.local
arp -a
```

Do not expose ports 8888, 9000 or 9999 to the wider network. The SSH tunnel is
the intended connection method.

## Tunnel closes or repeatedly reconnects

Confirm only one tunnel process is using each local port:

```powershell
Get-NetTCPConnection -LocalPort 18888,19000 -ErrorAction SilentlyContinue
Get-Process ssh
```

Restart the dedicated tunnel using the full keepalive command in step 3.

## API port 18888 refuses connections

On the Pi:

```bash
sudo systemctl restart nox-body nox-bridge
sudo systemctl status nox-body nox-bridge --no-pager -l
sudo ss -ltnp | grep -E ':(8888|9999)\b'
```

Then recreate the SSH tunnel.

## Camera stream port 19000 refuses connections

First warm the camera through port 18888:

```powershell
curl.exe --max-time 25 http://127.0.0.1:18888/frame.jpg --output "$env:TEMP\pidog-warmup.jpg"
```

Then check the Pi:

```bash
sudo ss -ltnp | grep -E ':(9000)\b'
sudo journalctl -u nox-body -n 80 --no-pager -l
```

If the camera still fails:

```bash
sudo systemctl restart nox-body
```

Warm `/frame.jpg` again before restarting the desktop program.

## `/frame.jpg` returns `camera not available`

Inspect the body log:

```bash
sudo journalctl -u nox-body -n 100 --no-pager -l
```

Also verify the camera outside the Nox service if necessary:

```bash
rpicam-hello --list-cameras
```

## Voice service is not recognising commands

Check the service and microphone:

```bash
sudo systemctl status nox-voice --no-pager -l
sudo journalctl -u nox-voice -n 100 --no-pager -l
arecord -l
```

Confirm the configured Vosk model exists:

```bash
grep NOX_VOSK_MODEL /home/david/pidog-embodiment/body/nox-voice.env
find /home/david -maxdepth 6 -type d -name 'vosk-model*' -print
```

Test the microphone directly:

```bash
arecord -D plughw:2,0 -f S16_LE -r 16000 -c 1 -d 5 /tmp/pidog-mic-test.wav
sox /tmp/pidog-mic-test.wav -n stat
aplay /tmp/pidog-mic-test.wav
```

If recording is clear but recognition is consistently quiet, change
`NOX_VOICE_GAIN=1.0` to `2.0` in `nox-voice.env`, then:

```bash
sudo systemctl restart nox-voice
```

## Voice command is heard but desktop does nothing

Confirm the desktop terminal says voice commands are enabled. Then check the
inbox manually:

```powershell
curl.exe http://127.0.0.1:18888/voice/inbox
```

This endpoint drains pending messages, so use it only for diagnosis while the
desktop tracker is stopped.

## Face recognition is disabled

On Windows:

```powershell
Get-ChildItem C:\Users\d_s_h\Documents\pidog-embodiment\face_profiles\*.npz
```

The files must be inside the `face_profiles` directory from which
`pidog_yolo_vlm.py` is launched.

## PyTorch cannot see CUDA

Inside the activated Windows environment:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

If `False` is returned, the environment contains a CPU-only PyTorch build or
an incompatible CUDA build. Repair that environment before launching the VLM.

## PiDog moves after the desktop application closes

Say `stop`, or use the PowerShell halt request above. Inspect whether the Pi
accepted the halt and cleared buffered frames:

```bash
sudo journalctl -u nox-body -n 80 --no-pager -l
```

The updated daemon's halt latch rejects new forward/turn commands until body
following explicitly rearms motion.

## Check everything at once

Pi:

```bash
sudo systemctl status nox-body nox-bridge nox-voice --no-pager -l
sudo ss -ltnp | grep -E ':(8888|9000|9999)\b'
sudo journalctl -u nox-body -u nox-bridge -u nox-voice -n 120 --no-pager -l
vcgencmd get_throttled
```

Windows:

```powershell
curl.exe --max-time 10 http://127.0.0.1:18888/status
Get-NetTCPConnection -LocalPort 18888,19000 -ErrorAction SilentlyContinue
python -c "import torch; print(torch.cuda.is_available())"
```

`vcgencmd get_throttled` should normally report `throttled=0x0`.

---

# One-time voice installation reference

This section is not required after every reboot. Use it only if the voice
environment or service has not yet been installed.

```bash
cd /home/david/pidog-embodiment
python3 -m venv --system-site-packages .voice-venv
.voice-venv/bin/python -m pip install --upgrade pip
.voice-venv/bin/python -m pip install vosk
```

Find an existing Vosk model:

```bash
find /home/david -maxdepth 6 -type d -name 'vosk-model*' -print
```

If none exists:

```bash
sudo apt install -y wget unzip
mkdir -p /home/david/.local/share/vosk
cd /home/david/.local/share/vosk
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
rm vosk-model-small-en-us-0.15.zip
```

Configure and install the service:

```bash
cd /home/david/pidog-embodiment/body
cp nox-voice.env.example nox-voice.env
nano nox-voice.env

sudo cp nox-voice.service /etc/systemd/system/nox-voice.service
sudo systemctl daemon-reload
sudo systemctl enable --now nox-body nox-bridge nox-voice
```

Set `NOX_VOSK_MODEL` in `nox-voice.env` if automatic discovery does not find
the model directory.
