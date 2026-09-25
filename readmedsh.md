# PiDog desktop AI — startup and recovery guide

This guide describes the current PiDog V2 arrangement:

- Raspberry Pi 4: hardware control, camera capture, ultrasonic safety and offline voice recognition.
- Windows desktop: YOLO, pose, face identity, Qwen VLM and following decisions.
- SSH: encrypted access and local port forwarding without a static Pi address.

The normal post-reboot procedure is now one SSH command followed by starting the desktop application.

## Safety before startup

1. Put PiDog on the floor with clear space around it.
2. Check the battery and ensure cables cannot enter the legs or head mechanism.
3. Do not arm body following while PiDog is on a desk.
4. Keep the ultrasonic emergency-stop behaviour enabled.
5. Body following remains a supervised experimental function.

## System addresses

| Component | Pi address | Windows tunnel address |
| --- | --- | --- |
| Nox HTTP bridge | `127.0.0.1:8888` | `127.0.0.1:18888` |
| Vilib MJPEG stream | `127.0.0.1:9000` | `127.0.0.1:19000` |
| Nox body socket | `127.0.0.1:9999` | Not forwarded |

The Pi services are:

```text
nox-body.service
nox-bridge.service
nox-voice.service
```

## Expected files

### Raspberry Pi

```text
/home/david/pidog-embodiment/body/nox_daemon.py
/home/david/pidog-embodiment/body/nox_brain_bridge.py
/home/david/pidog-embodiment/body/nox_voice.py
/home/david/pidog-embodiment/body/nox-voice.env
/home/david/pidog-embodiment/.voice-venv/
/home/david/.local/share/vosk/vosk-model-small-en-us-0.15/
/usr/local/sbin/pidog-start
```

### Windows PC

```text
C:\Users\d_s_h\Documents\pidog-embodiment\pidog_yolo_vlm.py
C:\Users\d_s_h\Documents\pidog-embodiment\pidog_face_identity.py
C:\Users\d_s_h\Documents\pidog-embodiment\face_profiles\David.npz
C:\Users\d_s_h\Documents\pidog-embodiment\face_profiles\Joss.npz
C:\Users\d_s_h\Documents\pidog-embodiment\venv\Scripts\Activate.ps1
```

If the environment is named `.venv`, use `.venv` instead of `venv` below.

# One-time Pi configuration

These steps do not need to be repeated after every reboot.

## 1. Ensure the services start automatically

SSH into the Pi:

```powershell
ssh david@raspberrypi.local
```

On the Pi:

```bash
sudo systemctl daemon-reload
sudo systemctl enable nox-body.service nox-bridge.service nox-voice.service
```

Confirm the unit files exist:

```bash
systemctl cat nox-body.service
systemctl cat nox-bridge.service
systemctl cat nox-voice.service
```

## 2. Use a stable Voice HAT device name

The numeric ALSA card can change after reboot. For example, the Voice HAT has already moved from card 2 to card 3. Do not use `plughw:2,0`.

Edit:

```bash
nano /home/david/pidog-embodiment/body/nox-voice.env
```

The audio setting must be:

```text
NOX_AUDIO_DEVICE=plughw:CARD=sndrpigooglevoi,DEV=0
```

The model setting should be:

```text
NOX_VOSK_MODEL=/home/david/.local/share/vosk/vosk-model-small-en-us-0.15
```

Restart and check it:

```bash
sudo systemctl restart nox-voice.service
sleep 5
sudo journalctl -u nox-voice -n 20 --no-pager -l
```

Expected:

```text
[voice] Listening on plughw:CARD=sndrpigooglevoi,DEV=0
```

There should be no `audio open error` message.

## 3. Install the combined startup helper

Download `pidog-start` to the Windows Downloads folder. From PowerShell:

```powershell
scp "$env:USERPROFILE\Downloads\pidog-start" david@raspberrypi.local:/tmp/pidog-start
```

Then install it:

```powershell
ssh -t david@raspberrypi.local "sudo install -m 755 /tmp/pidog-start /usr/local/sbin/pidog-start"
```

The helper enables and starts all three services, waits for the body and bridge, starts voice recognition, warms the camera, and checks camera port 9000.

# Normal startup after a fresh reboot

## 1. Wait for the Pi

Switch PiDog on and allow approximately 60 seconds for the Pi to boot and join Wi-Fi.

## 2. Run the single SSH startup command

Open PowerShell and run this as one line:

```powershell
ssh -t -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -L 127.0.0.1:18888:127.0.0.1:8888 -L 127.0.0.1:19000:127.0.0.1:9000 david@raspberrypi.local "sudo /usr/local/sbin/pidog-start && exec sleep infinity"
```

Enter the Pi password. A successful startup ends with output similar to:

```text
[pidog-start] camera frame ready (83894 bytes)
[pidog-start] service state:
  nox-body.service       active
  nox-bridge.service     active
  nox-voice.service      active
  bridge :8888           ready
  camera :9000           ready
[pidog-start] PiDog is ready; keep the SSH window open for the desktop tunnels
```

Leave this PowerShell window open throughout the experiment. It is both the startup session and the two SSH tunnels.

Do not launch a second copy. If ports 18888 or 19000 are already occupied:

```powershell
Get-Process ssh
```

## 3. Start the desktop application

Open a second PowerShell window:

```powershell
Set-Location C:\Users\d_s_h\Documents\pidog-embodiment
.\venv\Scripts\Activate.ps1
python .\pidog_yolo_vlm.py
```

Add the usual command-line options if required, including the desired follow distance.

The camera has already been warmed by `pidog-start`, so the desktop should not need to trigger cold camera initialisation.

## 4. Test voice control

With the desktop application running and a recognised person visible, try commands separately:

```text
Nox select me
Nox arm head
Nox follow me
Nox stop
Nox lie down
```

`select me` requires the visible identity to resolve unambiguously. `stop` is also accepted without the wake word as a fail-safe.

The Pi journal should show `ACCEPTED`, and the desktop terminal should show `VOICE COMMAND`.

# Normal shutdown

1. Say `Nox stop`, press the desktop stop control, or disarm following.
2. Quit `pidog_yolo_vlm.py` with `Q` or `Esc`.
3. Wait for PiDog to settle.
4. Press `Ctrl+C` in the SSH/tunnel PowerShell window.

Closing the SSH tunnel does not stop the systemd services. To power down safely:

```powershell
ssh -t david@raspberrypi.local "sudo poweroff"
```

Wait for shutdown to complete before removing power.

# Quick health checks

## Bridge and camera from Windows

With the tunnel running:

```powershell
curl.exe --max-time 10 http://127.0.0.1:18888/status
curl.exe --max-time 20 http://127.0.0.1:18888/frame.jpg --output "$env:TEMP\pidog-frame.jpg"
Get-Item "$env:TEMP\pidog-frame.jpg"
```

The status should contain `{"ok": true}`. The JPEG should be substantially larger than a short JSON error response.

## Check all Pi services

```powershell
ssh -t david@raspberrypi.local "systemctl is-active nox-body nox-bridge nox-voice"
```

Expected:

```text
active
active
active
```

## Follow all relevant Pi logs

```powershell
ssh -t david@raspberrypi.local "sudo journalctl -u nox-body -u nox-bridge -u nox-voice -f -l"
```

# Troubleshooting

## Voice commands are not heard

```powershell
ssh -t david@raspberrypi.local "arecord -l; grep -E 'NOX_(AUDIO|MIC|DEVICE|VOSK)' /home/david/pidog-embodiment/body/nox-voice.env; sudo journalctl -u nox-voice -n 60 --no-pager -l"
```

If the journal says `arecord: audio open error: No such file or directory`, ensure the environment contains:

```text
NOX_AUDIO_DEVICE=plughw:CARD=sndrpigooglevoi,DEV=0
```

Then:

```bash
sudo systemctl restart nox-voice
sleep 5
sudo journalctl -u nox-voice -n 20 --no-pager -l
```

To watch recognition live:

```powershell
ssh -t david@raspberrypi.local "sudo journalctl -u nox-voice -f -l"
```

| Observation | Fault area |
| --- | --- |
| No recognised text appears | Microphone, ALSA or Vosk |
| Pi shows `Ignored` | Speech recognition did not match a command |
| Pi shows `ACCEPTED` but desktop shows nothing | Bridge/inbox or desktop polling |
| Desktop shows `VOICE COMMAND` but does nothing | Desktop command handling or safety state |

## Service says active but repeatedly crashes

`systemctl is-active` can catch a service briefly during an automatic restart:

```bash
systemctl show nox-voice -p NRestarts -p ActiveState -p SubState
sudo journalctl -u nox-voice -n 60 --no-pager -l
```

## Camera returns HTTP 503

```bash
sudo journalctl -u nox-body -u nox-bridge -n 100 --no-pager -l
sudo systemctl restart nox-body nox-bridge
```

Then run the normal combined startup command again.

## Port 19000 reports connection refused

The tunnel may be correct while the Pi camera server has not opened port 9000:

```bash
sudo ss -ltnp | grep -E ':(8888|9000|9999)\b'
```

Requesting `/frame.jpg` through port 8888 should initialise the camera and open port 9000. `pidog-start` performs this automatically.

## SSH connection drops during movement

This has previously been associated with power demand, particularly sudden standing or aggressive servo movement.

- Use the slow-stand sequence.
- Keep the SSH server-alive options in the normal startup command.
- Check power and battery condition.
- Inspect undervoltage history:

```bash
vcgencmd get_throttled
```

`throttled=0x0` means no current or historical throttling flags are recorded since boot.

## Body commands report `unknown command: arm_motion`

The Pi is running an older `nox_daemon.py`. Update `nox_daemon.py` and `nox_brain_bridge.py` together, then:

```bash
sudo systemctl restart nox-body nox-bridge nox-voice
```

## Host identification changed after reinstalling the Pi

Only after confirming the Pi was intentionally reinstalled:

```powershell
ssh-keygen -R raspberrypi.local
ssh david@raspberrypi.local
```

# Everyday command summary

Pi startup, camera warm-up and persistent tunnels:

```powershell
ssh -t -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -L 127.0.0.1:18888:127.0.0.1:8888 -L 127.0.0.1:19000:127.0.0.1:9000 david@raspberrypi.local "sudo /usr/local/sbin/pidog-start && exec sleep infinity"
```

Desktop application:

```powershell
Set-Location C:\Users\d_s_h\Documents\pidog-embodiment
.\venv\Scripts\Activate.ps1
python .\pidog_yolo_vlm.py
```
