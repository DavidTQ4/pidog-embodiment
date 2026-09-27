# Fluffy chase target

Say **"Fluffy chase target"** with one red object clearly visible in the camera.
The desktop stands Fluffy, centres the head and starts red-colour tracking.
This direct voice command works with either conversation backend.

The controller adapts the supplied SunFounder `13_ball_track.py` behaviour:
red detection, bounded head tracking, left/right turns and forward walking at
speed 98. It uses the existing camera stream and Pi motion API, so there is no
second camera process or `Pidog()` instance. Person following is disarmed when
chasing begins and does not automatically resume afterwards.

Say **"Fluffy stop"** to stop. Pi-side voice stop remains local and latches
locomotion off. Any desktop key also cancels chasing before its usual function.
Chasing cancels if the red target disappears, multiple similarly sized targets
are visible, the target reaches 300 pixels wide at normalized 640x480 resolution,
the camera stops producing frames for one second, or a head/clearance/connection
check fails. Start it again explicitly after cancellation.

Each gait is one step, admitted only when the legs are idle and the existing
ultrasonic clearance check passes. The usual `--follow-distance`,
`--turn-clearance` and `--turn-interval` arguments apply. Three visible frames
are required before a gait. Colour detection can follow any red object, not just
a ball, so use one clear target and supervise the first physical test.

Games and active LLM replies block starting a chase. Conversation and other
tracking commands cancel an active chase. The desktop displays the chase state.

After these changes are pushed, update the Pi voice service:

```bash
cd /home/david/pidog-embodiment
git switch codex/fluffy-dance
git pull --ff-only
sudo systemctl restart nox-voice
sudo systemctl status nox-voice --no-pager -l
```

Update and restart the desktop brain using your usual command. No additional
packages are required. Do not launch `13_ball_track.py` alongside Fluffy.
