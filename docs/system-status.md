# On-demand Pi health for either conversation backend

Both Ollama/Qwen and OpenAI can call `get_system_status` during conversation.
Ask "How is your battery?", "What is your CPU temperature?", "How busy is your
CPU?", or "How much memory and disk space do you have?"

The readings describe the Raspberry Pi body, not the Windows desktop:

- Battery voltage, estimated percentage, and whether servo power is missing.
- CPU utilization sampled over 0.2 seconds, and CPU temperature in Celsius.
- Total and available RAM, RAM utilization, free disk space and disk utilization.
- Pi uptime, available robot diagnostics and behavior state.

Battery percentage is a rough voltage estimate. Charging state is unknown;
voltage alone does not establish charging. Failed readings are explicitly
unavailable rather than fabricated as zero. The tool uses a fixed read-only
`GET /system/status` endpoint and caches its result within each conversation
turn. It cannot execute shell commands or modify the system.

No extra Python packages are needed. Once these changes have been pushed, update
the Pi checkout and restart the bridge (with Fluffy stationary):

```bash
cd /home/david/pidog-embodiment
git switch codex/fluffy-dance
git pull --ff-only
sudo systemctl restart nox-bridge
curl http://127.0.0.1:8888/system/status
```

The service must run the updated `body/nox_brain_bridge.py`. Check
`systemctl cat nox-bridge` if it uses a separately copied script instead of the
repository checkout. Restart the desktop brain with your usual startup options
after updating its checkout too. An older Pi bridge returns an update-required
message to the model.
