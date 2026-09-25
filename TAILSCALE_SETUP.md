# PiDog Tailscale autostart and remote desktop brain

This setup lets the Raspberry Pi boot the PiDog body, bridge, voice service and
camera without an SSH session. It then maintains an outbound SSH connection over
Tailscale to one approved desktop brain.

The robot's control and camera services remain on the Pi's loopback interface.
They are **not** exposed to the public internet or directly to every member of
the tailnet.

## Result

After both machines boot:

- Pi local services start and the camera warms automatically.
- Tailscale reconnects using its persisted device identity.
- The Pi reconnects an outbound reverse SSH tunnel to the configured desktop.
- The desktop application continues to use:
  - bridge: `http://127.0.0.1:18888`
  - camera: `http://127.0.0.1:19000`
- If Wi-Fi, Tailscale or SSH drops, systemd retries the link.
- If the desktop is absent, Pi-local body safety, ultrasonic and voice services
  still run.

The Pi connects only to `PIDOG_BRAIN_HOST`. It does not trust or select an
arbitrary tailnet computer.

## Safety

Complete setup with PiDog supported safely and body following disarmed. The
link is transport only; it must never bypass ultrasonic stops, motion latching,
busy-leg rejection or desktop following disarm logic.

## 1. Install Tailscale on the desktop

Install Tailscale for Windows and sign in to the tailnet you intend to use. Give
the desktop a stable, distinctive machine name in the Tailscale admin console,
for example:

```text
pidog-brain
```

MagicDNS should be enabled. Confirm the name from PowerShell:

```powershell
tailscale status
tailscale ip -4
```

## 2. Enable the Windows OpenSSH server

Run an elevated PowerShell window:

```powershell
Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH.Server*'
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Set-Service -Name sshd -StartupType Automatic
Start-Service sshd
Get-Service sshd
```

The install normally creates the Windows Defender Firewall rule
`OpenSSH-Server-In-TCP`. Confirm it exists:

```powershell
Get-NetFirewallRule -Name OpenSSH-Server-In-TCP
```

Use a Tailscale access-control policy so only the Pi device (or its device tag)
can reach TCP port 22 on this desktop. Do not forward port 22 on the home router.

## 3. Install Tailscale on the Pi

SSH to the Pi by the local route for this one-time setup:

```powershell
ssh david@raspberrypi.local
```

On the Pi, install Tailscale using its official installer:

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo systemctl enable --now tailscaled
sudo tailscale up --hostname=pidog
```

Open the authentication URL printed by `tailscale up` and add the Pi to the
same tailnet. This login is persisted; no auth key needs to remain in a file.

Check it:

```bash
tailscale status
tailscale ip -4
getent hosts pidog-brain
```

If the tailnet uses device approval, approve the Pi in the Tailscale admin
console.

## 4. Install the Pi boot and link services

Check out this branch on the Pi:

```bash
cd /home/david/pidog-embodiment
git fetch origin
git switch codex/tailscale-autostart
git pull --ff-only
```

Run the installer with the desktop's MagicDNS name and Windows account name:

```bash
cd /home/david/pidog-embodiment/body
sudo bash ./install-tailscale-autostart.sh pidog-brain d_s_h
```

The installer:

1. installs `pidog-start`;
2. installs and enables `pidog-boot.service`;
3. installs the resilient `pidog-brain-link.service`;
4. creates a dedicated key at
   `/home/david/.ssh/pidog_brain_ed25519`;
5. prints the public key that must be authorised on Windows.

It does not store a Tailscale authentication key.

## 5. Authorise the Pi's dedicated key on Windows

Copy the single public-key line printed by the installer. It begins with
`ssh-ed25519`.

For a normal, non-administrator Windows account, run PowerShell as that account:

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.ssh"
notepad "$env:USERPROFILE\.ssh\authorized_keys"
```

Paste the public-key line as one line, save, then apply permissions:

```powershell
icacls "$env:USERPROFILE\.ssh\authorized_keys" /inheritance:r
icacls "$env:USERPROFILE\.ssh\authorized_keys" /grant:r "$env:USERNAME:(R,W)"
```

If `d_s_h` is a member of the local Administrators group, Windows OpenSSH
normally uses this file instead. Run elevated PowerShell:

```powershell
notepad C:\ProgramData\ssh\administrators_authorized_keys
icacls C:\ProgramData\ssh\administrators_authorized_keys /inheritance:r
icacls C:\ProgramData\ssh\administrators_authorized_keys /grant "Administrators:F" /grant "SYSTEM:F"
```

Paste the same single public-key line, save it, and restart SSH:

```powershell
Restart-Service sshd
```

## 6. Test the Pi-to-desktop login

On the Pi:

```bash
sudo -u david ssh \
  -i /home/david/.ssh/pidog_brain_ed25519 \
  -o IdentitiesOnly=yes \
  d_s_h@pidog-brain
```

The first connection records the desktop host key. Confirm the fingerprint
against the desktop before accepting if prompted. The login must complete
without asking for a password. Type `exit` afterwards.

If login fails, do not proceed until key authentication works.

## 7. Enable and start the persistent link

On the Pi:

```bash
sudo systemctl enable --now pidog-brain-link.service
sudo systemctl restart pidog-brain-link.service
systemctl status pidog-boot pidog-brain-link tailscaled --no-pager -l
sudo journalctl -u pidog-brain-link -n 50 --no-pager -l
```

On Windows, verify the reverse-forwarded endpoints:

```powershell
curl.exe --max-time 10 http://127.0.0.1:18888/status
curl.exe --max-time 20 http://127.0.0.1:18888/frame.jpg --output "$env:TEMP\pidog-frame.jpg"
Get-Item "$env:TEMP\pidog-frame.jpg"
Test-NetConnection 127.0.0.1 -Port 19000
```

You no longer run the old long `ssh -L ...` startup command. Start the desktop
application normally; its localhost URLs remain unchanged.

## 8. Reboot test

With PiDog safely supported and locomotion disarmed:

```bash
sudo reboot
```

After roughly one minute, check Windows again:

```powershell
curl.exe --max-time 10 http://127.0.0.1:18888/status
```

Then start:

```powershell
Set-Location C:\Users\d_s_h\Documents\pidog-embodiment
.\venv\Scripts\Activate.ps1
python .\pidog_yolo_vlm.py
```

## Change the approved brain computer

Edit the Pi's root-owned configuration:

```bash
sudo nano /etc/pidog-brain-link.env
sudo systemctl restart pidog-brain-link.service
```

Change `PIDOG_BRAIN_HOST` and `PIDOG_BRAIN_USER`, install the Pi public key
on that computer, and verify its SSH host key. Only one brain link service
instance is maintained.

## Diagnostics

### All boot services

```bash
systemctl is-active tailscaled pidog-boot nox-body nox-bridge nox-voice pidog-brain-link
systemctl is-enabled tailscaled pidog-boot nox-body nox-bridge nox-voice pidog-brain-link
```

### Tailscale

```bash
tailscale status
tailscale ping pidog-brain
sudo journalctl -u tailscaled -n 50 --no-pager -l
```

A relayed DERP connection is slower than a direct connection but should still
work. Camera latency and throughput are the practical test.

### Pi startup

```bash
sudo journalctl -u pidog-boot -u nox-body -u nox-bridge -u nox-voice -b --no-pager -l
```

### Persistent brain tunnel

```bash
sudo journalctl -u pidog-brain-link -f -l
```

Common messages:

| Message | Meaning |
| --- | --- |
| `waiting for Tailscale` | Wi-Fi/Tailscale is not ready yet |
| `Permission denied (publickey)` | Pi key is not installed in the correct Windows authorised-keys file |
| `remote port forwarding failed` | Desktop port 18888/19000 is occupied or forwarding is disabled |
| `Could not resolve hostname` | MagicDNS/hostname is wrong |
| repeated reconnects | inspect Wi-Fi, Tailscale and Windows `sshd` logs |

### Windows SSH server log

In elevated PowerShell:

```powershell
Get-WinEvent -LogName OpenSSH/Operational -MaxEvents 50 |
  Format-List TimeCreated,Id,LevelDisplayName,Message
```

## Disable remote auto-connect without disabling the robot

On the Pi:

```bash
sudo systemctl disable --now pidog-brain-link.service
```

The local PiDog services still boot and the camera still warms.

To disable all new boot automation:

```bash
sudo systemctl disable --now pidog-brain-link.service pidog-boot.service
```

This does not delete configuration, keys or the existing Nox service units.
