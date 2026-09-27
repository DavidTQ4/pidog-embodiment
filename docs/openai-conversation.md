# Optional OpenAI conversation

The desktop keeps Ollama `qwen3:8b` as its default. Existing startup commands
continue to work. To select GPT-6 Luna with Fast mode, add:

```powershell
python pidog_yolo_vlm.py --conversation-backend openai
```

Keep any other startup arguments you normally use. The API model is
`gpt-6-luna`, with `service_tier="fast"` and `reasoning_effort="none"`.
An optional `--openai-model` override is available for compatible models.

Set `OPENAI_API_KEY` in your Windows user environment, then open a new terminal
so the desktop process inherits it. For a temporary session, set
`$env:OPENAI_API_KEY` before launching. Never put the key in source files.
The program checks for the key before loading models when OpenAI is selected.
The existing `requests` dependency is sufficient; no OpenAI SDK is required.

## Persist the key on the Windows desktop

Run this once in PowerShell. The prompt hides the key and keeps its literal value
out of command history. This saves it for your Windows user (including this
project), not solely for this folder:

```powershell
$secret = Read-Host "OpenAI API key" -AsSecureString
$credential = New-Object System.Net.NetworkCredential('', $secret)
$env:OPENAI_API_KEY = $credential.Password
[Environment]::SetEnvironmentVariable("OPENAI_API_KEY", $env:OPENAI_API_KEY, "User")
Remove-Variable secret, credential
```

The current shell is ready immediately. Completely close and reopen Windows
Terminal for future sessions; new tabs in an existing process can inherit old
environment values. Restart other launchers if they were already running.
Verify without printing the key:

```powershell
[pscustomobject]@{
    CurrentKeyPresent = -not [string]::IsNullOrWhiteSpace($env:OPENAI_API_KEY)
    SavedKeyPresent = -not [string]::IsNullOrWhiteSpace(
        [Environment]::GetEnvironmentVariable("OPENAI_API_KEY", "User")
    )
}
```

## Update and launch

On another Windows checkout, stop the running desktop brain and run:

```powershell
Set-Location C:\Users\d_s_h\Documents\pidog-embodiment
git switch codex/fluffy-dance
git pull --ff-only
.\venv\Scripts\Activate.ps1
python .\pidog_yolo_vlm.py --conversation-backend openai
```

Keep any additional startup options you normally use. On the checkout where this
change was implemented, simply restart the desktop brain with the new option.

No Pi update, API key, package installation, or service restart is required for
this feature: only the Windows conversation path changed. To optionally keep
the Pi checkout synchronized, use:

```bash
cd /home/david/pidog-embodiment
git switch codex/fluffy-dance
git pull --ff-only
```

This switches spoken conversation and game commentary. YOLO, Qwen-VL vision,
Whisper transcription, speech output, and the robot action broker stay as they
are. The existing bounded tools, conversation history, and action limits apply
to either backend. OpenAI receives conversation text and verified robot context,
including scene descriptions when requested; camera images remain local.

Web search still uses the existing Ollama web tools and `OLLAMA_API_KEY`.
`--disable-web-search` disables those tools for either backend.
Conversation API failures are logged and use the existing local Qwen-VL fallback;
game commentary retains its existing error handling.

To switch back, omit `--conversation-backend openai` or use
`--conversation-backend ollama`. `--conversation-model` still selects the Ollama
model and does not change the OpenAI model.

References: [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna)
and [Fast mode pricing](https://developers.openai.com/api/docs/pricing?tab=suite).
