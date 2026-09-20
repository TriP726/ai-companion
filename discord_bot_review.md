# Review of Phi-3's Discord bot example

## Verdict: outdated. It will crash on discord.py 2.x.

The structure is right, but two things are wrong and both are fatal on the
version you actually have installed.

## Error 1 — `discord.Client()` takes required `intents` in 2.x

    client = discord.Client()        # TypeError on discord.py 2.0+

Intents became mandatory in discord.py 2.0. This raises immediately.

## Error 2 — `message.content` is empty without the Message Content intent

Even after you pass intents, `message.content` comes back as an **empty string**
unless you explicitly enable the privileged Message Content intent, BOTH in code
and in the Discord Developer Portal. So `if message.content == "!hello"` is
silently never true — no error, just a bot that ignores you. This is the single
most common "my bot doesn't respond" problem.

## Corrected version

```python
import discord

intents = discord.Intents.default()
intents.message_content = True          # privileged — see portal step below

client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(f"Logged in as {client.user}")


@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if message.content == "!hello":
        await message.channel.send("Hello!")


client.run("YOUR_BOT_TOKEN")
```

### Portal step the model did not mention

1. https://discord.com/developers/applications → your app → **Bot**
2. Scroll to **Privileged Gateway Intents**
3. Enable **MESSAGE CONTENT INTENT** → Save

Without this the corrected code still won't respond.

## Also worth knowing

- **Never hardcode the token.** Anyone who sees your screen or your repo owns
  your bot. Use an environment variable or a `.env` file that is gitignored:

  ```python
  import os
  client.run(os.environ["DISCORD_TOKEN"])
  ```

- `discord.Client` + `on_message` is the low-level API. For anything beyond a
  toy, `commands.Bot` or slash commands (`discord.app_commands`) is the modern
  path — Discord has been pushing hard toward slash commands.

## Why the model got this wrong

Phi-3-mini is a ~3.8B parameter model with a training cutoff well before now,
and small models are disproportionately likely to reproduce the most common
pattern in their training data — which for discord.py is pre-2.0 code, since
that era dominated tutorials for years.

**This is not a bug in your app.** It is the expected accuracy ceiling of a
2.3 GB quantized model. Treat its code output as a first draft to verify, never
as authoritative.
