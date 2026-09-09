# Writing a koe plugin

koe is composed the way [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)
is: everything is a plugin contributing to a shared context, and there is no
privileged core to patch. The ASR router, the tool registry, the file tools and
the 議事録 generator all attach the same way yours will.

## The shortest plugin that does something

Drop a `.py` file into the plugins directory — `%LOCALAPPDATA%\koe\plugins` on
Windows, `~/.local/share/koe/plugins` elsewhere — and restart, or press
**Reload** in Settings → Plugins.

```python
KOE_PLUGIN = {
    "name": "word-count",
    "description": "Counts words in the current transcript.",
    "version": "1.0.0",
    "inject": ["tools"],
}


def apply(ctx, config=None):
    registry = ctx.get("tools")

    async def count_words(args, run):
        transcript = ctx.get("last_transcript") or ""
        return f"{len(transcript.split()):,} words"

    from koe.tools import ToolSpec

    spec = ToolSpec(
        name="count_words",
        description="Count the words in the meeting transcript.",
        parameters={"type": "object", "properties": {}},
        execute=count_words,
    )
    ctx.scope.collect("tool:count_words", registry.register(spec))
```

That is the whole contract: module-level `KOE_PLUGIN` metadata, and an `apply`
that receives a context.

## Four things worth understanding

### `inject` decides *when* your plugin runs, not just whether

A plugin declaring `inject: ["tools"]` does not run at load time. It runs the
moment a `tools` service exists, and it is **torn down and rebuilt** if that
service is later replaced. That is what lets the running app swap an LLM
provider when you paste an API key without restarting anything.

If you inject a service that never appears, your plugin never activates. That
is the intended behaviour, and Settings → Plugins shows it.

### Registrations are effects, so put them on the scope

`registry.register(spec)` returns a disposer. Hand it to `ctx.scope.collect()`
and the kernel runs it when your plugin unloads.

```python
ctx.scope.collect("label", disposer)
```

Skip that step and disabling your plugin leaves a tool behind pointing at code
that is gone. Every registration API in koe returns a disposer for this reason:
`ctx.provide()`, `ctx.on()`, and `registry.register()`.

**Disabling a plugin unmounts it.** It is not a flag you are expected to check.

### A tool description is a prompt

It is the only thing between "the model uses this correctly" and "the model
guesses". Say when *not* to use the tool as well as what it does — that turns
out to matter more:

> Search file contents with a regular expression. Prefer this over reading
> whole files to look for something.

### Failures are results, not exceptions

Raise `ToolInvocationError` for a bad argument and the model is told what was
wrong and can retry. Any other exception is caught, logged, and returned as a
failed result — a tool crash never ends the turn.

```python
from koe.tools import ToolInvocationError

if not args.get("path"):
    raise ToolInvocationError("path is required")
```

## What a plugin can reach

| `ctx.get(...)` | What it is |
|---|---|
| `tools` | The tool registry |
| `workspace` | Workspace-fenced file access |
| `asr` | The speech-recognition router |
| `llm` | The configured LLM provider |
| `last_transcript` | The most recent transcript, as text |
| `last_minutes` | The most recent 議事録 |

Events, via `ctx.on(...)`:

| Event | When |
|---|---|
| `tools/pre-execute` | Before a tool runs. Return a `Denial` to refuse it. |
| `tools/post-execute` | After a tool returns. For auditing. |

An approval policy is an eight-line plugin:

```python
from koe.tools import Denial

def apply(ctx, config=None):
    async def gate(tool, run):
        if tool.dangerous and run.owner != "operator":
            return Denial("this tool needs approval")

    ctx.on("tools/pre-execute", gate)
```

Removing that plugin degrades to unguarded execution rather than breaking every
tool — which is the property that lets one tool set serve a headless deployment
and a desktop app.

## Security

**A plugin is trusted code.** It runs in-process with full Python privileges:
it can read your files, open sockets, and call anything koe can call. There is
no sandbox, and pretending otherwise would be worse than saying so. The
boundary that matters is the one at install time — read a plugin before you
drop it in, the same way you would a shell script.

## Debugging

A plugin that fails to import is listed in Settings → Plugins **with its
traceback**, not silently skipped. A plugin with no `apply` is listed with what
to add. Both stay disabled until they load cleanly.

To check a tool without a model in the loop:

```bash
curl -X POST http://127.0.0.1:8000/v1/tools/count_words -H 'Content-Type: application/json' -d '{"arguments":{}}'
```

It runs the same guarded pipeline, so a policy that would refuse the model
refuses this too.
