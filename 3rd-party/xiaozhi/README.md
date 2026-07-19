# xiaozhi RT-Smart client

This example is a native RT-Smart xiaozhi client for K230 boards. It follows
the WebSocket protocol used by the xiaozhi ESP32 reference and the CanMV
`26-xiaozhi` example, but keeps transport, protocol, MCP, audio, and startup
code in separate modules:

```text
xiaozhi/
  include/                 Public module interfaces
  src/main.c               Command-line parsing and signal handling
  src/xiaozhi_activation.c OTA device registration and activation polling
  src/xiaozhi_mpp.c        Process-wide K230 VB initialization
  src/xiaozhi_app.c       Session state and message dispatch
  src/xiaozhi_transport.c libwebsockets connection and TX queues
  src/xiaozhi_protocol.c  Hello, listen, abort, and MCP envelopes
  src/mcp/                MCP core and individual service implementations
    xiaozhi_mcp.c         JSON-RPC, schemas, registry, and request dispatch
    xiaozhi_mcp_services.c Default capability registration
    xiaozhi_mcp_audio.c   Speaker volume get/set services
    xiaozhi_mcp_light.c   Optional GPIO light service
  src/xiaozhi_light.c     GPIO hardware implementation used by the light service
  src/xiaozhi_audio.c     K230 AI/AENC/ADEC/AO Opus pipeline
  src/xiaozhi_lvgl.c      Optional LVGL display, touch, and status UI
```

The installed program remains `xiaozhi_client.elf`.

## Features

- WebSocket and TLS connections with the xiaozhi headers and hello handshake.
- MCP JSON-RPC responses for `initialize`, `tools/list`, and `tools/call`.
- A bounded MCP tool registry with declarative schemas, duplicate-name checks,
  user-only filtering, and cursor pagination for larger tool sets.
- `self.audio_speaker.get_volume`, `self.audio_speaker.set_volume`, and the
  optional `self.light.get_state` and
  `self.light.set_state` GPIO light tools.
- K230 Opus capture through AI and AENC, with bounded non-blocking WebSocket
  upload queues.
- K230 Opus playback through ADEC and AO, with a playback queue outside the
  libwebsockets callback.
- Local K230 KWS wake-word detection using the bundled `xiaonan` model; the
  WebSocket session enters realtime listening only after the wake word.
- Automatic reconnect with bounded control and audio queues; reconnects forever
  by default after a WebSocket disconnect (`-r N` can set a finite limit).
- `--no-audio` mode for transport and MCP bring-up on boards without audio.
- Process-wide VB initialization before audio or future LVGL/media users.
- Optional LVGL UI with activation code, connection/session state, chat text,
  emotion, audio state, and a touch volume slider.

Additional MCP tools are registered with `xiaozhi_mcp_register_tool()` using a
static `xiaozhi_mcp_tool` descriptor and a callback. The registry validates
basic argument types and ranges, rejects duplicate names, and handles the MCP
`tools/list` cursor so application code does not need another dispatch chain.

## Build

Enable these RT-Smart options in menuconfig:

- `RTSMART_3RD_PARTY_ENABLE_LIBWEBSOCKETS`
- `RTSMART_3RD_PARTY_ENABLE_CJSON`
- `RTSMART_3RD_PARTY_ENABLE_LVGL` for the display UI
- `RTSMART_3RD_PARTY_ENABLE_XIAOZHI_SAMPLES`
- `RTSMART_XIAOZHI_MCP_LIGHT` when the board has a controllable GPIO light

The xiaozhi target also links the K230 MPP audio libraries through
`libs/mk/libmpp.mk`.
When `RTSMART_3RD_PARTY_ENABLE_LVGL` is enabled, it also links the existing
K230 LVGL display and touch port. The default display is the 480x800 ST7701
MIPI panel on OSD0. Use `--no-lvgl` for a headless run, or override the
connector, layer, and touch device with `--lvgl-connector`, `--lvgl-layer`,
and `--lvgl-touch`.

## Run

Run without a token to register the board with the xiaozhi service. The
program prints the activation code, polls the OTA endpoint, and then starts
the WebSocket session after the board is accepted:

```text
/sdcard/app/examples/3rd_party/xiaozhi_client.elf
```

To connect with a token that was obtained elsewhere:

```text
/sdcard/app/examples/3rd_party/xiaozhi_client.elf \
  -k '<activated-token>'
```

The default mode is `realtime`, gated by the local `xiaonan` wake-word model.
After the server hello, the client keeps audio local and waits for the wake
word. It then sends `listen/detect` followed by `listen/start` with
`mode=realtime`. Use `--no-wake-word` to start the configured listen mode
immediately. A custom compatible model can be selected with `--wake-word`;
the bundled model uses task `xiaonan`, two outputs, and a `0.5` threshold:

```text
xiaozhi_client.elf
xiaozhi_client.elf --no-wake-word
xiaozhi_client.elf --wake-word /sdcard/my_kws.kmodel \
  --wake-word-task xiaonan --wake-word-keywords 2 \
  --wake-word-threshold 0.5
```

The wake-word protocol text defaults to `xiaonanxiaonan`; override it with
`--wake-word-text` when using a model trained for another phrase.

The local wake detector is keyword spotting, not a general local VAD. The
server still performs turn detection after realtime listening starts. A local
energy VAD can be added later for power or bandwidth reduction, but it must
not replace AEC or suppress the silence frames needed by server-side VAD.

The audio 3A mask is
`1=ANS`, `2=AGC`, and `4=AEC`; `7` enables all three. Override it with
`--audio3a 0` or another mask when needed.
When AEC is enabled, the client configures a 200 ms echo delay for the K230
VQE path. The SDK's explicit far-end reference API is obsolete on this target,
so the client does not call it for every playback frame. AEC still requires a
working microphone and speaker path on the same K230 audio setup; the delay
may need board-specific tuning.

The built-in `self.audio_speaker.get_volume` and
`self.audio_speaker.set_volume` tools control the internal codec through
`/dev/acodec_device`. Their value is the CanMV-style `0` to `100` range; both
tools are omitted from `tools/list` when audio or the internal codec is
disabled.

The optional `self.light.get_state` tool reports the current logical light
state as `available=true` and `on=true` or `false`; `on=false` means the
configured light is currently off. `self.light.set_state` accepts a boolean `state` property. The GPIO pin
and the output value used for on are configured with
`RTSMART_XIAOZHI_MCP_LIGHT_GPIO` and
`RTSMART_XIAOZHI_MCP_LIGHT_ON_VALUE`. The 01Studio default is GPIO52 with
output value 1. Other boards default to no configured GPIO, so the light tool
is omitted until a valid pin is selected.

For a transport and MCP-only check:

```text
xiaozhi_client.elf -i --no-audio -d 30
```

Without `--token`, the sample calls the OTA activation endpoint and prints the
device activation code. It keeps polling until the device is accepted. Use
`--token` or `--no-activation` to skip that flow. `--no-activation` retains the
legacy `test-token` default and is intended for protocol bring-up only.

The default libwebsockets log level is `info`. Use `--log-level debug` or
`--debug` to include parser, client, header, and transport diagnostics. The
other accepted levels are `error` and `warn`.

The sample derives `Device-Id` from the WLAN MAC and derives a stable
UUID-shaped `Client-Id` from the K230 chip ID, falling back to the WLAN MAC
when the chip ID is unavailable. Override either with `--device-id` or
`--client-id`.
The fallback token is `test-token` for `--no-activation` protocol bring-up;
normal startup obtains the WebSocket token from the OTA response.

TLS verification is disabled by default because a K230 image may start with an
unset RTC. Use `--verify-tls` only after the board has a valid clock and a CA
bundle is available to the libwebsockets build.

## LVGL status UI

The LVGL UI starts before activation polling, so the device code and activation
message are visible on the display while the board waits for approval. It keeps
a scrollable transcript of the returned STT, assistant sentence, and emotion
messages, matching the serial log. LVGL is driven by one dedicated UI thread;
network and audio callbacks only publish state to it.

The `llm.text` emoji is handled as a visual-only token. It is not added to the
spoken transcript or sent to TTS. The `emotion` value selects an image or
optional animated GIF. The default resource directory is:

```text
/sdcard/app/examples/3rd_party/xiaozhi_assets
```

The xiaozhi example installs its LVGL assets beside the ELF under
`RTT_3RD_PARTY_EXAMPLES_ELF_INSTALL_PATH/xiaozhi_assets`. The normal RT-Smart
image build copies that directory to the SD card with the path above. The
entire `assets/` directory, including its `gifs/` and `font/` subdirectories,
is copied there during the xiaozhi build. The default assets are:

```text
font/Source_Han_Sans_CN_Regular.ttf  Chinese and English FreeType font
gifs/*.gif                           the 21 supplied emotion animations
```

For another exact mood animation, add `<emotion>.gif`,
`emoji_<emotion>.gif`, or `img_<emotion>.gif` under
`assets/gifs`; the xiaozhi Makefile copies the GIFs and font into the generated
SD-card image. The UI also accepts the grouped `img_<emotion-group>.gif`
fallback when additional PNG assets are provided. Use
`--lvgl-resource-dir <dir>` when the files are deployed elsewhere. FreeType is
used when enabled in the SDK; the built-in LVGL font is used if the TTF file or
FreeType support is unavailable.

The default display settings match the existing K230 LVGL sample:

```text
--lvgl-connector <id>  connector type; default is ST7701 480x800
--lvgl-layer <id>      OSD layer; default is OSD0 (4)
--lvgl-touch <id>      touch device id; default is 0
--lvgl-resource-dir <dir> font and emotion image/GIF directory
--no-lvgl              disable the display UI
```

## Audio options

The microphone/uplink defaults match the CanMV audio example: 16 kHz, mono,
16-bit, 960 samples per Opus frame, and internal I2S codec. The server may
advertise 24 kHz TTS audio, but the local K230 AI, Opus ADEC, and AO path stay
at 16 kHz. This matches CanMV: the ADEC is configured to produce 16 kHz PCM
directly, and the 960-sample decoded frame is sent to AO without an external
resampler. The server rate is retained for session status and diagnostics; it
does not reconfigure the local hardware path.
Use these options when the board uses another input path:

```text
--input-device i2s       Internal I2S microphone (default)
--input-device pdm       PDM microphone on AI device 1
--input-channel <n>      AI channel, default 0
--output-channel <n>     AO channel, default 0
--external-codec         External I2S codec instead of the internal codec
--ans                    Enable K230 microphone noise suppression
```

Audio failures are reported without tearing down the WebSocket session, so
`--no-audio` is not required for testing MCP on a board whose audio wiring is
not configured.
