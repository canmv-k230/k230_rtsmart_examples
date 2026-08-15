# Audio sample

The sample separates process control, argument validation, mode dispatch, and
MPP resource ownership:

- `sample_audio.c`: signals, worker lifetime, logging, and VB lifetime
- `sample_audio_config.c`: defaults, command-line parsing, and validation
- `sample_audio_modes.c`: dispatch for modes 0 through 11
- `audio_sample.c`: AI, AO, AENC, ADEC, bind, buffer, and codec ownership
- `audio_io.c`: bounded asynchronous file reader and writer queues
- `audio_wav.c`: independent streaming WAV reader and writer contexts
- `tests/`: host tests for every mode dispatch and file helper edge cases

## Build and host tests

From this directory:

```sh
make all
make test
```

The host tests do not require K230 hardware. They cover command-line defaults
and invalid values, dispatch and return propagation for all 12 modes, unique
output paths, latest-recording selection, empty-output removal, and 44100 Hz
stereo WAV header/data-rate round trips. They also cover sequential reads,
EOF handling, truncated WAV rejection, and lossless asynchronous queue reads
and writes. Resource cleanup still requires the board tests below because the
MPP drivers own those resources.

## Usage

Modes can be selected by name or number. Named modes and separate input/output
options are preferred:

```sh
sample_audio.elf 0 -o /sdcard/mic.wav
sample_audio.elf 0 -s 1 -o /sdcard/pdm.wav
sample_audio.elf 1 -i /sdcard/mic.wav
sample_audio.elf 2
sample_audio.elf 2 -s 1
```

Run the sample without arguments, or with `--help`, for the complete mode and
option list. The older `-type`, `-filename`, `-samplerate`, and related options
remain accepted for compatibility.

## Board test matrix

Hardware behavior and driver cleanup must still be checked on a K230 board.
Run each capture mode once with its default I2S source and once with `-s 1` for
PDM. Let mode 0 finish once and interrupt it once with `Ctrl-C`. Let mode 1 play
the complete file once, then repeat it and stop early with `q` and `Ctrl-C`.
For modes 2 through 10, stop once with `q` and once with `Ctrl-C`; also interrupt
mode 11. After every run, start the same mode again to verify that AI, AO, AENC,
ADEC, bind, file, and VB resources were released.

| Group | Mode | Board case | File format |
| --- | ---: | --- | --- |
| WAV file I/O | 0 | Record selected input for 15 seconds | WAV output |
| WAV file I/O | 1 | Play I2S output once | WAV input |
| PCM loopback | 2 | Selected input to AO with frame APIs | None |
| PCM loopback | 3 | Bind selected input to AO | None |
| G711 file I/O | 4 | Bound selected input to G711 encoder | Raw G711A output |
| G711 file I/O | 5 | Bound G711 decoder to AO | Raw G711A input |
| G711 file I/O | 6 | Selected input to G711 encoder with frame APIs | Raw G711A output |
| G711 file I/O | 7 | API G711 decoder to AO | Raw G711A input |
| G711 file I/O | 8 | Concurrent selected-input record and playback | Raw G711A input and output |
| Codec loopback | 9 | Selected input through G711 encode/decode | None |
| Codec loopback | 10 | Selected input through Opus at 8 kHz | None |
| Codec control | 11 | Codec control menu, then `q` or `Ctrl-C` | None |

`-s 0` selects I2S input and `-s 1` selects PDM input. I2S is the default.
The old source-specific names `record-i2s`, `record-pdm`, `loop-i2s`,
`loop-pdm`, `bind-i2s`, and `bind-pdm` remain accepted. Numeric mode IDs were
renumbered; old numeric aliases are not retained because they overlap the new
canonical IDs.

File arguments must use absolute paths. WAV modes default to
`/sdcard/test.wav`; G711 file modes default to `/sdcard/test.g711a`.
Mode 8 records its encoded stream to a unique `/sdcard/test.g711a` path.
Output files are never overwritten. If a requested name already exists, the
sample adds a numeric suffix such as `_001` and prints the selected path.
When playback uses its default input path, it selects the newest matching base
or suffixed recording. An explicit `-i` path is always used exactly as given.
WAV playback preserves the PCM data, plays the file once, drains queued AO
frames, and exits. Press `q` or `Ctrl-C` to stop it early.
All file-backed modes use bounded 32-frame queues and dedicated file workers.
MPP frames are copied and released promptly, so ordinary SD-card latency does
not hold an AI, AO, AENC, or ADEC buffer. Writers drain queued data before WAV
headers are finalized or encoded files are closed.
An output file is removed if startup fails before any audio is written.
The internal codec is enabled by default, matching the onboard audio path.
I2S WAV recording defaults to the internal codec's right mono input. Use `-c 2`
when recording a stereo source; use `-M 1` to select the left I2S input for
mono recording.
Use `--codec external` for external I2S. External I2S duplex modes 2 and 3 then
default to 32-bit samples, as required by the hardware; explicitly selecting
16 or 24 bits is rejected before MPP resources are initialized. Modes 4
through 10 use 16-bit codec audio, and Opus mode 10 uses 8000 Hz. Supported
sample rates are 8000, 12000, 16000, 24000, 32000, 44100, 48000, 96000, and
192000 Hz.

PDM supports ANS only. I2S AGC supports 8000, 16000, 32000, and 48000 Hz. With
the current AEC reference-buffer limit, AEC is accepted only for I2S codec
duplex/loopback modes 8 through 10 at 8000 or 12000 Hz.
