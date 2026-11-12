# Exotel + Deepgram voice agent

This project accepts Exotel AgentStream WebSocket calls, streams caller audio
to Deepgram's Voice Agent API, and returns generated audio to Exotel. When
Deepgram reports `UserStartedSpeaking`, the server sends Exotel a `clear` event
to stop queued agent playback (barge-in).

## Run locally

Set `DEEPGRAM_API_KEY` in a `.env` file in this project folder, then install the
project and its declared dependencies:

```cmd
python -m pip install -e .
```

Start the WebSocket server:

```cmd
python main.py
```

It listens on port `5001` by default. Set `PORT` to change it. Leave this
terminal open while testing.

The packaged agent settings are in
[`src/voice_agent/config.json`](src/voice_agent/config.json). The audio
configuration currently expects mono `linear16` audio at 8 kHz in both
directions; it must match the audio format configured for the Exotel stream.

When Deepgram requests the configured `submit_customer_enquiry` tool, the app
validates and stages the enquiry, then responds immediately so the conversation
can continue. The actual email is sent after Exotel reports the call has ended
(or the Exotel WebSocket disconnects). Cancelled Deepgram function calls are
discarded and aren't emailed. Configure `BUSINESS_EMAIL`, `SMTP_EMAIL`, and
`SMTP_PASSWORD` in `.env`; for Gmail, `SMTP_PASSWORD` should be an app password.
Email sending requires all six enquiry fields defined in
`src/voice_agent/config.json`.

After each turn, the server logs Deepgram's STT, LLM, TTS, and total latency
report when provided. Every 50 caller audio chunks, it also logs the local
queue wait and Deepgram send time to help identify whether audio relay
backpressure is contributing to delay. A warning that the caller audio queue is
backing up indicates the local relay is falling behind; audio chunks are not
dropped.

## Expose it with ngrok

Open another Command Prompt window and run:

```cmd
ngrok http 5001
```

Use the public ngrok host with `wss://` and the `/media` path for Exotel's
AgentStream URL, for example `wss://example.ngrok-free.app/media`. Keep both
the server and ngrok running while testing.