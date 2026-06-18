# Gesture-Based Silent SOS (Demo)

A Streamlit + OpenCV + MediaPipe prototype: a predefined hand gesture in
front of a camera silently triggers a simulated emergency alert — no phone,
no call, no noise. Built as an extension of an "air-writing" hand-gesture
project toward a women's-safety use case.

## What's real vs. simulated

This is a **working, testable recognition pipeline** — the camera capture,
MediaPipe hand tracking, and the gesture state machine are fully real.
Security-dispatch and monitoring-center notifications are **simulated**:
they log and display exactly what a production system would do, without
making real network calls. The **WhatsApp alert is partially real**: it
builds a genuine `wa.me` click-to-chat link with the emergency message
pre-filled for each trusted contact — opening it starts a real WhatsApp
conversation, but (since this uses WhatsApp's free click-to-chat feature
rather than the paid Business API) a person still has to tap Send; it
can't auto-send silently. See the `PRODUCTION HOOK` comments in
`alert_system.py` for where to plug in fully automated services (Twilio,
SMTP, the WhatsApp Business Platform, your monitoring center's API, etc).

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

On first run, the app automatically downloads MediaPipe's hand-landmark
model (~10 MB) into `models/hand_landmarker.task`. This needs an internet
connection once; after that it's cached locally. If your machine has no
internet access, download it manually from the URL printed in the error
message and place it at that path.

## Using the demo

1. In the sidebar, choose **Webcam (live)** or **Upload video file** (to
   simulate a CCTV clip).
2. Choose which gesture should be armed:
   - **Signal for Help** — the real-world two-stage gesture: show an open
     palm with your thumb tucked across your palm, hold briefly, then close
     your fingers down over your thumb into a fist. Doing only the second
     half (a sudden fist) on its own will *not* trigger it — this two-stage
     requirement is what makes it resistant to accidental triggers.
   - **Fist Hold** — simply hold a closed fist for ~3 seconds (adjustable).
   - **Either gesture**.
3. Click **Start Monitoring**.
4. Perform the gesture in front of your camera. When recognized, the
   dashboard flashes red, a siren plays, a snapshot is saved, and the
   notification checklist (security, monitoring center, WhatsApp links,
   incident log) lights up.
5. Use **🚨 Test Alert** in the sidebar to preview the full alert flow
   instantly, without performing the gesture — handy for demos.

### Trusted contacts & WhatsApp

Enter one contact per line in the sidebar as `Name, Phone` (include the
country code, e.g. `Mother, +919999999999`). When an alert fires, each
contact with a valid number gets a real `wa.me` link in the dashboard and
incident log — clicking it opens WhatsApp with the emergency message
already typed in; you (or whoever's monitoring) just need to tap Send.
Contacts left without a number still appear in the list but won't get a
link.

## Tuning

If detection misfires (or doesn't trigger easily enough) for your camera/
lighting, adjust:
- `THUMB_TUCK_RATIO` and `MIN_CURLED_FOR_FIST` in `gesture_detector.py`
  (the geometric thresholds for classifying the hand pose)
- Fist-hold duration and cooldown sliders in the sidebar

## Project structure

```
app.py                 Streamlit UI, video loop, dashboard
gesture_detector.py     MediaPipe HandLandmarker wrapper + gesture state machine
alert_system.py         Incident creation/storage (simulated notifications)
sound_utils.py          In-code siren beep generator (no audio asset needed)
requirements.txt
models/                 Downloaded MediaPipe model (created on first run)
snapshots/              Saved incident snapshots
```

## Path to a real deployment

See the "How this would become a real production system" panel at the
bottom of the running app for the key additions needed: edge inference at
each camera, a real notification layer, live feed routing to a monitoring
center, human-in-the-loop confirmation before dispatch, and clear data
governance for handling sensitive footage and incident logs.
