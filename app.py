"""
app.py
-------
Gesture-based Silent SOS — a demo of how the hand-gesture-recognition idea
from a "write in the air" project can be redirected toward a women's-safety
use case.

Run with:
    streamlit run app.py

Everything that would require real infrastructure in production (CCTV
network ingestion, SMS/email gateways, a security-dispatch system) is
SIMULATED here — see alert_system.py for exactly where the "PRODUCTION
HOOK" seams are.
"""

import os
import tempfile
import time

import cv2
import streamlit as st

from gesture_detector import SOSGestureEngine, GestureConfig
from alert_system import (
    init_session_state,
    trigger_alert,
    dismiss_active_alert,
    clear_incident_log,
    get_active_incident,
    parse_contacts,
)
from sound_utils import generate_alert_beep_base64

st.set_page_config(
    page_title="Gesture SOS — Silent Emergency Alert (Demo)",
    page_icon="🆘",
    layout="wide",
)

init_session_state()

# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
st.sidebar.title("🆘 Gesture SOS — Controls")

st.sidebar.markdown("#### 1. Video source")
source_choice = st.sidebar.radio("Input", ["Webcam (live)", "Upload video file"], index=0)

uploaded_video_path = None
if source_choice == "Upload video file":
    uploaded = st.sidebar.file_uploader("Upload a video (simulating a CCTV clip)", type=["mp4", "mov", "avi", "mkv"])
    if uploaded is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
        tmp.write(uploaded.read())
        tmp.flush()
        uploaded_video_path = tmp.name

st.sidebar.markdown("#### 2. Armed gesture(s)")
gesture_choice = st.sidebar.radio(
    "Which gesture should trigger an alert?",
    ["Signal for Help (recommended)", "Fist Hold", "Either gesture"],
    index=0,
)
armed = set()
if gesture_choice == "Signal for Help (recommended)":
    armed = {"signal_for_help"}
elif gesture_choice == "Fist Hold":
    armed = {"fist_hold"}
else:
    armed = {"signal_for_help", "fist_hold"}

st.sidebar.markdown("#### 3. Thresholds")
fist_hold_seconds = st.sidebar.slider("Fist Hold duration (s)", 1.0, 6.0, 3.0, 0.5)
cooldown_seconds = st.sidebar.slider("Cooldown between alerts (s)", 5, 60, 20, 5)

st.sidebar.markdown("#### 4. Incident metadata (simulated)")
camera_label = st.sidebar.text_input("Camera ID / location label", value="Camera 04 — Main Lobby")
contacts_raw = st.sidebar.text_area(
    "Trusted contacts — one per line: Name, Phone (with country code)",
    value="Mother, +919999999999\nAsha, +918888888888",
    height=90,
)
contacts = parse_contacts(contacts_raw)
st.sidebar.caption(
    "Phone numbers are used to build real WhatsApp click-to-chat links "
    "(wa.me) for the alert — opening one pre-fills the emergency message, "
    "but a person still needs to tap Send."
)

show_landmarks = st.sidebar.checkbox("Show hand landmark overlay", value=True)

st.sidebar.markdown("#### 5. Monitoring")
run = st.sidebar.checkbox("▶ Start Monitoring", key="run")

st.sidebar.markdown("---")
test_col1, test_col2 = st.sidebar.columns(2)
trigger_test = test_col1.button("🚨 Test Alert")
clear_log = test_col2.button("🗑 Clear Log")

if clear_log:
    clear_incident_log()

st.sidebar.caption(
    "All alerts in this demo are simulated. No real SMS, email, or CCTV "
    "network calls are made — see alert_system.py for where to plug those in."
)

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
st.title("🆘 Gesture-Based Silent SOS")
st.caption(
    "A computer-vision prototype: a predefined hand gesture in front of a "
    "camera silently triggers a simulated emergency alert — no phone, no call, "
    "no noise."
)

status_banner = st.empty()

col_video, col_dashboard = st.columns([2, 1.3])

with col_video:
    st.subheader("📷 Camera Feed")
    frame_placeholder = st.empty()
    status_text_placeholder = st.empty()
    if not run:
        frame_placeholder.info("Monitoring is stopped. Toggle **Start Monitoring** in the sidebar.")

with col_dashboard:
    st.subheader("🛰️ Live Monitoring Dashboard")
    dashboard_placeholder = st.empty()

st.subheader("📋 Incident Log")
log_placeholder = st.empty()


def render_status_banner():
    active = get_active_incident()
    if st.session_state.alert_active and active:
        status_banner.error(
            f"🚨 **SOS ALERT ACTIVE** — {active['gesture_label']} detected at "
            f"{active['timestamp'].strftime('%H:%M:%S')} ({active['location']})"
        )
    else:
        status_banner.success("✅ Monitoring — All Clear")


def render_dashboard():
    active = get_active_incident()
    with dashboard_placeholder.container():
        if active:
            n = active["notifications"]
            st.markdown(f"**Active incident:** `{active['id']}`")
            st.markdown("✅ Security personnel notified" if n["security_personnel_notified"] else "⬜ Security personnel notified")
            st.markdown("✅ Live feed shared with monitoring center" if n["monitoring_center_feed_shared"] else "⬜ Live feed shared")
            st.markdown("**📱 WhatsApp alerts:**" if n["whatsapp_links"] else "⬜ No trusted contacts configured")
            for link in n["whatsapp_links"]:
                if link["link"]:
                    st.markdown(f"&nbsp;&nbsp;✅ {link['name']} — [open chat to send ↗]({link['link']})")
                else:
                    st.markdown(f"&nbsp;&nbsp;⬜ {link['name']} — no valid phone number")
            st.markdown("✅ Incident logged" if n["incident_logged"] else "⬜ Incident logged")
            if active["snapshot_path"] and os.path.exists(active["snapshot_path"]):
                st.image(active["snapshot_path"], caption="Snapshot at time of alert", use_container_width=True)
            if st.button("✔ Acknowledge & Clear Alert", key=f"ack_{active['id']}"):
                dismiss_active_alert()
                st.rerun()
        else:
            st.markdown("No active alert. System is idle and watching for the armed gesture(s).")
            st.markdown(f"**Armed gesture(s):** {', '.join(sorted(armed)) if armed else 'none'}")


def render_log():
    incidents = st.session_state.incidents
    with log_placeholder.container():
        if not incidents:
            st.caption("No incidents recorded yet.")
            return
        for inc in incidents[:10]:
            with st.expander(
                f"{inc['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} — {inc['gesture_label']} — {inc['location']}"
                + ("  (acknowledged)" if inc["acknowledged"] else "  (ACTIVE)" if inc["id"] == st.session_state.active_incident_id else "")
            ):
                c1, c2 = st.columns([1, 2])
                with c1:
                    if inc["snapshot_path"] and os.path.exists(inc["snapshot_path"]):
                        st.image(inc["snapshot_path"], use_container_width=True)
                with c2:
                    st.write(f"**Incident ID:** {inc['id']}")
                    st.write(f"**Camera:** {inc['location']}")
                    n = inc["notifications"]
                    st.write("- Security personnel notified ✅" if n["security_personnel_notified"] else "- Security personnel notified ❌")
                    st.write("- Monitoring center feed shared ✅" if n["monitoring_center_feed_shared"] else "- Monitoring center feed shared ❌")
                    st.write("**WhatsApp alerts:**")
                    if n["whatsapp_links"]:
                        for link in n["whatsapp_links"]:
                            if link["link"]:
                                st.markdown(f"- {link['name']} — [open chat to send ↗]({link['link']})")
                            else:
                                st.markdown(f"- {link['name']} — no valid phone number")
                    else:
                        st.write("- none configured")


def maybe_play_sound():
    if st.session_state.play_alert_sound:
        b64 = generate_alert_beep_base64()
        st.markdown(
            f'<audio autoplay="true"><source src="data:audio/wav;base64,{b64}" type="audio/wav"></audio>',
            unsafe_allow_html=True,
        )
        st.session_state.play_alert_sound = False


# manual test trigger works whether or not the camera loop is running
if trigger_test:
    blank = None
    trigger_alert("test_trigger", blank, camera_label, contacts)

render_status_banner()
render_dashboard()
render_log()
maybe_play_sound()

# ---------------------------------------------------------------------------
# Live processing loop
# ---------------------------------------------------------------------------
if run:
    config = GestureConfig(fist_hold_seconds=fist_hold_seconds, cooldown_seconds=cooldown_seconds)
    try:
        engine = SOSGestureEngine(config)
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    if source_choice == "Webcam (live)":
        cap = cv2.VideoCapture(0)
    else:
        if not uploaded_video_path:
            st.warning("Please upload a video file in the sidebar to start monitoring.")
            cap = None
        else:
            cap = cv2.VideoCapture(uploaded_video_path)

    if cap is not None and cap.isOpened():
        try:
            while st.session_state.run:
                ret, frame = cap.read()
                if not ret:
                    if source_choice == "Upload video file":
                        # loop the demo clip back to the start
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        st.error("Could not read from webcam. Check camera permissions/connection.")
                        break

                if source_choice == "Webcam (live)":
                    frame = cv2.flip(frame, 1)  # natural mirror view

                annotated_rgb, event, status_text = engine.process(frame, armed, draw=show_landmarks)

                if event is not None:
                    trigger_alert(event, frame, camera_label, contacts)

                frame_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)
                status_text_placeholder.caption(f"Gesture engine status: {status_text}")

                if event is not None:
                    render_status_banner()
                    render_dashboard()
                    render_log()
                    maybe_play_sound()

                time.sleep(0.01)
        finally:
            cap.release()
            engine.close()
    elif cap is not None:
        st.error("Could not open video source.")

st.markdown("---")
with st.expander("🏗️ How this would become a real production system"):
    st.markdown(
        """
This prototype proves the recognition pipeline end-to-end. To turn it into a
deployed public-safety system, the main pieces to add are:

- **Edge inference boxes** at each CCTV camera (or a central GPU server
  ingesting RTSP streams) running this same MediaPipe pipeline continuously.
- **A real notification layer** in `alert_system.py` — e.g. Twilio for SMS,
  SMTP/SendGrid for email, and a webhook into the security control room's
  existing incident-management software.
- **Live feed routing** — instead of a single snapshot, push the camera's
  RTSP/HLS stream to the monitoring center the moment an alert fires.
- **False-positive review** — a human-in-the-loop step where a monitoring
  operator briefly confirms the alert before dispatch, especially important
  given the safety-critical nature of the system.
- **Privacy & governance** — clear policies on retention of footage,
  who can access incident logs, and audit trails, since this handles
  sensitive personal-safety data.
        """
    )
