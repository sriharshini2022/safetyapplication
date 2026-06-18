"""
alert_system.py
-----------------
Manages incident state for the SOS demo. Most of this is SIMULATED: no real
SMS/email calls or CCTV-network pushes are made. The one exception is the
WhatsApp links, which are REAL wa.me click-to-chat links — they open an
actual WhatsApp conversation with the alert message pre-filled, but a human
still has to press Send (WhatsApp's free click-to-chat API can't auto-send
silently; only the paid WhatsApp Business Platform can).

To wire the rest of this up to a real system later, the natural seams are
marked with "PRODUCTION HOOK" comments — that's where you'd drop in a
Twilio call, an SMTP send, a push to your monitoring center's API, etc.
"""

import os
import time
import urllib.parse
import uuid
from datetime import datetime

import cv2
import streamlit as st

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

GESTURE_LABELS = {
    "signal_for_help": "Signal for Help (open palm → fist over thumb)",
    "fist_hold": "Sustained Fist Hold",
    "test_trigger": "Manual Test Trigger",
}


def parse_contacts(raw: str) -> list:
    """Parse the sidebar textarea (one contact per line, 'Name, Phone') into
    a list of {"name", "phone"} dicts."""
    contacts = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if "," in line:
            name, phone = line.split(",", 1)
        else:
            name, phone = line, ""
        contacts.append({"name": name.strip() or "Contact", "phone": phone.strip()})
    return contacts


def _digits_only(phone: str) -> str:
    return "".join(ch for ch in phone if ch.isdigit())


def build_whatsapp_links(contacts: list, message: str) -> list:
    """Build WhatsApp click-to-chat (wa.me) links for each contact that has a
    usable phone number. These are REAL links — opening one starts a WhatsApp
    chat with the message pre-filled. WhatsApp's click-to-chat API doesn't
    require any credentials, but it still requires a human to press Send in
    the chat window; it can't silently auto-send on its own. For fully
    automatic sending you'd need the WhatsApp Business Platform (Meta Cloud
    API or Twilio) — see the PRODUCTION HOOK note in trigger_alert below.
    """
    encoded_message = urllib.parse.quote(message)
    links = []
    for c in contacts:
        digits = _digits_only(c.get("phone", ""))
        link = f"https://wa.me/{digits}?text={encoded_message}" if len(digits) >= 8 else None
        links.append({"name": c.get("name", "Contact"), "phone": c.get("phone", ""), "link": link})
    return links


def init_session_state():
    defaults = {
        "incidents": [],
        "alert_active": False,
        "active_incident_id": None,
        "play_alert_sound": False,
        "run": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def trigger_alert(gesture_type: str, frame_bgr, location_label: str, contacts: list):
    """Create a new incident, save a snapshot, and mark every notification
    step as completed (simulated). Sets alert_active so the dashboard flashes."""

    incident_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now()

    snapshot_path = None
    if frame_bgr is not None:
        snapshot_path = os.path.join(SNAPSHOT_DIR, f"incident_{incident_id}.jpg")
        try:
            cv2.imwrite(snapshot_path, frame_bgr)
        except Exception:
            snapshot_path = None

    whatsapp_message = (
        "🚨 EMERGENCY ALERT 🚨\n"
        "This is an automated safety alert. The person may need urgent help.\n"
        f"Location: {location_label or 'Unknown'}\n"
        f"Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Incident ID: {incident_id}\n"
        "Please try to reach them or contact local authorities immediately."
    )
    whatsapp_links = build_whatsapp_links(contacts, whatsapp_message)

    # PRODUCTION HOOK: replace each "True" below with a real call, e.g.
    #   notify_security_dispatch(incident_id, location_label)
    #   push_feed_to_monitoring_center(snapshot_path, camera_id=location_label)
    #   send_whatsapp_via_business_api(contacts, whatsapp_message)  # for silent auto-send
    notifications = {
        "security_personnel_notified": True,
        "monitoring_center_feed_shared": True,
        "whatsapp_links": whatsapp_links,
        "whatsapp_alert_sent": any(link["link"] for link in whatsapp_links),
        "incident_logged": True,
    }

    incident = {
        "id": incident_id,
        "timestamp": timestamp,
        "gesture_type": gesture_type,
        "gesture_label": GESTURE_LABELS.get(gesture_type, gesture_type),
        "location": location_label or "Unknown camera location",
        "snapshot_path": snapshot_path,
        "notifications": notifications,
        "acknowledged": False,
    }

    st.session_state.incidents.insert(0, incident)
    st.session_state.alert_active = True
    st.session_state.active_incident_id = incident_id
    st.session_state.play_alert_sound = True
    return incident


def dismiss_active_alert():
    st.session_state.alert_active = False
    if st.session_state.active_incident_id:
        for inc in st.session_state.incidents:
            if inc["id"] == st.session_state.active_incident_id:
                inc["acknowledged"] = True
                break
    st.session_state.active_incident_id = None


def clear_incident_log():
    st.session_state.incidents = []
    st.session_state.alert_active = False
    st.session_state.active_incident_id = None
    # best-effort cleanup of saved snapshots
    for f in os.listdir(SNAPSHOT_DIR):
        try:
            os.remove(os.path.join(SNAPSHOT_DIR, f))
        except OSError:
            pass


def get_active_incident():
    if not st.session_state.active_incident_id:
        return None
    for inc in st.session_state.incidents:
        if inc["id"] == st.session_state.active_incident_id:
            return inc
    return None
