import tempfile
from ultralytics import YOLO
import cv2
from deepface import DeepFace
import streamlit as st
import os
import time

yolo_model = YOLO("yolov8n.pt")
os.makedirs("registered_faces", exist_ok=True)

def capture_webcam_frame_auto(count=4, delay_sec=1, resize_width=None, show_preview=True):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam.")
        return []

    captured_faces = []
    preview_placeholder = st.empty() if show_preview else None

    for i in range(count):
        st.info(f"Capturing frame {i + 1} of {count}...")

        start_time = time.time()
        while time.time() - start_time < delay_sec:
            ret, frame = cap.read()
            if not ret:
                continue

            display_frame = frame.copy()
            if resize_width:
                h, w = display_frame.shape[:2]
                scale = resize_width / w
                display_frame = cv2.resize(display_frame, (resize_width, int(h * scale)))

            if show_preview and preview_placeholder:
                display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                preview_placeholder.image(display_frame_rgb, channels="RGB", caption=f"Preview {i + 1}")

        ret, frame = cap.read()
        if ret:
            captured_faces.append(frame)

    cap.release()
    return captured_faces



def detect_and_crop_face(frame, model):
    results = model.predict(frame, conf=0.5, classes=0, verbose=False)
    for result in results:
        boxes = result.boxes.xyxy
        if boxes is not None and len(boxes) > 0:
            boxes = boxes.cpu().numpy().astype(int)
            largest_box = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
            x1, y1, x2, y2 = largest_box
            cropped_face = frame[y1:y2, x1:x2]
            return cropped_face
    return None

def save_face_image(cropped_face, username, i):
    path = os.path.join("registered_faces", f"{username}__{i}.jpg")
    cv2.imwrite(path, cropped_face)


def get_registered_faces():
    faces = []
    for file in os.listdir("registered_faces"):
        if file.endswith(".jpg"):
            username = file.split("__")[0]
            path = os.path.join("registered_faces", file)
            faces.append((username, path))
    return faces


def authenticate_user(cropped_face):
    registered = get_registered_faces()
    
    if not registered:
        print("No registered faces found.")
        return None

    for username, path in registered:
        result = DeepFace.verify(
            img1_path=cropped_face,
            img2_path=path,
            model_name="Facenet",
            enforce_detection=False,
            threshold=0.3
        )

        if result["verified"]:
            print(f"Access Granted: {username}")
            return username

    print("Access Denied: No match found.")
    return None

st.set_page_config(page_title="Face Unlock System", layout="centered")

st.markdown("""
    <style>
        .main { background-color: #f9f9f9; }
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        h1, h2, h3, h4 {
            color: #222;
            font-family: 'Segoe UI', sans-serif;
        }
        .stButton>button {
            border-radius: 8px;
            background-color: #2E8B57;
            color: white;
            padding: 0.5rem 1rem;
            font-size: 1rem;
        }
        .stTextInput>div>input {
            border-radius: 6px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Face Unlock System")
st.caption("Fast, lightweight and secure facial authentication.")

menu = st.sidebar.radio("Navigation", ["Register", "Login"])

if menu == "Register":
    st.subheader("Register New Face")
    username = st.text_input("Enter your name")

    if st.button("Start Registration") and username:
        with st.spinner("Capturing your face..."):
            faces = capture_webcam_frame_auto(count=1, delay_sec=1, resize_width=480)
            
            if faces:
                cropped = detect_and_crop_face(faces[0], yolo_model)
                if cropped is not None:
                    save_path = f"registered_faces/{username}.jpg"
                    cv2.imwrite(save_path, cropped)
                    st.success(f"{username} registered successfully.")
                    st.info("You can now login using your face.")
                else:
                    st.warning("No face detected. Please try again.")
            else:
                st.error("Webcam failed to capture.")


elif menu == "Login":
    st.subheader("Login via Face Scan")

    if st.button("Scan Face"):
        with st.spinner("Scanning..."):
            faces = capture_webcam_frame_auto(count=1, delay_sec=2, resize_width=480)
            if faces:
                cropped = detect_and_crop_face(faces[0], yolo_model)
                if cropped is not None:
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        cv2.imwrite(tmp.name, cropped)
                        user = authenticate_user(tmp.name)
                        os.remove(tmp.name)

                    if user:
                        st.success(f"Access Granted: Welcome, {user[:-4]}.")
                    else:
                        st.error("Access Denied: Face not recognized.")
                else:
                    st.warning("No face detected.")
            else:
                st.error("Webcam failed to capture.")