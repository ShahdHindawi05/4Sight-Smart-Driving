from PIL import Image
import sys
import torch
import numpy as np
import tempfile
import os
import streamlit as st
from pathlib import Path
from AI_Models.Drowsiness.drowsiness_detector import analyze_video
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "AI_Models"))
from traffic_sign_model import load_traffic_sign_model

TRAFFIC_MODEL_PATH = (
    ROOT / "AI_Models" / "4Sight_GTSRB_Model"
    / "traffic_sign_cnn_bundle.pth"
)

@st.cache_resource
def get_traffic_model():
    return load_traffic_sign_model(str(TRAFFIC_MODEL_PATH))


st.set_page_config(
    page_title="4Sight Smart Driving",
    page_icon="Ã°Å¸Å¡â€”",
    layout="wide"
)

# Professional 4Sight UI Design
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #080d18 0%, #111a2e 55%, #080d18 100%);
    color: #f1f5f9;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101a30, #080d18);
    border-right: 1px solid #263b60;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2 {
    color: #38bdf8;
}
h1, h2, h3 {
    color: #eaf4ff !important;
}
.stButton > button {
    background: linear-gradient(90deg, #0878e8, #2563eb);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 22px;
    font-weight: 600;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #2563eb, #0ea5e9);
    border: 1px solid #38bdf8;
    color: white;
}
[data-testid="stFileUploader"] {
    background: rgba(30, 45, 70, 0.55);
    border: 1px solid #334b70;
    border-radius: 12px;
    padding: 12px;
}
[data-testid="stMetric"] {
    background: rgba(30, 45, 70, 0.65);
    border: 1px solid #334b70;
    border-radius: 12px;
    padding: 15px;
}
div[data-testid="stAlert"] {
    border-radius: 10px;
}
hr {
    border-color: #29466f;
}
</style>
""", unsafe_allow_html=True)
# Main title
st.title("4Sight Smart Driving")
st.subheader("AI-Powered Smart Driving Monitoring System")

st.markdown("---")

# Sidebar
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a page:",
    [
        "Home",
        "AI Analysis",
        "Driver Drowsiness Analysis",
        "Seatbelt Detection",
        "Traffic Sign Recognition",
        "Lane Detection",
        "Vehicle Detection",
        "Safe Following Distance",
        "Users",
        "Reports",
        "About"
    ]
)

if page == "Home":
    st.header("Welcome to 4Sight!")
    st.write(
        "An intelligent driving monitoring system "
        "powered by Machine Learning and Computer Vision."
    )

    st.markdown("### Available AI Modules")

    col1, col2 = st.columns(2)

    with col1:
        st.info("Traffic Sign Recognition")
        st.info("Lane Detection")
    with col2:
        st.info("Seatbelt Detection")
        st.info("Vehicle Detection")
        st.info("Safe Following Distance")

    st.markdown("---")
    st.success("Welcome! Your Smart Driving System is ready.")

elif page == "AI Analysis":
    st.header("AI Analysis")
    st.write("Analyze driving images and videos using AI models.")
    st.success("All five AI modules are integrated and ready to use.")
    st.write("Choose an AI module from the sidebar to analyze driving images or videos.")


elif page == "Driver Drowsiness Analysis":

    st.header("Driver Drowsiness Analysis")

    st.write(
        "Analyze driver eye activity and detect "
        "possible prolonged eye closures."
    )

    uploaded_video = st.file_uploader(
        "Upload Driving Video",
        type=["mp4", "avi", "mov"],
        key="drowsiness_video"
    )

    if uploaded_video is not None:

        st.video(uploaded_video)

        if st.button("Analyze Video", type="primary"):

            progress_bar = st.progress(0)

            temp_video_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=os.path.splitext(
                        uploaded_video.name
                    )[1]
                ) as temp_file:

                    temp_file.write(
                        uploaded_video.getbuffer()
                    )

                    temp_video_path = temp_file.name

                with st.spinner(
                    "Analyzing driver eye activity..."
                ):

                    results_df, events_df, summary = analyze_video(
                        temp_video_path,
                        progress_callback=lambda value:
                            progress_bar.progress(value)
                    )

                progress_bar.progress(1.0)

                st.success("Video analysis completed!")

                st.subheader("Analysis Summary")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Frames Analyzed",
                    summary["Total Frames"]
                )

                col2.metric(
                    "Face Detection Rate",
                    f'{summary["Face Detection Rate"]}%'
                )

                col3.metric(
                    "Closure Events",
                    summary["Closure Events"]
                )

                st.metric(
                    "Longest Eye Closure",
                    f'{summary["Longest Closure"]} seconds'
                )

                st.subheader("Eye Activity")

                ear_chart = results_df.dropna(
                    subset=["EAR"]
                ).set_index("Time")[["EAR"]]

                if not ear_chart.empty:
                    st.line_chart(ear_chart)

                st.subheader("Detected Eye Closure Events")

                if not events_df.empty:
                    st.dataframe(
                        events_df,
                        use_container_width=True
                    )
                else:
                    st.info(
                        "No eye closure events were detected."
                    )

                st.subheader("Detailed Frame Results")

                st.dataframe(
                    results_df,
                    use_container_width=True
                )

                st.info(
                    "This is an EAR-based preliminary analysis. "
                    "Prolonged eye closure is an indicator, not "
                    "a confirmed diagnosis of driver drowsiness."
                )

            except Exception as e:
                st.error(f"Analysis failed: {e}")

            finally:
                if temp_video_path and os.path.exists(
                    temp_video_path
                ):
                    os.remove(temp_video_path)


elif page == "Lane Detection":
    import sqlite3
    from datetime import datetime

    st.header("Lane Detection")
    st.write("Upload a road image to detect lane markings.")

    db_path = ROOT / "4sight_users.db"

    with sqlite3.connect(db_path) as conn:
        drivers = conn.execute(
            "SELECT id, name FROM drivers ORDER BY name"
        ).fetchall()

    selected_driver_id = None

    if drivers:
        driver_options = {
            f"{name} (ID: {driver_id})": driver_id
            for driver_id, name in drivers
        }
        selected_driver = st.selectbox(
            "Select Driver",
            list(driver_options.keys()),
            key="lane_report_driver"
        )
        selected_driver_id = driver_options[selected_driver]
    else:
        st.warning("No registered drivers. Add a driver in the Users page to save reports.")

    uploaded_lane_image = st.file_uploader(
        "Upload Road Image",
        type=["jpg", "jpeg", "png"],
        key="lane_image"
    )

    if uploaded_lane_image is not None:
        lane_image = Image.open(uploaded_lane_image).convert("RGB")

        if st.button("Detect Lanes"):
            from Lane_Detection.lane_detector import load_lane_model, predict_lane

            lane_model_path = ROOT / "AI_Models" / "Lane_Detection" / "lane_unet_best.pth"

            with st.spinner("Loading lane detection model..."):
                if not lane_model_path.exists():
                    st.error("Lane detection model file was not found.")
                    st.stop()

                lane_model, lane_device = load_lane_model(str(lane_model_path))

            with st.spinner("Detecting lanes..."):
                lane_mask, lane_overlay = predict_lane(
                    lane_model, lane_device, lane_image, threshold=0.70
                )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(lane_image, caption="Original Image", use_container_width=True)

            with col2:
                st.image(lane_mask, caption="Predicted Lane Mask", use_container_width=True)

            with col3:
                st.image(lane_overlay, caption="Lane Detection Result", use_container_width=True)

            st.success("Lane Detection Completed!")

            st.info("Lane detection is an AI estimate and may not identify every lane correctly.")

            if selected_driver_id is not None:
                with sqlite3.connect(db_path) as conn:
                    conn.execute("""
                        INSERT INTO driving_reports
                        (driver_id, module, result, notes, report_date)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        selected_driver_id,
                        "Lane Detection",
                        "Lane detection completed successfully",
                        f"Automatically saved from image: {uploaded_lane_image.name}",
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ))

                st.success("Report saved automatically in Reports!")
            else:
                st.warning("Detection completed, but no report was saved because no driver is registered.")

elif page == "Traffic Sign Recognition":
    import sqlite3
    from datetime import datetime

    st.title("Traffic Sign Recognition")
    st.write("Upload a traffic sign image to recognize it.")

    db_path = ROOT / "4sight_users.db"

    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS drivers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER,
                experience INTEGER
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS driving_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                driver_id INTEGER NOT NULL,
                module TEXT NOT NULL,
                result TEXT NOT NULL,
                notes TEXT,
                report_date TEXT NOT NULL,
                score INTEGER DEFAULT 0,
                FOREIGN KEY (driver_id) REFERENCES drivers(id)
            )
        """)
        columns = [
            row[1] for row in
            conn.execute("PRAGMA table_info(driving_reports)").fetchall()
        ]
        if "score" not in columns:
            conn.execute(
                "ALTER TABLE driving_reports ADD COLUMN score INTEGER DEFAULT 0"
            )
        drivers = conn.execute(
            "SELECT id, name FROM drivers ORDER BY name"
        ).fetchall()

    selected_driver_id = None
    if drivers:
        driver_options = {
            f"{name} (ID: {driver_id})": driver_id
            for driver_id, name in drivers
        }
        selected_driver = st.selectbox(
            "Select Driver for Report",
            list(driver_options.keys()),
            key="traffic_sign_report_driver"
        )
        selected_driver_id = driver_options[selected_driver]
    else:
        st.warning("No registered drivers found. Register a driver to save reports.")

    uploaded_sign = st.file_uploader(
        "Upload Traffic Sign Image",
        type=["jpg", "jpeg", "png"],
        key="traffic_sign_image"
    )

    if uploaded_sign is not None:
        sign_image = Image.open(uploaded_sign).convert("RGB")
        st.image(sign_image, caption="Uploaded Traffic Sign", width=350)

        if st.button("Recognize Traffic Sign"):
            try:
                with st.spinner("Analyzing traffic sign..."):
                    model, bundle, device = get_traffic_model()

                    resized = sign_image.resize(tuple(bundle["input_size"]))
                    image_array = np.asarray(resized).astype("float32") / 255.0
                    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)

                    mean = torch.tensor(bundle["normalization_mean"]).view(3, 1, 1)
                    std = torch.tensor(bundle["normalization_std"]).view(3, 1, 1)

                    image_tensor = (
                        (image_tensor - mean) / std
                    ).unsqueeze(0).to(device)

                    with torch.no_grad():
                        outputs = model(image_tensor)
                        probabilities = torch.softmax(outputs, dim=1)[0]

                    top_values, top_indices = torch.topk(probabilities, 3)

                    class_names = bundle["class_names"]
                    predicted_index = top_indices[0].item()
                    predicted_sign = class_names[predicted_index]
                    confidence = top_values[0].item() * 100

                st.success("Traffic Sign Recognition Completed!")
                st.subheader("Prediction")
                st.write(predicted_sign)
                st.metric("Confidence", f"{confidence:.2f}%")

                st.subheader("Top 3 Predictions")
                for value, index in zip(top_values, top_indices):
                    st.write(
                        f"{class_names[index.item()]} - {value.item() * 100:.2f}%"
                    )

                if selected_driver_id is not None:
                    top_predictions = "; ".join(
                        f"{class_names[index.item()]}: {value.item() * 100:.2f}%"
                        for value, index in zip(top_values, top_indices)
                    )
                    with sqlite3.connect(db_path) as conn:
                        conn.execute(
                            """
                            INSERT INTO driving_reports
                            (driver_id, module, result, notes, report_date)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (
                                selected_driver_id,
                                "Traffic Sign Recognition",
                                f"{predicted_sign} | Confidence: {confidence:.2f}%",
                                f"Top predictions: {top_predictions}. Image: {uploaded_sign.name}",
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            )
                        )
                    st.success("Report saved automatically in Reports!")
                else:
                    st.warning("Recognition completed, but no report was saved because no driver is registered.")

            except Exception as error:
                st.error(f"Traffic sign recognition failed: {error}")

elif page == "Seatbelt Detection":
    from ultralytics import YOLO
    import tempfile
    import sqlite3
    from datetime import datetime

    st.header("Seatbelt Detection")
    st.write("Upload a car image to detect seatbelts.")

    db_path = ROOT / "4sight_users.db"

    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS drivers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER NOT NULL,
                experience INTEGER NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS driving_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                driver_id INTEGER NOT NULL,
                module TEXT NOT NULL,
                result TEXT NOT NULL,
                notes TEXT,
                report_date TEXT NOT NULL,
                score INTEGER DEFAULT 0,
                FOREIGN KEY (driver_id) REFERENCES drivers(id)
            )
        """)

        columns = [
            row[1] for row in
            conn.execute("PRAGMA table_info(driving_reports)").fetchall()
        ]

        if "score" not in columns:
            conn.execute(
                "ALTER TABLE driving_reports ADD COLUMN score INTEGER DEFAULT 0"
            )

        drivers = conn.execute(
            "SELECT id, name FROM drivers ORDER BY name"
        ).fetchall()

    selected_driver_id = None

    if drivers:
        driver_options = {
            f"{name} (ID: {driver_id})": driver_id
            for driver_id, name in drivers
        }

        selected_driver = st.selectbox(
            "Select Driver for Report",
            list(driver_options.keys()),
            key="seatbelt_report_driver"
        )
        selected_driver_id = driver_options[selected_driver]
    else:
        st.info("No registered drivers yet. You can still run detection, but automatic report saving requires a registered driver.")

    model_path = ROOT / "Datasets" / "Seatbelt.v1i.yolov8" / "runs" / "seatbelt_detection" / "weights" / "best.pt"

    uploaded_file = st.file_uploader(
        "Upload a car image",
        type=["jpg", "jpeg", "png"],
        key="seatbelt_image"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Detect Seatbelt"):
            if not model_path.exists():
                st.error(f"Model file not found: {model_path}")
            else:
                temp_path = None

                try:
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                        image.save(temp_file.name)
                        temp_path = temp_file.name

                    model = YOLO(str(model_path))
                    results = model.predict(source=temp_path, conf=0.25)

                    result_image = results[0].plot()
                    st.subheader("Detection Result")
                    st.image(result_image[:, :, ::-1], use_container_width=True)

                    count = len(results[0].boxes)
                    confidences = [
                        float(box.conf[0]) * 100
                        for box in results[0].boxes
                    ]

                    st.write(f"Detected Seatbelts: {count}")

                    if count > 0:
                        st.success("\u062a\u0645 \u0627\u0643\u062a\u0634\u0627\u0641 \u062d\u0632\u0627\u0645 \u0623\u0645\u0627\u0646 \u0641\u064a \u0627\u0644\u0635\u0648\u0631\u0629 \u0628\u0646\u062c\u0627\u062d! \U0001F397\uFE0F")

                        for confidence in confidences:
                            st.metric("Confidence", f"{confidence:.2f}%")
                    else:
                        st.warning("\u0644\u0645 \u064a\u062a\u0645 \u0627\u0643\u062a\u0634\u0627\u0641 \u062d\u0632\u0627\u0645 \u0623\u0645\u0627\u0646 \u0641\u064a \u0627\u0644\u0635\u0648\u0631\u0629.")

                    if selected_driver_id is not None:
                        confidence_text = ", ".join(
                            f"{value:.2f}%" for value in confidences
                        ) if confidences else "No detection"

                        result_summary = (
                            f"Detected Seatbelts: {count} | "
                            f"Confidence: {confidence_text}"
                        )

                        with sqlite3.connect(db_path) as conn:
                            conn.execute(
                                """
                                INSERT INTO driving_reports
                                (driver_id, module, result, notes, report_date)
                                VALUES (?, ?, ?, ?, ?)
                                """,
                                (
                                    selected_driver_id,
                                    "Seatbelt Detection",
                                    result_summary,
                                    f"Automatically saved from image: {uploaded_file.name}",
                                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                )
                            )

                        st.success("Report saved automatically in Reports!")

                except Exception as error:
                    st.error(f"Detection failed: {error}")

                finally:
                    if temp_path and Path(temp_path).exists():
                        Path(temp_path).unlink()

elif page == "Vehicle Detection":
    import sqlite3
    from datetime import datetime
    from ultralytics import YOLO

    st.header("Vehicle Detection")
    st.write("Upload a road image to detect cars, motorcycles, buses, and trucks.")

    db_path = ROOT / "4sight_users.db"

    with sqlite3.connect(db_path) as conn:
        drivers = conn.execute(
            "SELECT id, name FROM drivers ORDER BY name"
        ).fetchall()

    selected_driver_id = None

    if drivers:
        driver_options = {
            f"{name} (ID: {driver_id})": driver_id
            for driver_id, name in drivers
        }
        selected_driver = st.selectbox(
            "Select Driver",
            list(driver_options.keys()),
            key="vehicle_report_driver"
        )
        selected_driver_id = driver_options[selected_driver]
    else:
        st.warning("No registered drivers. Add a driver in the Users page to save reports.")

    uploaded_vehicle_image = st.file_uploader(
        "Upload Vehicle Image",
        type=["jpg", "jpeg", "png"],
        key="vehicle_image"
    )

    if uploaded_vehicle_image is not None:
        vehicle_image = Image.open(uploaded_vehicle_image).convert("RGB")
        st.image(vehicle_image, caption="Uploaded Road Image", width=500)

        if st.button("Detect Vehicles"):
            model_path = ROOT / "yolo11n.pt"

            if not model_path.exists():
                st.error("YOLO model file yolo11n.pt was not found.")
                st.stop()

            try:
                with st.spinner("Loading vehicle detection model..."):
                    vehicle_model = YOLO(str(model_path))

                with st.spinner("Detecting vehicles..."):
                    results = vehicle_model.predict(
                        source=vehicle_image,
                        conf=0.35,
                        classes=[2, 3, 5, 7],
                        verbose=False
                    )

                result = results[0]
                boxes = result.boxes
                count = len(boxes)

                vehicle_names = []
                confidences = []

                for box in boxes:
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item()) * 100
                    vehicle_names.append(vehicle_model.names[class_id])
                    confidences.append(confidence)

                annotated_image = result.plot()
                annotated_image = annotated_image[:, :, ::-1]
                st.image(
                    annotated_image,
                    caption="Vehicle Detection Result",
                    use_container_width=True
                )

                st.metric("Detected Vehicles", count)

                if count > 0:
                    st.success("Vehicle detection completed successfully!")
                    for name, confidence in zip(vehicle_names, confidences):
                        st.write(f"{name}: {confidence:.2f}% confidence")
                else:
                    st.info("No vehicles were detected in this image.")

                if selected_driver_id is not None:
                    if count > 0:
                        summary = ", ".join(
                            f"{name} ({confidence:.2f}%)"
                            for name, confidence in zip(vehicle_names, confidences)
                        )
                        result_summary = f"Detected Vehicles: {count} | {summary}"
                    else:
                        result_summary = "No vehicles detected"

                    with sqlite3.connect(db_path) as conn:
                        conn.execute(
                            """
                            INSERT INTO driving_reports
                            (driver_id, module, result, notes, report_date)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (
                                selected_driver_id,
                                "Vehicle Detection",
                                result_summary,
                                f"Automatically saved from image: {uploaded_vehicle_image.name}",
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            )
                        )

                    st.success("Report saved automatically in Reports!")
                else:
                    st.warning("Detection completed, but no report was saved because no driver is registered.")

            except Exception as error:
                st.error(f"Vehicle detection failed: {error}")

elif page == "Safe Following Distance":
    import sqlite3
    import cv2
    import tempfile
    from datetime import datetime
    from ultralytics import YOLO

    st.header("Safe Following Distance")
    st.write(
        "Upload a driving video to estimate visual vehicle proximity "
        "and identify possible following-distance risks."
    )

    st.warning(
        "This is an approximate visual indicator based on vehicle bounding-box size. "
        "It does not measure actual distance in meters or replace a real safety system."
    )

    db_path = ROOT / "4sight_users.db"

    with sqlite3.connect(db_path) as conn:
        drivers = conn.execute(
            "SELECT id, name FROM drivers ORDER BY name"
        ).fetchall()

    selected_driver_id = None

    if drivers:
        driver_options = {
            f"{name} (ID: {driver_id})": driver_id
            for driver_id, name in drivers
        }
        selected_driver = st.selectbox(
            "Select Driver",
            list(driver_options.keys()),
            key="safe_distance_driver"
        )
        selected_driver_id = driver_options[selected_driver]
    else:
        st.warning("No registered drivers. Add a driver in the Users page to save reports.")

    uploaded_video = st.file_uploader(
        "Upload Driving Video",
        type=["mp4", "avi", "mov"],
        key="safe_distance_video"
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button("Analyze Following Distance"):
            temp_video_path = None
            cap = None

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=Path(uploaded_video.name).suffix
                ) as temp_file:
                    temp_file.write(uploaded_video.getbuffer())
                    temp_video_path = temp_file.name

                model_path = ROOT / "yolo11n.pt"

                if not model_path.exists():
                    st.error("YOLO model file yolo11n.pt was not found.")
                    st.stop()

                with st.spinner("Loading vehicle detection model..."):
                    vehicle_model = YOLO(str(model_path))

                cap = cv2.VideoCapture(temp_video_path)

                if not cap.isOpened():
                    st.error("Could not open this video. Please try another video.")
                    st.stop()

                fps = cap.get(cv2.CAP_PROP_FPS)
                sample_interval = max(1, round(fps)) if fps > 0 else 10

                frame_index = 0
                analyzed_frames = 0
                detected_frames = 0
                max_area_ratio = 0.0
                best_frame = None
                best_vehicle_name = None

                progress = st.progress(0)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                with st.spinner("Analyzing video frames..."):
                    while True:
                        success, frame = cap.read()

                        if not success:
                            break

                        if frame_index % sample_interval == 0:
                            results = vehicle_model.predict(
                                source=frame,
                                conf=0.35,
                                classes=[2, 3, 5, 7],
                                verbose=False
                            )

                            analyzed_frames += 1
                            result = results[0]
                            boxes = result.boxes

                            if boxes is not None and len(boxes) > 0:
                                detected_frames += 1
                                height, width = frame.shape[:2]

                                for box in boxes:
                                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                                    box_area = max(0, x2 - x1) * max(0, y2 - y1)
                                    area_ratio = box_area / (width * height)

                                    if area_ratio > max_area_ratio:
                                        max_area_ratio = area_ratio
                                        class_id = int(box.cls[0].item())
                                        best_vehicle_name = vehicle_model.names[class_id]
                                        best_frame = result.plot()

                        frame_index += 1

                        if total_frames > 0 and frame_index % 10 == 0:
                            progress.progress(
                                min(frame_index / total_frames, 1.0)
                            )

                cap.release()
                cap = None
                progress.progress(1.0)

                if analyzed_frames == 0:
                    st.error("No video frames could be analyzed.")
                    st.stop()

                st.success("Following distance analysis completed!")

                st.metric("Analyzed Frames", analyzed_frames)
                st.metric("Frames With Vehicles", detected_frames)

                if best_frame is None:
                    risk_level = "No vehicles detected"
                    result_summary = (
                        f"No vehicles detected in {analyzed_frames} analyzed frames"
                    )
                    st.info("No vehicles were detected in the sampled video frames.")
                else:
                    area_percent = max_area_ratio * 100

                    if max_area_ratio >= 0.20:
                        risk_level = "Danger"
                        st.error("DANGER: A vehicle appears very large in the frame. Check the following distance.")
                    elif max_area_ratio >= 0.10:
                        risk_level = "Warning"
                        st.warning("WARNING: A vehicle appears relatively close. Increase the following distance.")
                    else:
                        risk_level = "Safe"
                        st.success("SAFE: No high visual-proximity warning was detected in the sampled frames.")

                    st.metric("Visual Proximity Indicator", f"{area_percent:.2f}%")
                    st.write(f"Closest-looking detected vehicle: {best_vehicle_name}")
                    st.image(
                        cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB),
                        caption="Frame with the largest detected vehicle",
                        use_container_width=True
                    )

                    result_summary = (
                        f"{risk_level} | Largest vehicle box: {area_percent:.2f}% "
                        f"| Vehicle: {best_vehicle_name} "
                        f"| Analyzed frames: {analyzed_frames}"
                    )

                if selected_driver_id is not None:
                    with sqlite3.connect(db_path) as conn:
                        conn.execute(
                            """
                            INSERT INTO driving_reports
                            (driver_id, module, result, notes, report_date)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (
                                selected_driver_id,
                                "Safe Following Distance",
                                result_summary,
                                (
                                    f"Automatically analyzed video: {uploaded_video.name}. "
                                    "Visual estimate only; not a real distance measurement."
                                ),
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            )
                        )

                    st.success("Report saved automatically in Reports!")
                else:
                    st.warning("Analysis completed, but no report was saved because no driver is registered.")

            except Exception as error:
                st.error(f"Following distance analysis failed: {error}")

            finally:
                if cap is not None:
                    cap.release()

                if temp_video_path and Path(temp_video_path).exists():
                    Path(temp_video_path).unlink()

elif page == "Users":
    import sqlite3

    st.header("Driver Management")
    st.write("Register and manage driver profiles.")

    db_path = ROOT / "4sight_users.db"

    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS drivers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER NOT NULL,
                experience INTEGER NOT NULL
            )
        """)

    st.subheader("Register a New Driver")

    with st.form("add_driver_form"):
        driver_name = st.text_input("Driver Name")
        driver_age = st.number_input("Age", min_value=16, max_value=100, value=18)
        driver_experience = st.number_input(
            "Driving Experience (Years)",
            min_value=0,
            max_value=80,
            value=0
        )
        submitted = st.form_submit_button("Add Driver")

        if submitted:
            if not driver_name.strip():
                st.error("Please enter the driver's name.")
            else:
                with sqlite3.connect(db_path) as conn:
                    conn.execute(
                        "INSERT INTO drivers (name, age, experience) VALUES (?, ?, ?)",
                        (driver_name.strip(), driver_age, driver_experience)
                    )
                st.success("Driver registered successfully!")
                st.rerun()

    st.markdown("---")
    st.subheader("Registered Drivers")

    with sqlite3.connect(db_path) as conn:
        drivers = conn.execute(
            "SELECT id, name, age, experience FROM drivers ORDER BY id DESC"
        ).fetchall()

    if drivers:
        driver_data = [
            {
                "ID": row[0],
                "Driver Name": row[1],
                "Age": row[2],
                "Experience (Years)": row[3]
            }
            for row in drivers
        ]

        st.dataframe(driver_data, use_container_width=True)
        st.info(f"Total registered drivers: {len(drivers)}")

        st.markdown("---")
        st.subheader("Edit or Delete a Driver")

        driver_options = {
            f"{row[1]} (ID: {row[0]})": row[0]
            for row in drivers
        }

        selected_label = st.selectbox(
            "Select a driver",
            list(driver_options.keys())
        )
        selected_id = driver_options[selected_label]
        selected_driver = next(
            row for row in drivers if row[0] == selected_id
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Edit Driver")

            with st.form("edit_driver_form"):
                updated_name = st.text_input(
                    "Driver Name",
                    value=selected_driver[1],
                    key=f"edit_name_{selected_id}"
                )
                updated_age = st.number_input(
                    "Age",
                    min_value=16,
                    max_value=100,
                    value=int(selected_driver[2]),
                    key=f"edit_age_{selected_id}"
                )
                updated_experience = st.number_input(
                    "Driving Experience (Years)",
                    min_value=0,
                    max_value=80,
                    value=int(selected_driver[3]),
                    key=f"edit_experience_{selected_id}"
                )

                save_changes = st.form_submit_button("Save Changes")

                if save_changes:
                    if not updated_name.strip():
                        st.error("Please enter the driver's name.")
                    else:
                        with sqlite3.connect(db_path) as conn:
                            conn.execute(
                                "UPDATE drivers SET name=?, age=?, experience=? WHERE id=?",
                                (
                                    updated_name.strip(),
                                    updated_age,
                                    updated_experience,
                                    selected_id
                                )
                            )
                        st.success("Driver updated successfully!")
                        st.rerun()

        with col2:
            st.markdown("#### Delete Driver")
            confirm_delete = st.checkbox(
                "I confirm that I want to delete this driver.",
                key=f"confirm_delete_{selected_id}"
            )

            if st.button(
                "Delete Selected Driver",
                disabled=not confirm_delete,
                key=f"delete_driver_{selected_id}"
            ):
                with sqlite3.connect(db_path) as conn:
                    conn.execute(
                        "DELETE FROM drivers WHERE id=?",
                        (selected_id,)
                    )
                st.success("Driver deleted successfully!")
                st.rerun()

    else:
        st.info("No drivers registered yet. Add your first driver above.")
elif page == "Reports":
    import sqlite3
    import csv
    import io
    import pandas as pd
    from datetime import datetime

    st.header("Driving Reports")
    st.write("Record and review driving analysis sessions.")

    db_path = ROOT / "4sight_users.db"

    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS driving_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                driver_id INTEGER NOT NULL,
                module TEXT NOT NULL,
                result TEXT NOT NULL,
                notes TEXT,
                report_date TEXT NOT NULL,
                FOREIGN KEY (driver_id) REFERENCES drivers(id)
            )
        """)

        report_columns = [
            row[1] for row in conn.execute(
                "PRAGMA table_info(driving_reports)"
            ).fetchall()
        ]

        if "score" not in report_columns:
            conn.execute(
                "ALTER TABLE driving_reports ADD COLUMN score INTEGER DEFAULT 0"
            )

        drivers = conn.execute(
            "SELECT id, name FROM drivers ORDER BY name"
        ).fetchall()

        total_reports = conn.execute(
            "SELECT COUNT(*) FROM driving_reports"
        ).fetchone()[0]

        total_drivers = conn.execute(
            "SELECT COUNT(*) FROM drivers"
        ).fetchone()[0]

    st.subheader("Reports Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Registered Drivers", total_drivers)

    with col2:
        st.metric("Recorded Analysis Sessions", total_reports)

    st.markdown("---")
    st.subheader("Record a Driving Analysis")

    if drivers:
        driver_options = {
            f"{name} (ID: {driver_id})": driver_id
            for driver_id, name in drivers
        }

        with st.form("add_report_form"):
            selected_driver = st.selectbox(
                "Select Driver",
                list(driver_options.keys())
            )

            selected_module = st.selectbox(
                "AI Module",
                [
                    "Traffic Sign Recognition",
                    "Lane Detection",
                    "Seatbelt Detection",
                    "Vehicle Detection",
                    "Safe Following Distance"
                ]
            )

            result_summary = st.text_area(
                "Analysis Result Summary",
                placeholder="Describe the result of the analysis..."
            )

            report_notes = st.text_area(
                "Additional Notes (Optional)",
                placeholder="Add any additional observations..."
            )

            driving_score = st.slider(
                "Driving Score (0-100)",
                min_value=0,
                max_value=100,
                value=80
            )

            save_report = st.form_submit_button("Save Report")

            if save_report:
                if not result_summary.strip():
                    st.error("Please enter an analysis result summary.")
                else:
                    with sqlite3.connect(db_path) as conn:
                        conn.execute(
                            """
                            INSERT INTO driving_reports
                            (driver_id, module, result, notes, report_date, score)
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (
                                driver_options[selected_driver],
                                selected_module,
                                result_summary.strip(),
                                report_notes.strip(),
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                driving_score
                            )
                        )

                    st.success("Driving report saved successfully!")
                    st.rerun()
    else:
        st.info("Please register a driver in the Users page before recording reports.")

    st.markdown("---")
    st.subheader("Driving Reports History")

    with sqlite3.connect(db_path) as conn:
        reports = conn.execute(
            """
            SELECT r.id, d.name, r.module, r.result, r.notes, r.report_date, r.score
            FROM driving_reports r
            LEFT JOIN drivers d ON r.driver_id = d.id
            ORDER BY r.id DESC
            """
        ).fetchall()

    if reports:
        st.markdown("---")
        st.subheader("Driver Performance Dashboard")

        with sqlite3.connect(db_path) as conn:
            scores = [
                row[0] for row in conn.execute(
                    "SELECT score FROM driving_reports WHERE score > 0"
                ).fetchall()
            ]

        if scores:
            average_score = round(sum(scores) / len(scores), 1)
            highest_score = max(scores)
            lowest_score = min(scores)

            if average_score >= 90:
                performance_label = "Excellent"
            elif average_score >= 75:
                performance_label = "Good"
            else:
                performance_label = "Needs Improvement"

            score_col1, score_col2, score_col3 = st.columns(3)

            with score_col1:
                st.metric("Average Driving Score", f"{average_score}/100")

            with score_col2:
                st.metric("Highest Score", f"{highest_score}/100")

            with score_col3:
                st.metric("Lowest Score", f"{lowest_score}/100")

            st.info(
                f"Overall Performance: {performance_label} "
                f"| Evaluated Reports: {len(scores)}"
            )
        else:
            st.info("No driving scores available yet. Record a report with a score.")

        st.markdown("---")
        st.subheader("Reports Analytics")

        with sqlite3.connect(db_path) as conn:
            module_stats = conn.execute(
                """
                SELECT module, COUNT(*)
                FROM driving_reports
                GROUP BY module
                ORDER BY COUNT(*) DESC
                """
            ).fetchall()

            driver_stats = conn.execute(
                """
                SELECT COALESCE(d.name, 'Deleted Driver'), COUNT(*)
                FROM driving_reports r
                LEFT JOIN drivers d ON r.driver_id = d.id
                GROUP BY d.name
                ORDER BY COUNT(*) DESC
                """
            ).fetchall()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Reports by AI Module")
            module_chart = pd.DataFrame(
                module_stats,
                columns=["AI Module", "Reports"]
            ).set_index("AI Module")
            st.bar_chart(module_chart)

        with col2:
            st.markdown("#### Reports by Driver")
            driver_chart = pd.DataFrame(
                driver_stats,
                columns=["Driver", "Reports"]
            ).set_index("Driver")
            st.bar_chart(driver_chart)

        report_data = [
            {
                "Report ID": row[0],
                "Driver": row[1] or "Deleted Driver",
                "AI Module": row[2],
                "Result Summary": row[3],
                "Notes": row[4] or "-",
                "Date": row[5],
                "Driving Score": row[6]
            }
            for row in reports
        ]

        st.dataframe(report_data, use_container_width=True)

        csv_buffer = io.StringIO(newline="")
        csv_writer = csv.writer(csv_buffer)

        csv_writer.writerow([
            "Report ID",
            "Driver",
            "AI Module",
            "Result Summary",
            "Notes",
            "Date",
            "Driving Score"
        ])

        csv_writer.writerows(reports)

        st.download_button(
            label="Download Reports as CSV",
            data=csv_buffer.getvalue().encode("utf-8-sig"),
            file_name="4Sight_Driving_Reports.csv",
            mime="text/csv"
        )
        st.info(f"Total reports: {len(reports)}")
    else:
        st.info("No driving reports recorded yet.")
elif page == "About":
    st.header("About 4Sight Smart Driving")

    st.markdown("""
    ### AI-Powered Smart Driving Monitoring System

    **4Sight** is an experimental intelligent driving monitoring
    system that uses Artificial Intelligence and Computer Vision
    to analyze road scenes and identify important driving conditions.
    """)

    st.markdown("---")
    st.subheader("Project Objectives")

    st.markdown("""
    - Improve driving awareness through AI-based analysis.
    - Recognize traffic signs.
    - Detect road lane markings.
    - Identify visible seatbelt usage.
    - Detect vehicles in road scenes.
    - Estimate safe-following-distance risks.
    """)

    st.markdown("---")
    st.subheader("AI Modules")

    modules = [
        ("Traffic Sign Recognition", "CNN model trained to classify traffic signs."),
        ("Lane Detection", "U-Net model that identifies lane markings in road images."),
        ("Seatbelt Detection", "YOLO model that detects visible seatbelts."),
        ("Vehicle Detection", "YOLO model that identifies vehicles in images and videos."),
        ("Safe Following Distance", "Video analysis that estimates following risk using vehicle bounding-box size. This is an approximate estimate, not an actual distance measurement.")
    ]

    for name, description in modules:
        with st.container(border=True):
            st.markdown(f"**{name}**")
            st.write(description)

    st.markdown("---")
    st.subheader("Technologies Used")

    st.markdown("""
    **Python** | **Streamlit** | **PyTorch** | **YOLO** |
    **OpenCV** | **NumPy** | **PIL**
    """)

    st.markdown("---")
    st.subheader("Important Note")

    st.warning(
        "4Sight is an experimental educational project. "
        "Its predictions are estimates and should not replace "
        "driver attention, road regulations, or professional safety systems."
    )









