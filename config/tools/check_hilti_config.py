"""Static configuration/source-calibration checks. Does not build or run SLAM."""
from pathlib import Path
import binascii
import json
import re
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config/hilti_robot_estimator.yaml"
REFERENCE = ROOT / "config/reference/hilti2023_robot"
checks = 0


def check(condition, message):
    global checks
    checks += 1
    if not condition:
        raise AssertionError(message)


def load(path):
    text = path.read_text(encoding="utf-8")
    if not text.startswith("%YAML:"):
        # The source archive uses valid YAML indentation-less block sequences;
        # OpenCV requires them to be indented deeper than their owning map key.
        text = "%YAML:1.0\n" + re.sub(r"(?m)^  - ", "    - ", text)
    file = cv2.FileStorage(text, cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
    check(file.isOpened(), f"Cannot parse {path}")
    return file


def sequence(node):
    check(node.isSeq(), "Expected numeric sequence")
    return np.array([node.at(i).real() for i in range(node.size())])


def source_matrix(node):
    return np.array([sequence(node.at(i)) for i in range(node.size())])


def valid_boolean_node(node):
    # Match ViParametersReader's accepted node types, not just key existence.
    # OpenCV can interpret a colon inside an unquoted scalar's inline comment
    # as a map delimiter, even though the source appears to contain "true".
    if node.isInt():
        return True
    if not node.isString():
        return False
    words = node.string().split()
    return bool(words) and words[0].lower() in {"true", "false", "yes", "no", "y", "n", "on", "off"}


def rigid_inverse(matrix):
    result = np.eye(4)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return result


def quaternion_transform(node):
    x, y, z, w = [node.getNode(k).real() for k in ("qx", "qy", "qz", "qw")]
    check(abs(x*x+y*y+z*z+w*w-1) < 1e-10, "Invalid official quaternion")
    result = np.eye(4)
    result[:3, :3] = [
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ]
    result[:3, 3] = [node.getNode(k).real() for k in ("x", "y", "z")]
    return result


def valid_transform(matrix, label):
    check(matrix.shape == (4, 4) and np.isfinite(matrix).all(), label + " shape/finite")
    check(np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-12, rtol=0), label + " last row")
    rotation = matrix[:3, :3]
    check(np.linalg.norm(rotation.T @ rotation - np.eye(3)) < 1e-10, label + " orthogonality")
    check(abs(np.linalg.det(rotation)-1) < 1e-10, label + " proper rotation")


def main():
    receipt = json.loads((REFERENCE / "SOURCE.json").read_text(encoding="utf-8"))
    for filename, properties in receipt["files"].items():
        data = (REFERENCE / filename).read_bytes()
        check(len(data) == properties["bytes"], filename + " size differs from archive")
        check(format(binascii.crc32(data), "08x") == properties["zip_crc32"], filename + " CRC differs from archive")

    target = load(CONFIG)
    source_cameras = load(REFERENCE / "calibration_cams.yaml")
    source_imu = load(REFERENCE / "calibration_imu.yaml")
    source_lidar = load(REFERENCE / "calibration_lidar.yaml")
    source_quaternions = load(REFERENCE / "calibration.yaml")
    cameras = target.getNode("cameras")
    check(cameras.isSeq() and cameras.size() == 4, "Exactly four cameras required")
    for index, (direction, source_id) in enumerate(zip(("front", "left", "rear", "right"), (0, 2, 4, 6))):
        camera = cameras.at(index)
        source = source_cameras.getNode("cam" + str(source_id))
        topic = f"/oak_cam_{direction}/left/image_raw"
        check(camera.getNode("rostopic").string() == source.getNode("rostopic").string() == topic, "Wrong camera topic/order")
        check(camera.getNode("source_camera").string() == "cam" + str(source_id), "Wrong original camera ID")
        check(camera.getNode("cam_model").string() == source.getNode("camera_model").string() == "pinhole", "Wrong camera model")
        check(camera.getNode("distortion_type").string() == source.getNode("distortion_model").string() == "equidistant", "Wrong distortion model")
        check(camera.getNode("camera_type").string() == "gray", "Raw left images are mono8")
        check(camera.getNode("slam_use").string() == "cgmli", "Selected camera disabled")
        check(np.array_equal(sequence(camera.getNode("image_dimension")), sequence(source.getNode("resolution"))), "Resolution mismatch")
        check(np.array_equal(np.r_[sequence(camera.getNode("focal_length")), sequence(camera.getNode("principal_point"))], sequence(source.getNode("intrinsics"))), "Intrinsics mismatch")
        check(np.array_equal(sequence(camera.getNode("distortion_coefficients")), sequence(source.getNode("distortion_coeffs"))), "Distortion mismatch")
        transform = sequence(camera.getNode("T_SC")).reshape(4, 4)
        valid_transform(transform, f"camera {index} T_SC")
        check(np.allclose(transform, rigid_inverse(source_matrix(source.getNode("T_here_imu0"))), atol=1e-12, rtol=0), "Extrinsic direction/inversion mismatch")
        check(np.allclose(transform, quaternion_transform(source_quaternions.getNode(f"T_imu0_cam{source_id}")), atol=1e-10, rtol=0), "Official quaternion cross-check mismatch")

    # Derive required top-level scalar keys from the actual C++ reader, so a
    # future schema addition cannot silently leave this configuration incomplete.
    code = (ROOT / "src/common/ViParametersReader.cpp").read_text(encoding="utf-8").split("// done!")[0]
    pattern = r'parseEntry\(file\["([^"\n]+)"\](?:\["([^"\n]+)"\])?,\s*"([^"\n]+)"'
    for section, child, key in re.findall(pattern, code):
        parent = target.getNode(section)
        if child:
            parent = parent.getNode(child)
        check(not parent.getNode(key).empty(), f"C++ reader requires {section}/{child}/{key}")

    for variable, section in (("frontendNode", "frontend_parameters"), ("estimatorNode", "estimator_parameters"), ("windowNode", "estimator_parameters")):
        for key in re.findall(r'parseEntry\(' + variable + r',\s*"([^"\n]+)"', code):
            check(not target.getNode(section).getNode(key).empty(), f"Missing {section}/{key}")
    window = target.getNode("estimator_parameters").getNode("window_size")
    check(window.isInt() and window.real() == 10, "Hilti local window must retain exactly 10 frame states")
    lightweight_limit = target.getNode("estimator_parameters").getNode("max_consecutive_lightweight_frames")
    check(lightweight_limit.isInt() and lightweight_limit.real() >= 0,
          "max_consecutive_lightweight_frames must be a nonnegative integer")
    new_imu_gate = target.getNode("estimator_parameters").getNode("lightweight_new_imu_nis_gate")
    check((new_imu_gate.isInt() and new_imu_gate.real() == 0)
          or (new_imu_gate.isString() and new_imu_gate.string().partition(" ")[0].lower() in {"false", "no", "n", "off"}),
          "Hilti lightweight_new_imu_nis_gate must default to false")
    visual_batch = target.getNode("estimator_parameters").getNode("backend_visual_batch")
    check(visual_batch.isString() and visual_batch.string() in {"cpu", "cuda", "hybrid", "schur", "audit"},
          "backend_visual_batch must be cpu, cuda, hybrid, schur, or audit")
    for obsolete in ("num_keyframes", "num_imu_frames"):
        check(target.getNode("estimator_parameters").getNode(obsolete).empty(), f"Obsolete window quota remains: {obsolete}")
    pipeline = (ROOT / "src/pipeline/ThreadedSlam.cpp").read_text(encoding="utf-8")
    check("options.windowSize = static_cast<size_t>(p.estimator.window_size);" in pipeline, "Pipeline must use the single window capacity")

    # Derive boolean fields from the C++ structs so optional ORB switches receive
    # the same type checks as the older required flags. Do not accept maps or
    # silently replace a malformed user-supplied flag with its default.
    parameter_types = (ROOT / "src/common/Parameters.hpp").read_text(encoding="utf-8")
    obsolete_sections = {
        "frontend_parameters": ("use_cnn", "detection_threshold", "absolute_threshold", "matching_threshold", "octaves", "keyframe_overlap", "keyframe_only_new_landmarks"),
        "camera_parameters": ("sync_cameras", "deep_stereo_indices", "fov_scale"),
        "estimator_parameters": ("num_loop_closure_frames", "drift_percentage_heuristic"),
        "output_parameters": ("display_topview", "enable_submapping"),
    }
    for section, fields in obsolete_sections.items():
        for key in fields:
            check(target.getNode(section).getNode(key).empty(), f"Unused configuration remains: {section}/{key}")
            check(re.search(r"\b"+key+r"\b", parameter_types) is None, f"Unused parameter storage remains: {key}")
    for source in ("src/frontend/Frontend.cpp", "src/frontend/Frontend.hpp", "src/visual/Frame.hpp", "src/visual/MultiFrame.hpp"):
        text = (ROOT / source).read_text(encoding="utf-8")
        check(not re.search(r"\b(use_cnn|CGMLI_USE_NN|endCnnThreads|computeClassifications|isClassified|getClassification)\b", text),
              "Removed CNN configuration still has a consumer in " + source)
    for struct, section in (("ImuParameters", "imu_parameters"), ("FrontendParameters", "frontend_parameters"),
                            ("EstimatorParameters", "estimator_parameters"), ("OutputParameters", "output_parameters")):
        body = re.search(r"struct\s+" + struct + r"\s*\{(.*?)\n\};", parameter_types, re.S)
        check(body is not None, f"Cannot locate C++ {struct}")
        for key in re.findall(r"\bbool\s+(\w+)\s*[;=]", body.group(1)):
            check(valid_boolean_node(target.getNode(section).getNode(key)), f"Invalid boolean node {section}/{key}")
    calibration = target.getNode("camera_parameters").getNode("online_calibration")
    for key in ("do_extrinsics", "do_extrinsics_final_ba", "estimate_camera_rig_translation", "estimate_td_camera"):
        check(valid_boolean_node(calibration.getNode(key)), f"Invalid calibration boolean {key}")
    for key in ("sigma_td_camera_s", "max_td_change_s", "max_rig_translation_change_m", "max_rig_rotation_change_rad"):
        check(calibration.getNode(key).real() > 0, f"Nonpositive calibration setting {key}")

    # Parser-only fixtures: exercise the predicate against the original failure
    # as well as quoted/unquoted text, integer flags and invalid composite nodes.
    fixture = cv2.FileStorage("%YAML:1.0\nbroken: true # false: ROI reference\n"
                             'plain: true\nquoted: "false"\ninteger: 0\nsequence: [1]\nreal: 0.5\n',
                             cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
    for key, expected in (("broken", False), ("plain", True), ("quoted", True),
                          ("integer", True), ("sequence", False), ("real", False)):
        check(valid_boolean_node(fixture.getNode(key)) == expected, f"Boolean type guard regression: {key}")
    fixture.release()

    parameters = target.getNode("imu_parameters")
    for key, original in (("sigma_g_c", "gyroscope_noise_density"), ("sigma_a_c", "accelerometer_noise_density"),
                          ("sigma_gw_c", "gyroscope_random_walk"), ("sigma_aw_c", "accelerometer_random_walk")):
        check(parameters.getNode(key).real() == source_imu.getNode("imu0").getNode(original).real(), "Continuous noise mapping mismatch: " + key)
    body = sequence(parameters.getNode("T_BS")).reshape(4, 4)
    valid_transform(body, "T_BS")
    check(np.array_equal(body, np.eye(4)), "This config explicitly defines B=S")
    check(target.getNode("camera_parameters").getNode("image_delay").real() == 0, "Official image shift is zero")
    check(target.getNode("camera_parameters").getNode("timestamp_tolerance").real() == 0.025, "Initial mean-time grouping span is 25 ms")
    shifts = re.findall(r"timeshift cam(\d+) to imu0:[^\n]*\n([+0-9eE.\-]+)", (REFERENCE / "calibration.txt").read_text())
    check(len(shifts) == 8 and all(float(shift) == 0 for _, shift in shifts), "Official camera timing report changed")
    check(np.array_equal(sequence(parameters.getNode("s_a")), [1, 1, 1]), "Do not add an unmeasured acceleration scale")
    check(target.getNode("lidar").empty(), "LiDAR fusion must remain explicitly inactive")
    transform = sequence(target.getNode("lidar_reference").getNode("T_SL")).reshape(4, 4)
    valid_transform(transform, "reference T_SL")
    check(np.allclose(transform, rigid_inverse(source_matrix(source_lidar.getNode("lidar0").getNode("T_here_imu0"))), atol=1e-12, rtol=0), "LiDAR inverse mismatch")
    check(np.allclose(transform, quaternion_transform(source_quaternions.getNode("T_imu0_lidar0")), atol=1e-10, rtol=0), "LiDAR quaternion cross-check mismatch")

    reader_header = (ROOT / "src/offline/HiltiBinReader.hpp").read_text(encoding="utf-8")
    topics = re.findall(r'"(/oak_cam_[^"\n]+/left/image_raw)"', reader_header)
    check(len(topics) == 4, "Reader must select four left cameras")
    for index, topic in enumerate(topics):
        check(topic == cameras.at(index).getNode("rostopic").string(), "Reader/calibration topic order mismatch")
    app = (ROOT / "src/app/offline_main.cpp").read_text(encoding="utf-8")
    check("runHilti(fs::u8path(argv[1]),dataset,output)" in app, "Direct config/dataset entry missing")
    check("parameters.camera.timestamp_tolerance*1e9" in app, "Hilti application must use the configured bundle span")
    for file in (target, source_cameras, source_imu, source_lidar, source_quaternions):
        file.release()
    print(f"Hilti config static checks passed: {checks}; parser OpenCV {cv2.__version__}. No compilation or SLAM execution.")


if __name__ == "__main__":
    main()
