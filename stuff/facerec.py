import numpy as np


def get_sface_embedding(jpeg_bytes: bytes) -> list:
    """
    Takes a JPEG byte array containing one unaligned face,
    and returns the SFace embedding as a Python list.
    """
    import cv2 as cv
    # Load JPEG image from bytes
    image_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv.imdecode(image_array, cv.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image data")

    # Initialize face detector
    detector = cv.FaceDetectorYN.create(
        model='/mldata/facerec/onnx/face_detection_yunet_2023mar.onnx',
        config='',
        input_size=(img.shape[1], img.shape[0]),
        score_threshold=0.9,
        nms_threshold=0.3,
        top_k=5000
    )

    # Resize input to match model input size
    detector.setInputSize((img.shape[1], img.shape[0]))

    # Detect faces
    faces = detector.detect(img)
    if faces[1] is None or len(faces[1]) == 0:
        raise ValueError("No face detected in the image")

    # Initialize face recognizer
    recognizer = cv.FaceRecognizerSF.create(
        model='/mldata/facerec/onnx/face_recognition_sface_2021dec.onnx',
        config=''
    )

    # Align and crop face
    face_aligned = recognizer.alignCrop(img, faces[1][0])

    # Get face feature embedding
    embedding = recognizer.feature(face_aligned)

    # Convert to list and return
    return embedding.flatten().tolist()
