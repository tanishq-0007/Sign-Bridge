def extract_features(hand_landmarks):
    features = []

    for landmark in hand_landmarks:
        features.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    return features