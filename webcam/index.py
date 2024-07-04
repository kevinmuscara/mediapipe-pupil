import matplotlib.pyplot as plt
import mediapipe as mp
import cv2

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

options = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task'))

video = cv2.VideoCapture(1)

left_points_x = []
right_points_x = []
left_points_y = []
right_points_y = []

def graph(points, context):
  x_vals = [entry['x'] for entry in points]
  y_vals = [entry['y'] for entry in points]

  plt.plot(x_vals, y_vals, marker='o')
  plt.xlabel('Frame')
  plt.ylabel('Movement')
  plt.title(context + ' Pupil Movement')
  plt.grid(True)
  plt.show()

def analyze_eyes(c, landmark_result, frame):
    face_landmarks_list = landmark_result.face_landmarks
    if face_landmarks_list:
        for face_landmarks in face_landmarks_list:
            left_eye_left_bound = face_landmarks[362]
            x = int(left_eye_left_bound.x * frame.shape[1])
            y = int(left_eye_left_bound.y * frame.shape[0])
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

            left_eye_right_bound = face_landmarks[263]
            x = int(left_eye_right_bound.x * frame.shape[1])
            y = int(left_eye_right_bound.y * frame.shape[0])
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

            left_eye_upper_bound = face_landmarks[386]
            x = int(left_eye_upper_bound.x * frame.shape[1])
            y = int(left_eye_upper_bound.y * frame.shape[0])
            cv2.circle(frame, (x, y), 3, (255,165,0), -1)

            left_eye_lower_bound = face_landmarks[374]
            x = int(left_eye_lower_bound.x * frame.shape[1])
            y = int(left_eye_lower_bound.y * frame.shape[0])
            cv2.circle(frame, (x, y), 3, (255,215,0), -1)

            left_eye_pupil = face_landmarks[473]
            x = int(left_eye_pupil.x * frame.shape[1])
            y = int(left_eye_pupil.y * frame.shape[0])
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            # middle x
            bound_distance = left_eye_right_bound.x - left_eye_left_bound.x
            middle = left_eye_left_bound.x + (bound_distance / 2)
            mean_y = (left_eye_right_bound.y + left_eye_left_bound.y) / 2
            x = int(middle * frame.shape[1])
            y = int(mean_y * frame.shape[0])
            diff = (left_eye_pupil.x * frame.shape[1]) - x
            left_points_x.append({ "x": c, "y": diff })
            cv2.circle(frame, (x, y), 3, (128,0,128), -1)

            # middle y
            bound_distance = left_eye_upper_bound.y - left_eye_lower_bound.y
            middle = left_eye_lower_bound.y + (bound_distance / 2)
            mean_x = (left_eye_upper_bound.x + left_eye_lower_bound.x) / 2
            x = int(mean_x * frame.shape[1])
            y = int(middle * frame.shape[0])
            diff = (left_eye_pupil.y * frame.shape[0]) - y
            left_points_y.append({ "x": c, "y": diff })
            cv2.circle(frame, (x, y), 3, (255,0,255), -1)

            right_eye_left_bound = face_landmarks[33]
            x = int(right_eye_left_bound.x * frame.shape[1])
            y = int(right_eye_left_bound.y * frame.shape[0])
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
            
            right_eye_right_bound = face_landmarks[133]
            x = int(right_eye_right_bound.x * frame.shape[1])
            y = int(right_eye_right_bound.y * frame.shape[0])
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

            right_eye_upper_bound = face_landmarks[159]
            x = int(right_eye_upper_bound.x * frame.shape[1])
            y = int(right_eye_upper_bound.y * frame.shape[0])
            cv2.circle(frame, (x, y), 3, (255,165,0), -1)

            right_eye_lower_bound = face_landmarks[145]
            x = int(right_eye_lower_bound.x * frame.shape[1])
            y = int(right_eye_lower_bound.y * frame.shape[0])
            cv2.circle(frame, (x, y), 3, (255,215,0), -1)

            right_eye_pupil = face_landmarks[468]
            x = int(right_eye_pupil.x * frame.shape[1])
            y = int(right_eye_pupil.y * frame.shape[0])
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            # middle x
            bound_distance = right_eye_right_bound.x - right_eye_left_bound.x
            middle = right_eye_left_bound.x + (bound_distance / 2)
            mean_y = (right_eye_right_bound.y + right_eye_left_bound.y) / 2
            x = int(middle * frame.shape[1])
            y = int(mean_y * frame.shape[0])
            diff = (right_eye_pupil.x * frame.shape[1]) - x
            right_points_x.append({ "x": c, "y": diff })
            cv2.circle(frame, (x, y), 3, (128,0,128), -1)

            # middle y
            bound_distance = right_eye_upper_bound.y - right_eye_lower_bound.y
            middle = right_eye_lower_bound.y + (bound_distance / 2)
            mean_x = (right_eye_upper_bound.x + right_eye_lower_bound.x) / 2
            x = int(mean_x * frame.shape[1])
            y = int(middle * frame.shape[0])
            diff = (right_eye_pupil.y * frame.shape[0]) - y
            right_points_y.append({ "x": c, "y": diff })
            cv2.circle(frame, (x, y), 3, (255,0,255), -1)

def modify(c, frame, marker):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    face_landmarker_result = marker.detect(mp_image)
    analyze_eyes(c, face_landmarker_result, frame)

with FaceLandmarker.create_from_options(options) as landmarker:
    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            print('Error reading frame from the webcam')
            break

        frame = cv2.resize(frame, (640, 480))
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        modify(frame_count, frame_rgb, landmarker)
        frame_count += 1

        cv2.imshow('Webcam Landmarks', cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            graph(right_points_x, 'Right Pupil X')
            graph(right_points_y, 'Right Pupil Y')
            graph(left_points_x, 'Left Pupil X')
            graph(left_points_y, 'Left Pupil Y')
            break

video.release()
cv2.destroyAllWindows()
