import matplotlib.pyplot as plt
import mediapipe as mp
import cv2

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task'), running_mode=VisionRunningMode.IMAGE)

video = cv2.VideoCapture('moving.mov')

left_points_x = []
right_points_x = []

left_points_y = []
right_points_y = []

def graph(points, context):
  x_vals = [entry['x'] for entry in points]
  y_vals = [entry['y'] for entry in points]
  # y_vals = [min(1, max(-1, entry['y'])) for entry in points]

  plt.plot(x_vals, y_vals, marker='o')
  plt.xlabel('Frame')
  plt.ylabel('Movement')
  plt.title(context + ' Pupil Movement')
  # plt.ylim(-0.05, 0.05)
  plt.grid(True)
  plt.show()

def calc_difference_x(count, left_bound, right_bound, pupil, context):
  bound_difference = right_bound.x - left_bound.x
  middle = left_bound.x + (bound_difference / 2)
  direction = pupil.x - middle

  # data = { "x": count, "y": round(direction, 2) }
  data = { "x": count, "y": direction }

  if context == 'left':
    left_points_x.append(data)
  else:
    right_points_x.append(data)

def calc_difference_y(count, left_bound, right_bound, pupil, context):
  bound_difference = right_bound.y - left_bound.y
  middle = left_bound.y + (bound_difference / 2)
  direction = pupil.y - middle

  # data = { "x": count, "y": round(direction, 2) }
  data = { "x": count, "y": direction }

  if context == 'left':
    left_points_y.append(data)
  else:
    right_points_y.append(data)

def analyze(count, landmark_result):
  face_landmarks_list = landmark_result.face_landmarks

  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    left_eye_left_bound = face_landmarks[362]
    left_eye_right_bound = face_landmarks[263]
    left_eye_upper_bound = face_landmarks[386]
    left_eye_lower_bound = face_landmarks[374]
    left_eye_pupil = face_landmarks[473]

    right_eye_left_bound = face_landmarks[33]
    right_eye_right_bound = face_landmarks[133]
    right_eye_upper_bound = face_landmarks[159]
    right_eye_lower_bound = face_landmarks[145]
    right_eye_pupil = face_landmarks[468]

    calc_difference_x(count, left_eye_left_bound, left_eye_right_bound, left_eye_pupil, 'left')
    calc_difference_x(count, right_eye_left_bound, right_eye_right_bound, right_eye_pupil, 'right')

    calc_difference_y(count, left_eye_upper_bound, left_eye_lower_bound, left_eye_pupil, 'left')
    calc_difference_y(count, right_eye_upper_bound, right_eye_lower_bound, right_eye_pupil, 'right')

def modify(frame, count, marker):
  mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
  face_landmarker_result = marker.detect(mp_image)

  analyze(count, face_landmarker_result)

with FaceLandmarker.create_from_options(options) as landmarker:
  frame_count = 0
  while(video.isOpened()):
    ret, frame = video.read()
    if not ret:
      graph(left_points_x, 'Left')
      graph(right_points_x, 'Right')

      graph(left_points_y, 'Left')
      graph(right_points_y, 'Right')
      break

    modify_frame = modify(frame, frame_count, landmarker)
    frame_count += 1

video.release()
cv2.destroyAllWindows()