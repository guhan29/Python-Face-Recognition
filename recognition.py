import cv2
import numpy as np
import face_recognition as fr
import os

img_path = 'images'
images = []
image_names = []
image_name_list = os.listdir(img_path)

# Append images to images list and append their names in image_names
for name in image_name_list:
	cur_img = cv2.imread(f'{img_path}/{name}')
	images.append(cur_img)
	image_names.append(os.path.splitext(name)[0])

# Encodes all the given list of images and return the encoded images list
def encode_images(images):
	encoded_list = []
	for img in images:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		encode = fr.face_encodings(img)[0]
		encoded_list.append(encode)

	return encoded_list

encoded_images_list = encode_images(images)
print("Encoding Completed")

# Initialize the video capture
cap = cv2.VideoCapture(0)

''' Uncomment these lines of you want to set frame height and width '''
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# cap.set(cv2.CAP_PROP_FPS, 25)

# print(cap.get(cv2.CAP_PROP_FPS))
# print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

while True:
	success, img = cap.read()
	resized_img = cv2.resize(img, (0, 0), None, 0.25, 0.25)
	resized_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	face_cur_frame = fr.face_locations(resized_img)
	encoded_cur_frame = fr.face_encodings(resized_img, face_cur_frame)

	for encoded_face, face_loc in zip(encoded_cur_frame, face_cur_frame):
		matches = fr.compare_faces(encoded_images_list, encoded_face)
		face_dis = fr.face_distance(encoded_images_list, encoded_face)
		match_index = np.argmin(face_dis)
		# print(face_dis)
		# print(match_index)
		name = "Unknown"

		if matches[match_index]:
			name = image_names[match_index].title()
			# print(name)
			y1, x2, y2, x1 = face_loc
			# y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
			cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
			cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 0, 255), cv2.FILLED)
			cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
		else:
			y1, x2, y2, x1 = face_loc
			# y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
			cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
			cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 0, 255), cv2.FILLED)
			cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

	cv2.imshow('Python Face Recognition', img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		break