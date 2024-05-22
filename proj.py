import matplotlib.pyplot as plt
import numpy as np
import cv2

cap = cv2.VideoCapture("object_tracking.mp4")
cent_arr =np.array([0,0])
my_frame =np.array([0,0,0])
token =0

while(True):
	ret, frame = cap.read()

	if not ret:
		break

	token = token+1

	#Grabbing 400th frame
	if token == 500:
		my_frame = frame
		# print("500th frame")
		# print(np.shape(my_frame))


	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# token = token+1

	my_arr = np.array([[0, 0]])

	#Finding rows and col of black pixels
	row, col = np.where(gray_frame < 10)
	my_arr = np.column_stack((row, col))

	#Calculating centroid
	if my_arr.size > 0:
		min_x = np.min(my_arr[1:,1])
		min_y =	np.min(my_arr[1:,0])
		max_x = np.max(my_arr[1:,1])
		max_y =	np.max(my_arr[1:,0])

		#centroid(x,y)
		centroid = 	np.array([min_x +((max_x - min_x)//2),min_y+((max_y - min_y)//2)])
		# print(centroid)
		cent_arr = np.vstack((cent_arr,centroid))
		# cent_arr = np.vstack((cent_arr,centroid))

		#Plotting Circle
		orign = (centroid[0],centroid[1])
		frame = cv2.circle(frame,orign, 5, (0, 255, 0,),-1)

	#Displaying Image
	cv2.imshow("Centroid window", frame)

	ch = cv2.waitKey(1)
	if ch & 0xFF == ord('q'):
		break
	
# print(cent_arr)
cap.release()
cv2.destroyAllWindows()
# print(cent_arr)

# Graph fitting

# Matrix A
A = np.column_stack([cent_arr[:, 0] ** 2, cent_arr[:, 0], np.ones(len(cent_arr))])

# Vector B
B = cent_arr[:, 1]

# Calculating A transpose 
A_t = np.transpose(A)

# Calculating (A tanspose)*B
A_t_B = np.matmul(A_t, B)

# Calculating (A transpose)*A
A_t_A = np.matmul(A_t,A)

# Calculating inverse((A transpose)*A)
inv_A_t_A = np.linalg.inv(A_t_A)

# Solving the equation inverse((A transpose)*A)*((A tanspose)*B)
B = np.matmul(inv_A_t_A,A_t_B)
# print(B)

#Coefficients of Parabola equation
a, b, c = tuple(B)
# print(a, b, c)
xi = 1000
yi = a*xi**2 + b*xi + c
# print(yi)

# Plotting parabola 

x = np.linspace(cent_arr[0, 0], cent_arr[-1, 0], 1000)
y = a * x**2 + b * x + c

fig1 = plt.figure()
plt.plot(x,y)

# plt.plot(cent_arr[:,0],cent_arr[:,1])
plt.scatter(cent_arr[:,0],cent_arr[:,1],c ="green")

#Displaying 500th frame
my_frame = cv2.cvtColor(my_frame,cv2.COLOR_BGR2RGB) 
plt.imshow(my_frame)
plt.show()

