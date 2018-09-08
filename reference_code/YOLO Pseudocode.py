#this is an Image of size 140x140. We will assume it to be black and white (ie only one channel, it would have been 140x140x3 for rgb)
image = readImage()

#We will break the Image into 7 coloumns and 7 rows and process each of the 49 different parts independently
NoOfCells = 7

#we will try and predict if an image is a dog, cat, cow or wolf. Therfore the number of classes is 4
NoOfClasses = 4
threshold = 0.7

#step will be the size of step to take when moving across the image. Since the image has 7 cells step will be 140/7 = 20
step = height(image)/NoOfCells

#stores the class for each of the 49 cells, each cell will have 4 values which correspond to the probability of a cell being 1 of the 4 classes
#prediction_class_array[i,j] is a vector of size 4 which would look like [0.5 #cat, 0.3 #dog, 0.1 #wolf, 0.2 #cow]
prediction_class_array = new_array(size(NoOfCells,NoOfCells,NoOfClasses))

#stores 2 bounding box suggestions for each of the 49 cells, each cell will have 2 bounding boxes, with each bounding box having x, y, w ,h and c predictions. (x,y) are the coordinates of the center of the box, (w,h) are it's height and width and c is it's confidence
predictions_bounding_box_array = new_array(size(NoOfCells,NoOfCells,NoOfCells,NoOfCells))

#it's a blank array in which we will add the final list of predictions
final_predictions = []

#minimum confidence level we require to make a prediction
threshold = 0.7

for (i<0; i<NoOfCells; i=i+1):
	for (j<0; j<NoOfCells;j=j+1):
		#we will get each "cell" of size 20x20, 140(image height)/7(no of rows)=20 (step) (size of each cell)"
		#each cell will be of size (step, step)
		cell = image(i:i+step,j:j+step) 

		#we will first make a prediction on each cell as to what is the probability of it being one of cat, dog, cow, wolf
		#prediction_class_array[i,j] is a vector of size 4 which would look like [0.5 #cat, 0.3 #dog, 0.1 #wolf, 0.2 #cow]
		#sum(prediction_class_array[i,j]) = 1
		#this gives us our preidction as to what each of the different 49 cells are
		#class predictor is a neural network that has 9 convolutional layers that make a final prediction
		prediction_class_array[i,j] = class_predictor(cell)

		#predictions_bounding_box_array is an array of 2 bounding boxes made for each cell
		#size(predictions_bounding_box_array[i,j]) is [2,5]
		#predictions_bounding_box_array[i,j,1] is bounding box1, predictions_bounding_box_array[i,j,2] is bounding box 2
		#predictions_bounding_box_array[i,j,1] has 5 values for the bounding box [x,y,w,h,c]
		#the values are x, y (coordinates of the center of the bounding box) which are whithin the bounding box (values ranging between 0-20 in your case)
		#the values are h, w (height and width of the bounding box) they extend outside the cell and are in the range of [0-140]
		#the value is c a confidence of overlap with an acutal bounding box that should be predicted
		predictions_bounding_box_array[i,j] = bounding_box_predictor(cell)

		#predictions_bounding_box_array[i,j,0, 4] is the confidence value for the first bounding box prediction
		best_bounding_box =  [0 if predictions_bounding_box_array[i,j,0, 4] > predictions_bounding_box_array[i,j,1, 4] else 1]

		# we will get the class which has the highest probability, for [0.5 #cat, 0.3 #dog, 0.1 #wolf, 0.2 #cow], 0.5 is the highest probability corresponding to cat which is at position 0. So index_of_max_value will return 0
		predicted_class = index_of_max_value(prediction_class_array[i,j])

		#we will check if the prediction is above a certain threshold (could be something like 0.7)
		if predictions_bounding_box_array[i,j,best_bounding_box, 4] * max_value(prediction_class_array[i,j]) > threshold:

			#the prediction is an array which has the x,y coordinate of the box, the height and the width
			prediction = [predictions_bounding_box_array[i,j,best_bounding_box, 0:4], predicted_class]

			final_predictions.append(prediction)


print final_predictions