import sys
import numpy as np
import pandas as pd
from sklearn import datasets 
from sklearn.preprocessing import LabelEncoder

# class used to manipulate the given dataset and take input from the user
class main():

	
	def main():

		# Read in csv file. Has Headers.
		data = pd.read_csv(sys.argv[2])

		# Trim the dataset to needed traits and target value. Will be needed for both training and testing data files.
			#To Be written

		# Convert categorical data into numerical, with respect to columns. Uses headers.
		converter = preprocessing.LabelEncoder()
		converter.fit(obj_df["AnimalType"].astype(str))
		list(le.classes_)

		# Use outside ensemble methods with 
			# To be written
	
		# Report time, accuracy, and anything else needed for the report 
			# To be Written
		
	main()





