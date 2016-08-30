
#Principal Component Analysis for Kaggle Digit Recognizer Competition
#utilizing Dr. Crawford's slides from http://faculty.tarleton.edu/crawford/documents/math505/PrincipalComponents.pdf
#and John Koo's pcademo.R

#NOTE: If we change the number of components, we need to manually 
#change the names of the imported training and testing data for the
#read.csv() in the randomForest part of this script.

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Create the pca training data
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#memory.limit(20000)
#Juliann's Import
 train <- read.csv("C:\\Users\\jbooth\\Dropbox\\Optimistic Tundra Zombies\\kaggle_digit\\train.csv")
#Cheyenne's Import
#train <- read.csv("C:\\Users\\cmccoy\\Dropbox\\Optimistic Tundra Zombies\\kaggle_digit\\train.csv")
# Lain's Import
#train <- read.csv("C:\\Users\\ltomlinson\\Dropbox\\Optimistic Tundra Zombies\\kaggle_digit\\train.csv")

#---------------------------------------------------------------------
#step 1: get vectors from the training data

#Save the labels of train to place in pca at the end
labels_column <- train[,1]

#Don't use column 1 because it is the labels.
train <- train[,2:785]


#--------------------------------------------------------------------

#Step2: find covariance matrix and eigenvalues/vectors

s <- cov(train)
eigenvalues <- eigen(s)


#----------------------------------------------------------------
#step3: Plot the eigenvalues and their partial sums. 
#Note: eigen(s)$values automatically sorts the values from maximum to minimum.


e.val <- eigenvalues$values
e.vect <- eigenvalues$vectors
exp.var <- e.val/sum(e.val)
exp.var.sum <- cumsum(exp.var)


plot(exp.var, ylim=c(0,1))
points(exp.var.sum, col='red')

#---------------------------------------------------------------------
#step 4: derive the new data set

#take the transpose of the neweigen and multiply it on the
#left of the original data set, transposed. 
#(http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf)

#Calculate the threshold
n.components <- 1 + sum(exp.var.sum < 0.9) 
#it gives 87

pca <- t(e.vect[,1:n.components]) %*% t(train) #[n.components*784] x [784*42000]
pca <- t(pca) #gives us our newly improved training data of dimension 42000 x n.components


#place the labels into the pca file from the training data
pca <- cbind(labels_column, pca)

#saving our new dimension-reduced training data as a csv.
#The paste function just prevents us from having to modify the file
 #everytime we use a different number of components. Note that the
 #paste does not work in read.csv()...or at least I haven't been
 #able to get it to work. 
#Use row.names = FALSE to prevent a row ID column being placed in the
 #data set.
write.csv(pca, paste("pca_train_",n.components,".csv",sep=""), row.names=FALSE)
 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Create the pca testing data
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#Juliann's
test <- read.csv("C:\\Users\\jbooth\\Dropbox\\Optimistic Tundra Zombies\\kaggle_digit\\test.csv")
#Cheyenne's
#test <- read.csv("C:\\Users\\cmccoy\\Dropbox\\Optimistic Tundra Zombies\\kaggle_digit\\test.csv")
#Lain's
#test <- read.csv("C:\\Users\\ltomlinson\\Dropbox\\Optimistic Tundra Zombies\\kaggle_digit\\test.csv")

#________________________________________________________________________________

#step 1: derive the new data set

#Use the basis we computed above, now just multiply it by test

pca_test <- t(e.vect[,1:n.components]) %*% t(test) 
pca_test <- t(pca_test) 


#saving our new dimension-reduced testing data as a csv
write.csv(pca_test, paste("pca_test_",n.components,".csv",sep=""), row.names=FALSE)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Test accuracy of pca through randomForest code
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Compare pca to test and find accuracy by uploading
#to kaggle...or by comparing it to our submission excel files that we already know the accuracy of.
#by using a loop to check if the labels agree (boolean) and count
#the number of TRUE's we get verses the number of FALSE's. This will help us know how 
#many components we need.


library(randomForest)
library(readr)

#Cheyenne's Import
 #pca<- read.csv("C:\\Users\\cmccoy\\Dropbox\\Optimistic Tundra Zombies\\kaggle_digit\\Principal Component Analysis\\pca_train_87.csv")
 #test <- read.csv("C:\\Users\\cmccoy\\Dropbox\\Optimistic Tundra Zombies\\kaggle_digit\\Principal Component Analysis\\pca_test_87.csv")
#Juliann's Import
pca <- read.csv("C:\\Users\\jbooth\\Dropbox\\Optimistic Tundra Zombies\\kaggle_digit\\Principal Component Analysis\\pca_train_87.csv")
test <- read.csv("C:\\Users\\jbooth\\Dropbox\\Optimistic Tundra Zombies\\kaggle_digit\\Principal Component Analysis\\pca_test_87.csv")
#Lain's Import
#pca <- read.csv("C:\\Users\\ltomlinson\\Dropbox\\Optimistic Tundra Zombies\\kaggle_digit\\Principal Component Analysis\\pca_train_87.csv")
#test <- read.csv("C:\\Users\\ltomlinson\\Dropbox\\Optimistic Tundra Zombies\\kaggle_digit\\Principal Component Analysis\\pca_test_87.csv")


set.seed(0) 
numTrain <- nrow(pca) 
numTest <- nrow(test)
numTrees <- 100
rows <- sample(1:nrow(pca), numTrain)
labels <- as.factor(pca[,1])


#Testing Memory
#Size <- object.size(pca)/1048600
#Size

rf <- randomForest(x= pca[rows,-1)], y = labels[rows], xtest = test[,], 
                   ntree = numTrees, replace = TRUE)

 
predictions <- data.frame( ImageId = 1:nrow(test),
                           Label = levels(labels)[rf$test$predicted])
#head(predictions)


write_csv(predictions, paste("pca_rf_dim",n.components,".csv",sep="")) 


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Step 6: Analyze the randomForest predictions
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#Check how many times each digit was predicted to see if it is logical
#recall that "Label" is from predictions above (Label = levels(labels)[rf$test$predicted])

list_of_values = function() 
{
  mylist <- list(zero=1,one=1,two=1,three=1,four=1,five=1,six=1,seven=1,eight=1,nine=1)
  j=0
  while (j<10)
  {
    count = 0
    for (i in 1:length(Label))
    {
      if (Label[i] == j)
      {
        count = count+1
      }  
    }
    percent_of_data <- (count/nrow(test))*100
    j = j+1
    #add 1 to j before placing in list because the lists start at 1
    mylist[j] <- percent_of_data
  }
mylist
}
list_of_values()

 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Attempt a k-nearest neighbor with our PCA data sets as seen on kaggle
#source: https://www.kaggle.com/winsxx/digit-recognizer/pca-knn-with-r
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
library(readr)
library(class)
 
#Juliann's Import
pca <- read.csv("C:\\Users\\jbooth\\Dropbox\\Optimistic Tundra Zombies\\kaggle_digit\\Principal Component Analysis\\pca_train_87.csv")
test <- read.csv("C:\\Users\\jbooth\\Dropbox\\Optimistic Tundra Zombies\\kaggle_digit\\Principal Component Analysis\\pca_test_87.csv")

 
set.seed(0)
#numTrain <- nrow(pca)
#rows <- sample(1:nrow(pca), numTrain)
#train.col.used <- 1:43
prediction <- knn(pca[,-1], test[,], pca[,1], k=3)
prediction.table <- data.frame(ImageId=1:nrow(test), Label=prediction)
Label <- prediction
write_csv(prediction.table, "pca_knn.csv")

#uploaded to kaggle and received an accuracy of 0.97186, best score to date.

 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#attempting an SVM
#https://www.kaggle.com/lucianolattes/digit-recognizer/classifying-with-svm-using-polydot-ker
#Using a polynomial kernel (http://www.cs.columbia.edu/~kathy/cs4701/documents/jason_svm_tutorial.pdf)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 library(readr)
 library(kernlab)
 
 pca <- read.csv("C:\\Users\\jbooth\\Dropbox\\Optimistic Tundra Zombies\\kaggle_digit\\Principal Component Analysis\\pca_train_87.csv")
 test <- read.csv("C:\\Users\\jbooth\\Dropbox\\Optimistic Tundra Zombies\\kaggle_digit\\Principal Component Analysis\\pca_test_87.csv")
 
 pca <- read.csv("C:\\Users\\ltomlinson\\Dropbox\\Optimistic Tundra Zombies\\kaggle_digit\\Principal Component Analysis\\pca_train_87.csv")
 test <- read.csv("C:\\Users\\ltomlinson\\Dropbox\\Optimistic Tundra Zombies\\kaggle_digit\\Principal Component Analysis\\pca_test_87.csv")
 
 colnames(test) <- colnames(pca[-1])
 #This fixes the error that happened for labels below.
 
 pca$labels_column <- as.factor(pca$labels_column)
 #pca$labels_column
 numTrain <- nrow(pca)
 set.seed(15)
 rows <- sample(1:nrow(pca), numTrain)
 pca2 <- pca[rows,]
 
 ptm <- proc.time()
 
 filter <- ksvm(labels_column ~ ., data = pca2, kernel = "polydot",
                kpar = list(degree = 3), cross = 3)
 labels <- predict(filter, test)
 
 svm_time <- proc.time()- ptm
 svm_time
 
 # took 1682.99 seconds, or 28 minutes for using all of the pca training data
 
 predictions <- data.frame(ImageId = 1:nrow(test), Label = levels(pca$labels_column)[labels])
 Label <- levels(pca$labels_column)[labels]
 write_csv(predictions, "polydot_d3_c3_42000.csv")

 
