install.packages("h2o")
library(h2o)
localH2o = h2o.init()

setwd("Documents/Project Team/Rossman")

library("readr")
library("randomForest")

#Importing train, test, and store data
train <- read_csv("train.csv")
test <- read_csv("test.csv")
store <- read_csv("store.csv")

#Merge the two datasets (store.csv and train.csv)
train <- merge(train, store, by = "Store")
test <- merge(test, store, by = "Store")

#Cleaning up data
train$Open[train$Open == -1] <- 0
#train <- train[train$Open != 0,]
test$Open[test$Open == -1] <- 0
#test$Sales[test$Sales < 0] <- 0
train[is.na(train)] <- -1
test[is.na(test)]<- -1

#Creating new columns with month, day and year
train$Month <- as.integer(format(train$Date, "%m"))
train$Day <- as.integer(format(as.Date(train$Date),"%d"))
train$Year <- as.integer(format(as.Date(train$Date),"%y"))

#Delete the original datetime column and stateholiday, promo2, promo_interval in train (showed improvement when deleted)
train <- train[,-c(3,8,15,18)]

#Do the same thing for test
test$Month <- as.integer(format(test$Date, "%m"))
test$Day <- as.integer(format(as.Date(test$Date),"%d"))
test$Year <- as.integer(format(as.Date(test$Date),"%y"))

test <- test[,-c(4,7,14,17)]

#Convert more values to factors, did not show improvement
# train$Promo <- as.factor(train$Promo)
# train$Open <- as.factor(train$Open)
# train$DayOfWeek <- as.factor(train$DayOfWeek)
# train$SchoolHoliday <- as.factor(train$SchoolHoliday)
#Take store B as a seperate column (Store type)
#train$StoreType <- as.factor(train$StoreType)
#train$TypeB <- train$StoreType == "b"

#test$StoreType <- as.factor(test$StoreType)
#test$TypeB <- test$StoreType == "b"

#Closed for a long datetime
#if closed in test, sales = 0

#Extract feature names
feature.names <- names(train)[c(1,2,6:9,15:18)]

#Convert columns to integers
for (f in feature.names) {
    if (class(train[[f]])=="character") {
        levels <- unique(c(train[[f]], test[[f]]))
        train[[f]] <- as.integer(factor(train[[f]], levels=levels))
        test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
    }
}

#Create log of Sales
train$logSales<-log1p(train$Sales)

#Additional Feature Engineering
amean = aggregate(Sales~Store, data = train, mean)
train$MeanSales <- amean$Sales[train$Store]
test$MeanSales <- amean$Sales[test$Store]

amedian = aggregate(Sales~Store, data = train, median)
train$MedianSales <- amedian$Sales[train$Store]
test$MedianSales <- amedian$Sales[test$Store]

train$NumberOfYearsOpen <- (2000 + train$Year) - train$CompetitionOpenSinceYear
test$NumberOfYearsOpen <- (2000 + test$Year) - test$CompetitionOpenSinceYear

train$NumberOfMonthsOpen <- (12 - train$CompetitionOpenSinceMonth) + (train$NumberOfYearsOpen - 1)*12 + (train$Month) 
test$NumberOfMonthsOpen <- (12 - test$CompetitionOpenSinceMonth) + (test$NumberOfYearsOpen - 1)*12 + (test$Month)


## Use H2O's random forest

## Start cluster with all available threads
h2o.init(nthreads=-1,max_mem_size='6G')
## Load data into cluster from R
trainHex<-as.h2o(train)
features<-colnames(train)[!(colnames(train) %in% c("Id","Date","logSales","Sales","Customers"))]
rfHex1 <- h2o.randomForest(x=features,
                          y="logSales", 
                          ntrees = 100,
                          mtry = 6,
                          max_depth = 30,
                          nbins_cats = 1115, 
                          training_frame=trainHex)
                          
rfHex2 <- h2o.randomForest(x=features,
                          y="logSales", 
                          ntrees = 100,
                          mtry = 7,
                          max_depth = 30,
                          nbins_cats = 1115, 
                          training_frame=trainHex)
                          
#Overfit data slightly
rfHex3 <- h2o.randomForest(x=features,
                          y="logSales", 
                          ntrees = 200,
                          mtry = 7,
                          max_depth = 30,
                          nbins_cats = 1115, 
                          training_frame=trainHex)

## Load test data into cluster from R
testHex<-as.h2o(test[,features])

#RandomForest did not work as well as H2o Random Forest
#clf <- randomForest(train[,feature.names], 
#        log(train$Sales+1), mtry=5, ntree=40, sampsize=30000, importance = TRUE) 
#imp <- importance(clf, type=1)
#pred <- exp(predict(clf, test)) +1
#pred <- exp(predict(clf, test)) -1
#submission <- data.frame(Id=test$Id, Sales=pred)
#write_csv(submission, "random_forest_submission.csv")

## Get predictions out; predicts in H2O, as.data.frame gets them into R
predictions1<-as.data.frame(h2o.predict(rfHex1,testHex))
predictions2<-as.data.frame(h2o.predict(rfHex2,testHex))
predictions3<-as.data.frame(h2o.predict(rfHex3,testHex))

## Return the predictions to the original scale of the Sales data
pred1 <- expm1(predictions1[,1])
pred2 <- expm1(predictions2[,1])
pred3 <- expm1(predictions3[,1])

#Ensemble
pred <- pred1*0.3 + pred2*0.3 + pred3*0.4

#Create csv file 
submission <- data.frame(Id=test$Id, Sales=pred)
write.csv(submission, "h2o_random_forest.csv",row.names=F)