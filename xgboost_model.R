setwd("Documents/Project Team/Rossman")
getwd()


library(readr)  
library(xgboost)  #don't forget to install.packages("xgboost")

set.seed(345)  #most models train themselves in a "random" alogrithm, using set.seed allows you to maintain consistency from one model to another

cat("reading the train and test data\n")
train<- read.csv("train.csv",stringsAsFactors=FALSE)
#train=read.csv("train2.csv",stringsAsFactors=FALSE)
test<- read.csv("test.csv",stringsAsFactors=FALSE)
store<- read.csv("store.csv",stringsAsFactors=FALSE)  

# removing the date column (since elements are extracted) and also StateHoliday which has a lot of NAs (may add it back in later)
train <- merge(train,store)
test <- merge(test,store)

all_stores <- unique(train$Store)
stores_reporting <- train$Store[train$Date == as.Date("2014-7-1")]
missing_stores <- all_stores[!(all_stores %in% stores_reporting)]
missing_stores

# There are some NAs in the integer columns so conversion to zero
train[is.na(train)]   <- -1
test[is.na(test)]   <- -1

#displaying data information
cat("train data column names and details\n")
names(train)
str(train)
summary(train)
cat("test data column names and details\n")
names(test)
str(test)
summary(test)

# looking at only stores that were open in the train set
# may change this later
train <- train[ which(train$Open=='1'),]
train <- train[ which(train$Sales!='0'),]
# seperating out the elements of the date column for the train set



train$Date= as.Date(train$Date)
test$Date=as.Date(test$Date)
train$Month <- as.integer(format(train$Date, "%m"))
train$Year <- as.integer(format(train$Date, "%y"))
train$Day <- as.integer(format(train$Date, "%d"))

# removing the date column (since elements are extracted) and also StateHoliday which has a lot of NAs (may add it back in later)
train <- train[,-c(3,8,15,18)]

# seperating out the elements of the date column for the test set
test$Month <- as.integer(format(test$Date, "%m"))
test$Year <- as.integer(format(test$Date, "%y"))
test$Day <- as.integer(format(test$Date, "%d"))

# removing the date column (since elements are extracted) and also StateHoliday which has a lot of NAs (may add it back in later)
test <- test[,-c(4,7,14,17)]


#Take store B as a seperate column (Store type)
#train$StoreType <- as.factor(train$StoreType)
#train$TypeB <- train$StoreType == "b"

#test$StoreType <- as.factor(test$StoreType)
#test$TypeB <- test$StoreType == "b"

amean = aggregate(Sales~Store, data = train, mean)
train$MeanSales <- amean$Sales[train$Store]
test$MeanSales <- amean$Sales[test$Store]

#I think "open" is an really important variable, so I didn't delete it
feature.names <- names(train)[c(1,2,5:18)]
feature.names
#Convert columns to integers
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}


cat("train data column names after slight feature engineering\n")
names(train)
cat("test data column names after slight feature engineering\n")
names(test)
tra<-train[,feature.names]

#they created their own RMPSE equation
RMPSE<- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  elab<-exp(as.numeric(labels))-1
  epreds<-exp(as.numeric(preds))-1
  err <- sqrt(mean((epreds/elab-1)^2))
  return(list(metric = "RMPSE", value = err))
}
nrow(train)

#taking sample, for large samples taking samples helps model train faster
h<-sample(nrow(train),15000)

#takes in predictors (x variables) and y and creates a xgb.DMatrix
#in this case they took log of y 
dval<-xgb.DMatrix(data=data.matrix(tra[h,]),label=log(train$Sales+1)[h])
dtrain<-xgb.DMatrix(data=data.matrix(tra[-h,]),label=log(train$Sales+1)[-h])
watchlist<-list(val=dval,train=dtrain)  #used for watchlist parameter, not neccessary

#creating a list of parameters
param <- list(  objective           = "reg:linear", #other options include binary:logistic, but in this competition reg:linear is fine
                booster = "gbtree",   #which booster to use, can be gbtree or gblinear. For the competition, gbtree works better
                eta                 = 0.02, #control the learning rate: scale the contribution of each tree by a factor of 0 < eta < 1 when it is added to the current approximation. Used to prevent overfitting by making the boosting process more conservative. 
                #Lower value for eta implies larger value for nrounds: low eta value means model more robust to overfitting but slower to compute.
                max_depth           = 10, #step size of each boosting step, higher value tends to overfit so keep it around 5 to 8
                subsample           = 0.9, #maximum depth of the tree. . Setting it to 0.5 means that xgboost randomly collected half of the data instances to grow trees and this will prevent overfitting.
                colsample_bytree    = 0.7 #subsample ratio of columns when constructing each tree. Default: 1
)

#An advanced interface for training xgboost model
#helps you find the optimal parameter values for the model and returns that model
clf <- xgb.train(   params              = param, #takes in list of parameters
                    data                = dtrain, #takes in xgb.DMatrix data
                    nrounds             = 3000, #the max number of iterations similar to n.tree in randomForest
                    #higher the better but watch out for overfitting 100 ~ 500 is fine
                    verbose             = 1,  #prints progress, verbose = 0 if you don't want it to print
                    early.stop.round    = 50,  #If set to an integer k, training with a validation set will stop if the performance keeps getting worse consecutively for k rounds.
                    #increases efficiency of the training
                    watchlist           = watchlist,  #what information should be printed when verbose=1 or verbose=2. Watchlist is used to specify validation set monitoring during training. For example user can specify watchlist=list(validation1=mat1, validation2=mat2) to watch the performance of each round's model on mat1 and mat2
                    #not entirely neccessary
                    maximize            = FALSE,  #If feval and early.stop.round are set, then maximize must be set as well. maximize=TRUE means the larger the evaluation score the better.
                    #in this case it is set to false because in RMPSE, the lower the score the better
                    feval=RMPSE   #custimized evaluation function. created earlier, takes in two parameters y and y hat
)
pred1 <- exp(predict(clf, data.matrix(test[,feature.names]))) -1
submission <- data.frame(Id=test$Id, Sales=pred1)
cat("saving the submission file\n")
write_csv(submission, "xgb7.csv")
save(clf, file = "xgboost2.RData")
load("xgboost2.RData")

pred2 = exp(predict(clf, data.matrix(train[-h,feature.names]))) -1
library(cvTools)
index=t(h)
rmspe(train$Sales[-index],pred2)
#h=10000, nround=700   rmspe=600
#h=25000, nround=200   rmspe=760





