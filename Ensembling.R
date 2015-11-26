setwd("Documents/Project Team/Rossman")

#Importing prediction csv from different models
pred1 <- read_csv("xgb7.csv")
pred2 <- read_csv("h2o_random_forest2.csv")
pred2 <- read_csv("h2o_random_forest1.csv")
pred2 <- read_csv("h2o_random_forest3.csv")

#After few tries, decided with the following weights
pred <- 0.8*pred1$Sales + 0.2*(0.3*pred2$Sales + 0.3*pred3$Sales + 0.4*pred4$Sales)
submission <- data.frame(Id=pred1$Id, Sales=pred)
write.csv(submission, "combined1.csv",row.names=F)
