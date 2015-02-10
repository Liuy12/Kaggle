library('caret')
training <- read.csv('train.csv',header=T,row.names=1)
testing <- read.csv('test.csv',header=T,row.names=1)
dim(training)
dim(testing)
colnames(training)
with(training,table(Cover_Type))
NA_prop <- sapply(training,function(i) {sum(is.na(i))/length(i)})
table(NA_prop)
Missing_prop <- sapply(training,function(i) {sum(i=='')/length(i)})
table(Missing_prop)


training$Cover_Type <- as.factor(paste('C',training$Cover_Type,sep=''))


### Try random forest using all the features
t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
modelfit_rf_class<-train(Cover_Type~.,method='rf',data=training,
                   trControl = trainControl(method = "repeatedcv", 
                                            savePred=T, classProb=T,
                                            repeats=3), 
                   importance=T,ntree=1000)
t2 <- Sys.time()
t2-t1
## training set accuracy
sum(diag(modelfit_rf$finalModel$confusion))/nrow(training)
testing_class <- predict(modelfit_rf_class, testing)

####
# Try to scale the parameters
# But exclude dummy variables
train_subset2 <- training[,1:10]
nzv <- nearZeroVar(train_subset2)
mean_subset2 <- apply(train_subset2,2,mean)
sd_subset2 <- apply(train_subset2,2,sd)
train_subset2_scaled <- sweep(train_subset2,2,mean_subset2,'-')
train_subset2_scaled <- sweep(train_subset2_scaled,2,sd_subset2,'/')
training_scaled <- training
training_scaled[,1:10] <- train_subset2_scaled

# corMat <- cor(train_subset2_scaled)
# Highcorr <- findCorrelation(corMat,cutoff = 0.9)
t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
modelfit_rf_class1<-train(Cover_Type~.,method='rf',data=training_scaled,
                          trControl = trainControl(method = "repeatedcv", 
                                                   savePred=T, classProb=T,
                                                   repeats=5), 
                          importance=T,ntree=1000, tuneLength=10)
t2 <- Sys.time()
t2-t1

testing_subset <- testing[,1:10]
testing_subset_scaled <- sweep(testing_subset,2,mean_subset2,'-')
testing_subset_scaled <- sweep(testing_subset_scaled,2,sd_subset2,'/')
testing_scaled <- testing
testing_scaled[,1:10] <- testing_subset_scaled

testing_class1 <- predict(modelfit_rf_class1, testing_scaled)

temp <- substr(testing_class1,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class1.csv',row.names=F,quote=F)



####### Use a full set either dummy variables or factor categorical variables
###### Use reduced set, remove nzv, correlated(findCorrelation), and linearly combined (findLinearCombos)

# Try C5.0 with factor variables 
Wilderness <- factor(c(paste('wilder',1:4,sep='')))
Soiltype <- factor(c(paste('soil',1:40,sep='')))
train_subset <- training[,15:54]
train_subset1 <- training[,11:14]
index_soil <- apply(train_subset,1,function(i) which(i==1))
index_wild <- apply(train_subset1,1,function(i) which(i==1))
training_trans <- training[1:10]
training_trans$Wild <- Wilderness[index_wild]
training_trans$Soil <- Soiltype[index_soil]
training_trans[,1:10] <- train_subset2_scaled
training_trans$Cover_Type <- training$Cover_Type


# Create data partition 
set.seed(1990)
Intrain <- createDataPartition(training_trans$Cover_Type, p = 0.75, list = F)
training_trans_train <- training_trans[Intrain,]
training_trans_test <- training_trans[-Intrain,]

t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
modelfit_C50 <- train(Cover_Type~.,data=training_trans_train,
               method = "C5.0", tuneLength = 10,
               trControl = trainControl(method = "repeatedcv",
                                           repeats = 5),
               control = C5.0Control(earlyStopping = FALSE))
t2 <- Sys.time()
t2-t1



postResample(predict(modelfit_C50, training_trans_test), training_trans_test$Cover_Type)

testing_trans <- testing_subset_scaled
testing_subset1 <- testing[,15:54]
testing_subset2 <- testing[,11:14]
index_soil <- apply(testing_subset1,1,function(i) which(i==1))
index_wild <- apply(testing_subset2,1,function(i) which(i==1))

testing_trans$Wild <- Wilderness[index_wild]
testing_trans$Soil <- Soiltype[index_soil]

testing_class2 <- predict(modelfit_C50, testing_trans)

temp <- substr(testing_class2,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class2.csv',row.names=F,quote=F)

### accurasy 0.71

### Try ensemble of C50 and rf 
# Not working, need to redo rf with only the training set 
temp1 <- predict(modelfit_rf_class1, training_scaled[-Intrain,-55])
temp2 <- predict(modelfit_C50, training_trans_test)
temp3 <- data.frame(pred1=temp1, pred2=temp2, Outcome = training_trans_test$Cover_Type)

t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)

modelfit_rf_C50_ensembel <- train(Outcome~., data=temp3, method = 'rf', 
                                  trControl = trainControl(method = "repeatedcv", 
                                                           repeats=5), 
                                  importance=T,ntree=1000, tuneLength=10)

t2 <- Sys.time()
t2-t1

temp4 <- data.frame(pred1=testing_class1, pred2=testing_class2)
testing_class3 <- predict(modelfit_rf_C50_ensembel, temp4)

temp <- substr(testing_class3,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class3.csv',row.names=F,quote=F)

################ redo rf with only the training set

t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
modelfit_rf_class2<-train(Cover_Type~.,method='rf',data=training_scaled[Intrain,],
                          trControl = trainControl(method = "repeatedcv", 
                                                   savePred=T, classProb=T,
                                                   repeats=3), 
                          importance=T,ntree=1000, tuneLength=5)
t2 <- Sys.time()
t2-t1
save.image('data.rdata')

postResample(predict(modelfit_rf_class2, training_scaled[-Intrain,]), training_scaled[-Intrain,]$Cover_Type)

testing_class4 <- predict(modelfit_rf_class2, testing_scaled)

temp1 <- predict(modelfit_rf_class2, training_scaled[-Intrain,-55])
temp2 <- predict(modelfit_C50, training_trans_test)
temp3 <- data.frame(pred1=temp1, pred2=temp2, Outcome = training_trans_test$Cover_Type)

t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)

modelfit_rf_C50_ensembel <- train(Outcome~., data=temp3, method = 'rf', 
                                  trControl = trainControl(method = "repeatedcv", 
                                                           repeats=3), 
                                  importance=T,ntree=1000, tuneLength=5)

t2 <- Sys.time()
t2-t1

temp4 <- data.frame(pred1=testing_class4, pred2=testing_class2)
testing_class3 <- predict(modelfit_rf_C50_ensembel, temp4)

temp <- substr(testing_class3,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class3.csv',row.names=F,quote=F)

### accuracy 0.735
##############
temp <- substr(testing_class4,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class4.csv',row.names=F,quote=F)

##### Accuracy 0.730

#################
# try C5.0 to ensembel
registerDoMC(cores=4)
set.seed(1990)
modelfit_rf_C50_ensembel_C50 <- train(Outcome~., data=temp3, method = 'C5.0', 
                                  trControl = trainControl(method = "repeatedcv", 
                                                           repeats=3), 
                                  importance=T,tuneLength=10)

temp4 <- data.frame(pred1=testing_class4, pred2=testing_class2)
testing_class5 <- predict(modelfit_rf_C50_ensembel_C50, temp4)

temp <- substr(testing_class5,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class5.csv',row.names=F,quote=F)

#################3 Accuracy 0.729

### try rf with pca 
prepro<-preProcess(training_scaled[Intrain,1:10],method='pca',thresh=0.9)
trainpc<-predict(prepro,training_scaled[Intrain,-55])

t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
modelfit_rf_class3<-train(training_scaled[Intrain,]$Cover_Type~.,method='rf',data=trainpc,
                   trControl = trainControl(method = "repeatedcv", 
                                            repeats=5), 
                   importance=T,ntree=1000, tuneLength=5)
t2 <- Sys.time()
t2-t1

######################## gbm

gbmGrid <- expand.grid(.interaction.depth = seq(1,7, by=2),
                       .n.trees = seq(100,1000,by=200),
                       .shrinkage =  0.01 )


t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
temp <- sample(1:10000,200)
modelfit_gbm <- train(Cover_Type~., data= training_trans_train[temp,],
                 method = "gbm",
                 tuneGrid = gbmGrid,
                 trControl = trainControl(method = "repeatedcv", 
                                          repeats=3), 
                 ## The gbm() function produces copious amounts
                 ## of output, so pass in the verbose option
                 ## to avoid printing a lot to the screen
                 verbose = FALSE)
t2 <- Sys.time()
t2-t1

testing_class6 <- predict(modelfit_gbm, testing_trans)

temp <- substr(testing_class6,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class6.csv',row.names=F,quote=F)

########### Accuracy 0.65791
####################### use dummy variables
gbmGrid <- expand.grid(.interaction.depth = seq(1,7, by=2),
                       .n.trees = seq(100,1000,by=200),
                       .shrinkage =  0.01 )

t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
modelfit_gbm_dummy <- train(Cover_Type~., data= training_scaled[Intrain,],
                      method = "gbm",
                      tuneGrid = gbmGrid,
## The gbm() function produces copious amounts
## of output, so pass in the verbose option
## to avoid printing a lot to the screen
verbose = FALSE)
t2 <- Sys.time()
t2-t1

testing_class7 <- predict(modelfit_gbm_dummy, testing_trans)




postResample(predict(modelfit_gbm, training_trans_test), training_trans_test$Cover_Type)

temp <- substr(testing_class6,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class6.csv',row.names=F,quote=F)

############### not much difference in terms of accuracy 


###############################Try random forrest again but exclude soil type 7,8, 15, 25, 
training_scaled_sel <- training_scaled[,-c(21,22,29,39)]
t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
rf_tuneGrid <- expand.grid(.mtry= seq(15, 27, by=4))
modelfit_rf_class3<-train(Cover_Type~.,method='rf',data=training_scaled_sel,
                          trControl = trainControl(method = "repeatedcv", 
                                                   repeats=5), 
                          ntree=1000, tuneGrid=rf_tuneGrid)
t2 <- Sys.time()
t2-t1
testing_class8 <- predict(modelfit_rf_class3, testing[,-c(21,22,29,39)])

#######################################################
## tune rf again with mtry = 17:21
t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
rf_tuneGrid <- expand.grid(.mtry= 17:21)
modelfit_rf_class4<-train(Cover_Type~.,method='rf',data=training_scaled_sel,
                          trControl = trainControl(method = "repeatedcv", 
                                                   repeats=5), 
                          ntree=1000, tuneGrid=rf_tuneGrid)
t2 <- Sys.time()
t2-t1

testing_class9 <- predict(modelfit_rf_class4, testing_scaled[,-c(21,22,29,39)])

################ mtry 18 is the best

############################### Try GBM tune again 
gbmGrid <- expand.grid(.interaction.depth = seq(7,15, by=2),
                       .n.trees = seq(1000,2000,by=200),
                       .shrinkage =  0.01)

t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
modelfit_gbm_dummy <- train(Cover_Type~., data=training_scaled_sel,
                            method = "gbm",
                            tuneGrid = gbmGrid,
                            ## The gbm() function produces copious amounts
                            ## of output, so pass in the verbose option
                            ## to avoid printing a lot to the screen
                            verbose = FALSE)
t2 <- Sys.time()
t2-t1

testing_class13 <- predict(modelfit_gbm_dummy, data=testing_scaled[,-c(21,22,29,39)])
#######################################################
## tune rf again finally with mtry set to 18
## do more CV
t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
rf_tuneGrid <- expand.grid(.mtry= 18)
modelfit_rf_class5<-train(Cover_Type~.,method='rf',data=training_scaled_sel,
                          trControl = trainControl(method = "repeatedcv", 
                                                   repeats=20), 
                          ntree=1000, tuneGrid=rf_tuneGrid)
t2 <- Sys.time()
t2-t1

testing_class10 <- predict(modelfit_rf_class5, testing_scaled[,-c(21,22,29,39)])
temp <- substr(testing_class10,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class10.csv',row.names=F,quote=F)
################ mtry 18 is the best, Accuracy is 0.75623

################################## Try regularized rf
# 
# t1 <- Sys.time()
# registerDoMC(cores=4)
# set.seed(1990)
# rf_tuneGrid <- expand.grid(.mtry= 18)
# modelfit_rrf<-train(Cover_Type~.,method='RRF',data=training_scaled_sel,
#                           trControl = trainControl(method = "repeatedcv", 
#                                                    repeats=2),
# 							tuneGrid = rf_tuneGrid, 					   
#                           ntree=1000)
# t2 <- Sys.time()
# t2-t1
# 
# testing_class9 <- predict(modelfit_rf_class4, testing_scaled[,-c(21,22,29,39)])

####################### Try svm 

t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
#rf_tuneGrid <- expand.grid(.mtry= 18)
modelfit_svm <- train(Cover_Type~.,method='svmRadial',data=training_scaled_sel,
                    trControl = trainControl(method = "repeatedcv", 
                                             repeats=5), tuneLength = 10)
t2 <- Sys.time()
t2-t1

testing_class11 <- predict(modelfit_svm, testing_scaled[,-c(21,22,29,39)])

temp <- substr(testing_class11,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class11.csv',row.names=F,quote=F)

############################ Accuracy 0.70
########################### Tune svm again 
tuneGrid_svm <- expand.grid(.C = c(200, 128*c(2:10)),
                            .sigma = seq(0.01,0.08,by=0.02)
)
t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
#rf_tuneGrid <- expand.grid(.mtry= 18)
modelfit_svm_class2 <- train(Cover_Type~.,method='svmRadial',data=training_scaled_sel,
                      trControl = trainControl(method = "repeatedcv", 
                                               repeats=5), 
                      tuneGrid= tuneGrid_svm)
t2 <- Sys.time()
t2-t1

testing_class12 <- predict(modelfit_svm_class2, testing_scaled[,-c(21,22,29,39)])

temp <- substr(testing_class12,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class12.csv',row.names=F,quote=F)

pred1 <- predict(modelfit_rf_class1, testing_scaled, type='prob')
pred2 <- predict (modelfit_svm_class2, testing_scaled[,-c(21,22,29,39)], type='prob')
############################## Accuracy 0.72782
temp <- -2:8
temp1 <- -5:-1

tuneGrid_svm <- expand.grid(.C = 10^temp,
                            .sigma = 10^temp1)
t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
#rf_tuneGrid <- expand.grid(.mtry= 18)
modelfit_svm_class3 <- train(Cover_Type~.,method='svmRadial',data=training_scaled_sel,
                             trControl = trainControl(method = "repeatedcv", 
                                                      repeats=5), 
                             tuneGrid= tuneGrid_svm)

testing_class15 <- predict(modelfit_svm_class3, testing_scaled[,-c(21,22,29,39)])

temp <- substr(testing_class14,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class14.csv',row.names=F,quote=F)# Naive bayes, feed with factor variables rather than dummy variables
############ Accurary 0.72846

########## Tune svm again 
tuneGrid_svm <- expand.grid(.C = c(0.1,1,10,100),
                            .sigma = c(0.2,0.4,0.6,0.8))
t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
modelfit_svm_class4 <- train(Cover_Type~.,method='svmRadial',data=training_scaled_sel,
                             trControl = trainControl(method = "repeatedcv", 
                                                      repeats=2), 
                             tuneGrid= tuneGrid_svm)
t2 <- Sys.time()
t2-t1

testing_class15 <- predict(modelfit_svm_class3, testing_scaled[,-c(21,22,29,39)])

temp <- substr(testing_class15,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class15.csv',row.names=F,quote=F)# Naive bayes, feed with factor variables rather than dummy variables


############################### Try GBM tune again 
##### interaction.depth from 17-25, n.trees from 2400-4000,
gbmGrid <- expand.grid(.interaction.depth = seq(17,25, by=2),
                       .n.trees = seq(2400,4000,by=400),
                       .shrinkage =  0.01)

t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
modelfit_gbm_dummy2 <- train(Cover_Type~., data=training_scaled_sel,
                             method = "gbm",
                             tuneGrid = gbmGrid,
                             trControl = trainControl(method = "repeatedcv",  
                                                      repeats = 2),
                             ## The gbm() function produces copious amounts
                             ## of output, so pass in the verbose option
                             ## to avoid printing a lot to the screen
                             verbose = F)
t2 <- Sys.time()
t2-t1


testing_class19 <- predict(modelfit_gbm_dummy2, testing_scaled[,-c(21,22,29,39)])

temp <- substr(testing_class19,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class19.csv',row.names=F,quote=F)# Naive bayes, feed with factor variables rather than dummy variables

###################### best try 0.76569
################################# Try rf with deeper trees 

t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
rf_tuneGrid <- expand.grid(.mtry= 18)
modelfit_rf_class6<-train(Cover_Type~.,method='rf',data=training_scaled_sel,
                          trControl = trainControl(method = "repeatedcv", 
                                                   repeats=20), 
                          ntree=3000, tuneGrid=rf_tuneGrid)
t2 <- Sys.time()
t2-t1

testing_class13 <- predict(modelfit_rf_class6, testing_scaled[,-c(21,22,29,39)])

temp <- substr(testing_class13,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class13.csv',row.names=F,quote=F)

######################### Best entry 0.75644
################################# Try rf with more folds 

t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
rf_tuneGrid <- expand.grid(.mtry= 18)
modelfit_rf_class7<-train(Cover_Type~.,method='rf',data=training_scaled_sel,
                          trControl = trainControl(method = "repeatedcv", 
                                                   repeats=2,
                                                   number=50), 
                          ntree=1000, tuneGrid=rf_tuneGrid)
t2 <- Sys.time()
t2-t1

testing_class14 <- predict(modelfit_rf_class7, testing_scaled[,-c(21,22,29,39)])

temp <- substr(testing_class14,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class14.csv',row.names=F,quote=F)# Naive bayes, feed with factor variables rather than dummy variables

################################# Try rf with more folds 

t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
rf_tuneGrid <- expand.grid(.mtry= 18)
modelfit_rf_class8<-train(Cover_Type~.,method='rf',data=training_scaled_sel,
                          trControl = trainControl(method = "repeatedcv", 
                                                   repeats=2,
                                                   number=500), 
                          ntree=1000, tuneGrid=rf_tuneGrid)
t2 <- Sys.time()
t2-t1

testing_class16 <- predict(modelfit_rf_class8, testing_scaled[,-c(21,22,29,39)])

temp <- substr(testing_class16,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class16.csv',row.names=F,quote=F)# Naive bayes, feed with factor variables rather than dummy variables

### try neural networks
t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
my.grid <- expand.grid(.decay = c(0.5, 0.1, 0), 
                       .size = c(5, 6, 7))
modelfit_nn <- train(Cover_Type~., data = training_scaled_sel,
                     trControl = trainControl(method = "repeatedcv", 
                                              repeats=2),
                      method = "nnet", 
                     maxit = 1000, 
                     tuneGrid = my.grid) 
t2 <- Sys.time()
t2-t1

testing_class17 <- predict(modelfit_nn, testing_scaled[,-c(21,22,29,39)])

temp <- substr(testing_class17,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class17.csv',row.names=F,quote=F)# Naive bayes, feed with factor variables rather than dummy variables

# Tune neural net again 
t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
my.grid <- expand.grid(.decay = c(0.05, 0.1, 0.2), 
                       .size = c(1, 3, 10, 15, 20))
modelfit_nn1 <- train(Cover_Type~., data = training_scaled_sel,
                     trControl = trainControl(method = "repeatedcv", 
                                              repeats=2),
                     method = "nnet", 
                     maxit = 1000, 
                     tuneGrid = my.grid) 
t2 <- Sys.time()
t2-t1



##################
# tuneGrid_svm <- expand.grid(.C = c(0.1,1,10,100),
#                             .sigma = c(0.2,0.4,0.6,0.8))
t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
modelfit_svm_class5 <- train(Cover_Type~.,method='svmPoly',data=training_scaled_sel,
                             trControl = trainControl(method = "repeatedcv", 
                                                      repeats=2), 
                             tuneLength=10)
t2 <- Sys.time()
t2-t1


# Tune neural net again 
t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
my.grid <- expand.grid(.decay = c(0.01,0.001,0.0001), 
                       .size = c(20,30,40,80,100,200))
modelfit_nn2 <- train(Cover_Type~., data = training_scaled_sel,
                      trControl = trainControl(method = "repeatedcv", 
                                               repeats=2),
                      method = "nnet", 
                      maxit = 1000, 
                      tuneGrid = my.grid) 
t2 <- Sys.time()
t2-t1

################# Try rf with deeper trees
t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
rf_tuneGrid <- expand.grid(.mtry= 18)
modelfit_rf_class9<-train(Cover_Type~.,method='rf',data=training_scaled_sel,
                          trControl = trainControl(method = "none"), 
                          ntree=10000, 
                          tuneGrid=rf_tuneGrid)
t2 <- Sys.time()
t2-t1

testing_class18 <- predict(modelfit_rf_class9, testing_scaled[,-c(21,22,29,39)])

temp <- substr(testing_class18,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class18.csv',row.names=F,quote=F)# Naive bayes, feed with factor variables rather than dummy variables

########################################################3
###### model_averaged_neural nets
t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
modelfit_avNNet <- train(Cover_Type~., data = training_scaled_sel,
                      trControl = trainControl(method = "repeatedcv", 
                                               repeats=2),
                      method = "avNNet", 
                      maxit = 1000, 
                      tuneLength=10) 
t2 <- Sys.time()
t2-t1

#################################################### gbm tune again 
gbmGrid <- expand.grid(.interaction.depth = seq(30,80, by=10),
                       .n.trees = seq(5000,15000,by=2000),
                       .shrinkage =  0.01)

t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
modelfit_gbm_dummy3 <- train(Cover_Type~., data=training_scaled_sel,
                             method = "gbm",
                             tuneGrid = gbmGrid,
                             trControl = trainControl(method = "repeatedcv",  
                                                      repeats = 1),
                             ## The gbm() function produces copious amounts
                             ## of output, so pass in the verbose option
                             ## to avoid printing a lot to the screen
                             verbose = F)
t2 <- Sys.time()
t2-t1

testing_class20 <- predict(modelfit_gbm_dummy3, testing_scaled[,-c(21,22,29,39)])

temp <- substr(testing_class20,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class20.csv',row.names=F,quote=F)# Naive bayes, feed with factor variables rather than dummy variables

##################### Best accuracy 0.77544
#################### Tune C50 again 

tuneGrid_C50 <- expand.grid(.trials= 100, 
                            .model= 'rules',
                            .winnow = FALSE
                              )
t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
modelfit_C50_2 <- train(Cover_Type~.,data=training_scaled_sel,
                      method = "C5.0", tuneGrid = tuneGrid_C50,
                      trControl = trainControl(method = 'none'),
                      control = C5.0Control(earlyStopping = FALSE))
t2 <- Sys.time()
t2-t1

testing_class21 <- predict(modelfit_C50_2, testing_scaled[,-c(21,22,29,39)])

temp <- substr(testing_class21,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class21.csv',row.names=F,quote=F)

############################## Accuracy 0.74165

# Naive bayes, feed with factor variables rather than dummy variables

############ AdaBag


############### Conditional Inference Random Forest cforest
t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
modelfit_crf <- train(Cover_Type~., data = training_scaled_sel,
                             trControl = trainControl(method = "repeatedcv", 
                                                      repeats=1),
                             method = "cforest",
                             tuneLength=10) 
t2 <- Sys.time()
t2-t1


#################### Not good even on training set 

########## extreme learning machine elm
tuneGrid_elm <- expand.grid(.nhid = seq(1,20,by=2),
                            .actfun = c('sig',
                                        'sin',
                                        'radbas',
                                        'hardlim',
                                        'hardlims',
                                        'satlins',
                                        'tansig',
                                        'tribas',
                                        'poslin',
                                        'purelin'))
t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
modelfit_elm <- train(Cover_Type~., data = training_scaled_sel,
                      trControl = trainControl(method = "repeatedcv", 
                                               repeats=1),
                      method = "elm", tuneGrid=tuneGrid_elm) 
t2 <- Sys.time()
t2-t1


############### Tune again 
tuneGrid_elm <- expand.grid(.nhid = seq(30,210,by=20),
                            .actfun = c('sig',
                                        'sin',
                                        'radbas',
                                        'hardlim',
                                        'hardlims',
                                        'satlins',
                                        'tansig',
                                        'tribas',
                                        'poslin',
                                        'purelin'))
t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
modelfit_elm1 <- train(Cover_Type~., data = training_scaled_sel,
                      trControl = trainControl(method = "repeatedcv", 
                                               repeats=1),
                      method = "elm", tuneGrid=tuneGrid_elm) 
t2 <- Sys.time()
t2-t1


# TUne again
tuneGrid_elm2 <- expand.grid(.nhid = 2^c(8:12),
                            .actfun = c('sig',
                                        'sin',
                                        'radbas',
                                        'hardlim',
                                        'hardlims',
                                        'satlins',
                                        'tansig',
                                        'tribas',
                                        'poslin',
                                        'purelin'))
t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
modelfit_elm1 <- train(Cover_Type~., data = training_scaled_sel,
                       trControl = trainControl(method = "repeatedcv", 
                                                repeats=1),
                       method = "elm", tuneGrid=tuneGrid_elm) 
t2 <- Sys.time()
t2-t1

####################################### No good

########### Random Forest by Randomization  extraTrees
t1 <- Sys.time()
#registerDoMC(cores=4)
set.seed(1990)
modelfit_extratrees <- train(Cover_Type~., data = training_scaled_sel,
                         trControl = trainControl(method = "repeatedcv", 
                                                  repeats=1),
                         method = "extraTrees", 
                         ntree = 1000, 
                         tuneLength=10) 
t2 <- Sys.time()
t2-t1

##########  Boosted Generalized Additive Model  gamboost
training_trans_sel <- training_trans[,-c(21,22,29,39)]
t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
modelfit_gamboost <- train(Cover_Type~., data = training_trans_sel,
                             trControl = trainControl(method = "repeatedcv", 
                                                      repeats=1),
                             method = "gamboost") 
t2 <- Sys.time()
t2-t1

########################################## not working


########## glmnet  glmnet
t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
modelfit_glmnet <- train(Cover_Type~., data = training_scaled_sel,
                           trControl = trainControl(method = "repeatedcv", 
                                                    repeats=1),
                           method = "glmnet", family= 'multinomial') 
t2 <- Sys.time()
t2-t1





###########  Linear Discriminant Analysis  lda2

###########   Boosted Logistic Regression  LogitBoost

############3   Regularized Random Forest  RRF


t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
rf_tuneGrid <- expand.grid(.mtry= 18)
modelfit_rrf<-train(Cover_Type~.,method='RRF',data=training_scaled_sel,
                          trControl = trainControl(method = 'none'),
  						tuneGrid = rf_tuneGrid, 					   
                          ntree=1000)
t2 <- Sys.time()
t2-t1

################################# Too too slow



############################################## Try average the prediction probablity 
########################### of rf and gbm 

pred1 <- predict(modelfit_rf_class6, testing_scaled[,-c(21,22,29,39)], type='prob')
pred2 <- predict(modelfit_gbm_dummy3, testing_scaled[,-c(21,22,29,39)], type='prob')


















# Try rpart with factor variables

# The randomForest function has a limitation that all factor predictors must
# not have more than 32 levels.


