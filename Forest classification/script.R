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
                   importance=T,ntree=1000)testing_class25 <- testing_class20
testing_class25[testing_class13=='C2'] <- 'C2'
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
testing_scaled_sel <- testing_scaled[,-c(21,22,29,39)]
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
#  tuneGrid_svm <- expand.grid(.degree= c(3,4,5),
#                              .C = c(0.1,1,10,100,1000),
#                              .scale = c(0.2,0.4,0.6,0.8))
t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
modelfit_svm_class5 <- train(Cover_Type~.,method='svmPoly',data=training_scaled_sel,
                             trControl = trainControl(method = "repeatedcv", 
                                                      repeats=1))
t2 <- Sys.time()
t2-t1


############################ Tune again 
tuneGrid_svm <- expand.grid(.degree= c(3,4,5),
                             .C = c(0.1,1,10,100, 1000),
                             .scale = c(0.1,0.3,0.6,0.9))
t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
modelfit_svm_class6 <- train(Cover_Type~.,method='svmPoly',data=training_scaled_sel,
                             trControl = trainControl(method = "repeatedcv", 
                                                      repeats=1),
                             tuneGrid = tuneGrid_svm)
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

training_accu_gbm <-predict(modelfit_gbm_dummy3$finalModel, training_scaled_sel[,-51])

testing_class20 <- predict(modelfit_gbm_dummy3, testing_scaled[,-c(21,22,29,39)])

temp <- substr(testing_class20,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class20.csv',row.names=F,quote=F)# Naive bayes, feed with factor variables rather than dummy variables
##################### Best accuracy 0.77544

gbmGrid <- expand.grid(.interaction.depth = 45,
                       .n.trees = 11000,
                       .shrinkage =  0.01)

t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
modelfit_gbm_dummy4 <- train(Cover_Type~., data=training_scaled_sel,
                             method = "gbm",
                             tuneGrid = gbmGrid,
                             trControl = trainControl(method = "none"),
                             ## The gbm() function produces copious amounts
                             ## of output, so pass in the verbose option
                             ## to avoid printing a lot to the screen
                             verbose = F)
t2 <- Sys.time()
t2-t1

testing_class26 <- predict(modelfit_gbm_dummy4, testing_scaled[,-c(21,22,29,39)])
temp <- substr(testing_class26,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class26.csv',row.names=F,quote=F)
############### Accuracy 0.77631

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

############ Adaboost.M1
t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
modelfit_Adaboost <- train(Cover_Type~., data = training_scaled_sel,
                      trControl = trainControl(method = "repeatedcv", 
                                               repeats=1),
                      method = "AdaBoost.M1",
                      tuneLength=10) 
t2 <- Sys.time()
t2-t1


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
extratreesGrid <- expand.grid(.mtry=seq(30,50, by=5),
                              .numRandomCuts=1)
t1 <- Sys.time()
#registerDoMC(cores=4)
set.seed(1990)
modelfit_extratrees <- train(Cover_Type~., data = training_scaled_sel,
                         trControl = trainControl(method = 'repeatedcv',
                                                  repeats=1),
                         method = "extraTrees", 
                         ntree = 3000, 
                         tuneGrid = extratreesGrid) 
t2 <- Sys.time()
t2-t1
testing_class48 <- predict(modelfit_extratrees, testing_scaled_sel)
temp <- substr(testing_class48,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing_scaled_sel),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class48.csv',row.names=F,quote=F)
################################## Accuracy  0.78121
extratreesGrid <- expand.grid(.mtry=45,
                              .numRandomCuts=1)
t1 <- Sys.time()
#registerDoMC(cores=4)
set.seed(1990)
modelfit_extratrees_class1 <- train(Cover_Type~., data = training_scaled_sel[Intrain, ],
                             trControl = trainControl(method = 'none'),
                             method = "extraTrees", 
                             ntree = 3000, 
                             tuneGrid = extratreesGrid) 
t2 <- Sys.time()
t2-t1

pred_extratrees <- predict(modelfit_extratrees_class1, training_scaled_sel[-Intrain,])
extratrees_conf <- confusionMatrix(pred_extratrees, training_scaled_sel[-Intrain, 51])
extratrees_accu <- diag(extratrees_conf$table)/apply(extratrees_conf$table, 1, sum)


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
pred3 <- (pred1 + pred2)/2
pred4 <- as.matrix(apply(pred3, 1, which.max))
pred4 <- cbind(rownames(pred1),pred4)
colnames(pred4) <- c('Id','Cover_type')
write.csv(pred4,'testing_class22.csv',row.names=F,quote=F)

######################### Best try accuracy 0.77582

##########################################  Try using voting 
vote <- as.matrix(data.frame( vote1 = testing_class13, vote2 = testing_class20, vote3 = testing_class21))
rownames(vote) <- rownames(testing)
temp <- sapply(1:nrow(vote), function(i) {
  uniq <- unique(vote[i,])
  temp <- table(vote[i,])
  uniq[which.max(temp)]
})
vote1<- matrix(nrow=nrow(vote), ncol=2)
vote1[,1] <- rownames(vote)
vote1[,2] <- substr(temp,2,2)
colnames(vote1) <- c('Id','Cover_type')
write.csv(vote1, 'testing_class23.csv', quote = F, row.names=F)

######################### accuracy only 0.76...

#########################  deepnet
t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
modelfit_deepnet<-train(Cover_Type~.,method='dnn',data=training_scaled_sel,
                    trControl = trainControl(method = 'repeatedcv', 
                                             repeats=1))
t2 <- Sys.time()
t2-t1

################################################################
# 
# t1 <- Sys.time()
# modelfit_gbm_dummy4 <- gbm.fit(training_scaled_sel[,-51], 
#                                training_scaled_sel$Cover_Type,
#                                n.trees = 11000, 
#                                distribution = 'multinomial',
#                                interaction.depth = 40, 
#                                shrinkage = 0.01)
# t2 <- Sys.time()
# t2-t1
# 

#################################################################
gbmGrid <- expand.grid(.interaction.depth = 45,
                       .n.trees = 11000,
                       .shrinkage =  0.01)

t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
modelfit_gbm_dummy5 <- train(Cover_Type~., data=training_scaled_sel[Intrain,],
                             method = "gbm",
                             tuneGrid = gbmGrid,
                             trControl = trainControl(method = "none"),
                             ## The gbm() function produces copious amounts
                             ## of output, so pass in the verbose option
                             ## to avoid printing a lot to the screen
                             verbose = F)
t2 <- Sys.time()
t2-t1

gbm_test <- predict(modelfit_gbm_dummy5, training_scaled_sel[-Intrain,-51])
gbm_test_prob <- predict(modelfit_gbm_dummy5, training_scaled_sel[-Intrain,-51], type='prob')

gbm_conf <- confusionMatrix(gbm_test, training_scaled_sel[-Intrain,51])
gbm_class_accu <- diag(gbm_conf$table)/apply(gbm_conf$table, 1, sum)

t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
rf_tuneGrid <- expand.grid(.mtry= 18)
modelfit_rf_class10<-train(Cover_Type~.,method='rf',data=training_scaled_sel[Intrain,],
                          trControl = trainControl(method = "none"), 
                          ntree=3000, tuneGrid=rf_tuneGrid)
t2 <- Sys.time()
t2-t1

rf_test <- predict(modelfit_rf_class10, training_scaled_sel[-Intrain,-51])
rf_test_prob <- predict(modelfit_rf_class10, training_scaled_sel[-Intrain,-51], type='prob')
rf_conf <- confusionMatrix(rf_test, training_scaled_sel[-Intrain,51])
rf_class_accu <- diag(rf_conf$table)/apply(rf_conf$table, 1, sum)






rf6_class_accu <- 1 - modelfit_rf_class6$finalModel$confusion[,8]
weight_gbm <- gbm_class_accu/(gbm_class_accu + rf6_class_accu)
weight_rf6 <- rf6_class_accu/(gbm_class_accu + rf6_class_accu)
pred5 <- sweep(pred1, 2, weight_rf6, '*') + sweep(pred2, 2, weight_gbm, '*')
pred6 <- as.matrix(apply(pred5, 1, which.max))
pred6 <- cbind(rownames(pred1),pred6)
colnames(pred6) <- c('Id','Cover_type')
write.csv(pred6,'testing_class24.csv',row.names=F,quote=F)

##### Not much difference between pred6 and pred4
########################################################################
### take C2 from rf
### take C1 from gbm
### take C6 from gbm
### take C3 from gbm
### take C5 from gbm
### take C4 from gbm
### take C7 from gbm
testing_class25 <- testing_class20
testing_class25[testing_class13=='C2'] <- 'C2'

temp <- substr(testing_class25,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class25.csv',row.names=F,quote=F)

########### best accuracy 0.78551
testing_class27 <- testing_class26
testing_class27[testing_class13=='C2'] <- 'C2'

temp <- substr(testing_class27,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class27.csv',row.names=F,quote=F)

########### accuracy 0.78585

##### sub C2 from extratrees
testing_class55 <- testing_class26
testing_class55[testing_class48=='C2'] <- 'C2'

temp <- substr(testing_class55,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class55.csv',row.names=F,quote=F)
###################### 0.79226

######### sub gbm C1 extratrees
testing_class56 <- testing_class48
testing_class56[testing_class26=='C1'] <- 'C1'

temp <- substr(testing_class56,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class56.csv',row.names=F,quote=F)
############## Accuracy 0.77828

############# sub C1 and C2 from extratrees
testing_class57 <- testing_class26
testing_class57[testing_class48=='C2'] <- 'C2'
testing_class57[testing_class48=='C1'] <- 'C1'

temp <- substr(testing_class57,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class57.csv',row.names=F,quote=F)
######### Accuracy 0.79425

############### take C2 from ensembel1_test
testing_class58 <- testing_class57
testing_class58[ensembel1_test=='C2'] <- 'C2'

temp <- substr(testing_class58,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class58.csv',row.names=F,quote=F)
########## accuracy 0.80283

################## take C1 and C2 from class58
testing_class59 <- testing_class43
testing_class59[testing_class58=='C2'] <- 'C2'
testing_class59[testing_class58=='C1'] <- 'C1'

temp <- substr(testing_class59,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class59.csv',row.names=F,quote=F)
############## Accuracy 0.80768

################ Try use vote class54, class57, class43
vote14 <- as.matrix(data.frame( vote1 = testing_class57, vote2 = testing_class54, vote3 = testing_class43))
rownames(vote14) <- rownames(testing)
temp <- sapply(1:nrow(vote14), function(i) {
  uniq <- sort(unique(vote14[i,]))
  temp <- table(vote14[i,])
  uniq[which.max(temp)]
})
vote15<- matrix(nrow=nrow(vote14), ncol=2)
vote15[,1] <- rownames(vote14)
vote15[,2] <- substr(temp,2,2)
colnames(vote15) <- c('Id','Cover_type')
write.csv(vote15, 'testing_class60.csv', quote = F, row.names=F)









############### Try svm on Intrain sets
tuneGrid_svm <- expand.grid(.C = 100,
                            .sigma = 0.1)
t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
modelfit_svm_class1 <- train(Cover_Type~.,method='svmRadial',data=training_scaled_sel[Intrain,],
                             trControl = trainControl(method = "none"), 
                             tuneGrid= tuneGrid_svm)
t2 <- Sys.time()
t2-t1

svm_test <- predict(modelfit_svm_class1, training_scaled_sel[-Intrain,])
svm_conf <- confusionMatrix(svm_conf, training_scaled_sel[Intrain,] )
svm_accu <- diag(svm_conf$table)/apply(svm_conf$table, 1, sum)

########### See C50
tuneGrid_C50 <- expand.grid(.trials= 100, 
                            .model= 'rules',
                            .winnow = FALSE
)
t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
modelfit_C50_3 <- train(Cover_Type~.,data=training_scaled_sel[Intrain,],
                        method = "C5.0", tuneGrid = tuneGrid_C50,
                        trControl = trainControl(method = 'none'),
                        control = C5.0Control(earlyStopping = FALSE))
t2 <- Sys.time()
t2-t1

C50_test <- predict(modelfit_C50_3, training_scaled_sel[-Intrain,])
C50_conf <- confusionMatrix(C50_test, training_scaled_sel[-Intrain, 51])
C50_accu <- diag(C50_conf$table)/apply(C50_conf$table, 1, sum)


#######################################################################
#######################################################3 
############ Try multiclass metrics rather than accuracy 
#########33 rf 
############################# only for C1 and C2
index <- training_scaled_sel$Cover_Type =='C1'|training_scaled_sel$Cover_Type =='C2'
training_scaled_sel_subset <- subset(training_scaled_sel, index)
training_scaled_sel_subset$Cover_Type <- as.factor(as.character(training_scaled_sel_subset$Cover_Type))

t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
rf_tuneGrid <- expand.grid(.mtry= seq(14, 24, by = 2))
modelfit_rf_type2<-train(Cover_Type~.,method='rf',data=training_scaled_sel_subset,
                          trControl = trainControl(method = "repeatedcv", 
                                                   repeats=1,
                                                   summaryFunction = twoClassSummary,
                                                   classProbs = TRUE), 
                          ntree=1000, 
                          tuneGrid=rf_tuneGrid,
                          metric = 'ROC')
t2 <- Sys.time()
t2-t1

t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
rf_tuneGrid <- expand.grid(.mtry= seq(30, 50, by = 5))
modelfit_rf_type3<-train(Cover_Type~.,method='rf',data=training_scaled_sel_subset,
                         trControl = trainControl(method = "repeatedcv", 
                                                  repeats=1,
                                                  summaryFunction = twoClassSummary,
                                                  classProbs = TRUE), 
                         ntree=1000, 
                         tuneGrid=rf_tuneGrid,
                         metric = 'ROC')
t2 <- Sys.time()
t2-t1

################################# Use gbm to tune C1 and C2
gbmGrid <- expand.grid(.interaction.depth = c(25,35,45),
                       .n.trees = seq(5000,14000,by=3000),
                       .shrinkage =  0.01)

t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
modelfit_gbm_dummy6 <- train(Cover_Type~., data=training_scaled_sel_subset,
                             method = "gbm",
                             tuneGrid = gbmGrid,
                             trControl = trainControl(method = "repeatedcv",  
                                                      repeats = 1,
                                                      summaryFunction = twoClassSummary,
                                                      classProbs = TRUE),
                             ## The gbm() function produces copious amounts
                             ## of output, so pass in the verbose option
                             ## to avoid printing a lot to the screen
                             verbose = F)
t2 <- Sys.time()
t2-t1

#######################################
############# use gbm_test and rf_test as features
rf_gbm_ensembel <- data.frame(feature1 = gbm_test ,feature2 = rf_test,
                              obs = training[-Intrain,]$Cover_Type)
gbmGrid <- expand.grid(.interaction.depth = 1,
                       .n.trees = c(1000, 5000, 10000),
                       .shrinkage =  0.01)
t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
modelfit_gbm_ensembel <- train(obs~., data=rf_gbm_ensembel,
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

rf_gbm_test <- data.frame(feature1 = testing_class26,
                          feature2 = testing_class13)
testing_class30 <- predict(modelfit_gbm_ensembel, rf_gbm_test)

temp <- substr(testing_class30,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class30.csv',row.names=F,quote=F)

############################################### use prob of rf and gbm as features
ensemble1 <- cbind(rf_test_prob, gbm_test_prob)
colnames(ensemble1) <- c(paste('C',1:7,'rf', sep=''), paste('C',1:7,'gbm', sep=''))
ensemble1$obs <- training[-Intrain,]$Cover_Type

t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
rf_tuneGrid <- expand.grid(.mtry= seq(2, 14, by = 2))
modelfit_rf_type4<-train(obs~.,method='rf',data=ensemble1,
                         trControl = trainControl(method = "repeatedcv", 
                                                  repeats=1),                         ntree=1000, 
                         tuneGrid=rf_tuneGrid)
t2 <- Sys.time()
t2-t1

ensembel1_test_input <- cbind(pred1, pred2)
colnames(ensembel1_test_input) <- c(paste('C',1:7,'rf', sep=''), paste('C',1:7,'gbm', sep=''))
ensembel1_test <- predict(modelfit_rf_type4, ensembel1_test_input)

temp <- substr(ensembel1_test,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class33.csv',row.names=F,quote=F)
############################## 0.78385

############ use gbm to train the ensembel prob
gbmGrid <- expand.grid(.interaction.depth = c(1:5),
                       .n.trees = c(100, 500, 1000, 5000),
                       .shrinkage =  0.01)
t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
modelfit_gbm_ensembel_prob <- train(obs~., data=ensemble1,
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

ensembel1_test_gbm <- predict(modelfit_gbm_ensembel_prob, ensembel1_test_input)

temp <- substr(ensembel1_test_gbm,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class35.csv',row.names=F,quote=F)
################ Accuracy 0.78290

################### vote using testing_class13, testing_class26, ensembel1_test
vote2 <- as.matrix(data.frame( vote1 = testing_class26, vote2 = ensembel1_test, vote3 = testing_class13))
rownames(vote2) <- rownames(testing)
temp <- sapply(1:nrow(vote2), function(i) {
  uniq <- sort(unique(vote2[i,]))
  temp <- table(vote2[i,])
  uniq[which.max(temp)]
})
vote3<- matrix(nrow=nrow(vote2), ncol=2)
vote3[,1] <- rownames(vote2)
vote3[,2] <- substr(temp,2,2)
colnames(vote3) <- c('Id','Cover_type')
write.csv(vote3, 'testing_class34.csv', quote = F, row.names=F)
###################3 Accuracy 0.78028
#####################3 vote using testing_class27, ensembel1_test, emsembel1_test_gbm
vote4 <- as.matrix(data.frame( vote1 = testing_class27, vote2 = ensembel1_test, vote3 = ensembel1_test))
rownames(vote4) <- rownames(testing)
temp <- sapply(1:nrow(vote4), function(i) {
  uniq <- sort(unique(vote4[i,]))
  temp <- table(vote4[i,])
  uniq[which.max(temp)]
})
vote5<- matrix(nrow=nrow(vote4), ncol=2)
vote5[,1] <- rownames(vote4)
vote5[,2] <- substr(temp,2,2)
colnames(vote5) <- c('Id','Cover_type')
write.csv(vote5, 'testing_class36.csv', quote = F, row.names=F)
############## 0.78358

testing_class37 <- testing_class27
testing_class37[ensembel1_test=='C2'] <- 'C2'
temp <- substr(testing_class37,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class37.csv',row.names=F,quote=F)
############# best Accuracy 0.79359
testing_class38 <- testing_class26
testing_class38[ensembel1_test=='C2'] <- 'C2'
temp <- substr(testing_class38,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class38.csv',row.names=F,quote=F)
############## Accuracy 0.79151
testing_class39 <- ensembel1_test
testing_class39[testing_class26=='C1'] <- 'C1'
temp <- substr(testing_class39,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class39.csv',row.names=F,quote=F)
###### Accuracy 0.78194
testing_class40 <- testing_class37
testing_class40[ensembel1_test_gbm=='C1'] <- 'C1'
temp <- substr(testing_class40,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class40.csv',row.names=F,quote=F)
################ Accuracy 0.78680
testing_class41 <- testing_class37
testing_class41[ensembel1_test=='C3'] <- 'C3'
temp <- substr(testing_class41,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class41.csv',row.names=F,quote=F)
################## Accuracy 0.79376

############### Vote using 4 
vote6 <- as.matrix(data.frame( vote1 = testing_class26, vote2 = ensembel1_test, vote3 = ensembel1_test_gbm,
                               vote4 = testing_class13))
rownames(vote6) <- rownames(testing)
temp <- sapply(1:nrow(vote6), function(i) {
  uniq <- sort(unique(vote6[i,]))
  temp <- table(vote6[i,])
  uniq[which.max(temp)]
})
vote7<- matrix(nrow=nrow(vote6), ncol=2)
vote7[,1] <- rownames(vote6)
vote7[,2] <- substr(temp,2,2)
colnames(vote7) <- c('Id','Cover_type')
write.csv(vote7, 'testing_class42.csv', quote = F, row.names=F)
############## Accuracy 0.78291 , made a mistake (replicate ensembel1_test), rerun again 
############## correct is 0.78745

###########################################
ensemble1_combined <- cbind(ensemble1, rf_test, gbm_test)
colnames(ensemble1_combined) <- c(paste('C',1:7,'rf', sep=''), 
                                  paste('C',1:7,'gbm', sep=''),
                                  'obs', 'rf_pred','gbm_pred')
t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
rf_tuneGrid <- expand.grid(.mtry= 1:5)
modelfit_rf_type5<-train(obs~.,method='rf',data=ensemble1_combined,
                         trControl = trainControl(method = "repeatedcv", 
                                                  repeats=1),                        
                         ntree=3000, 
                         tuneGrid=rf_tuneGrid)
t2 <- Sys.time()
t2-t1

ensembel1_test_input_combined <- cbind(ensembel1_test_input, testing_class13,
                                       testing_class26)
colnames(ensembel1_test_input_combined) <- c(paste('C',1:7,'rf', sep=''), 
                                             paste('C',1:7,'gbm', sep=''),
                                              'rf_pred','gbm_pred')
testing_class43 <- predict(modelfit_rf_type5, ensembel1_test_input_combined)

temp <- substr(testing_class43,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class43.csv',row.names=F,quote=F)
########### Accuracy 0.79190

################## Try gbm this time 
gbmGrid <- expand.grid(.interaction.depth = c(1:5),
                       .n.trees = c(100, 500, 1000, 5000),
                       .shrinkage =  0.01)
t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
modelfit_gbm_ensembel_combined <- train(obs~., data=ensemble1_combined,
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

testing_class47 <- predict(modelfit_gbm_ensembel_combined, ensembel1_test_input_combined)

temp <- substr(testing_class47,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class47.csv',row.names=F,quote=F)
############### Accuracy 0.78323


###################### 
testing_class44 <- testing_class41
testing_class44[testing_class43=='C1'] <- 'C1'
temp <- substr(testing_class44,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class44.csv',row.names=F,quote=F)
############### Accuracy 0.79424

testing_class45 <- testing_class43
testing_class45[testing_class41=='C2'] <- 'C2'
temp <- substr(testing_class45,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class45.csv',row.names=F,quote=F)
############## Accuracy 0.80131
testing_class46 <- testing_class45
testing_class46[testing_class41=='C7'] <- 'C7'
temp <- substr(testing_class46,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class46.csv',row.names=F,quote=F)
##### 

testing_class49 <- testing_class43
testing_class49[ensembel1_test=='C2'] <- 'C2'
temp <- substr(testing_class49,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class49.csv',row.names=F,quote=F)
########### Accuracy 0.79375

############ try vote again 
######## rf, gbm, rf_prob, rf_prob_combined
vote8 <- as.matrix(data.frame( vote1 = testing_class43, vote2 = testing_class26, vote3 = ensembel1_test,
                               vote4 = testing_class13))
rownames(vote8) <- rownames(testing)
temp <- sapply(1:nrow(vote8), function(i) {
  uniq <- sort(unique(vote8[i,]))
  temp <- table(vote8[i,])
  uniq[which.max(temp)]
})
vote9<- matrix(nrow=nrow(vote8), ncol=2)
vote9[,1] <- rownames(vote8)
vote9[,2] <- substr(temp,2,2)
colnames(vote9) <- c('Id','Cover_type')
write.csv(vote9, 'testing_class50.csv', quote = F, row.names=F)
############## Accuracy 0.79215

########### add gbm_prob methods, 6 votes

vote8 <- as.matrix(data.frame( vote1 = testing_class43, vote2 = testing_class26, vote3 = ensembel1_test,
                               vote4=ensembel1_test_gbm, vote5= testing_class47, vote6 = testing_class13))
rownames(vote8) <- rownames(testing)
temp <- sapply(1:nrow(vote8), function(i) {
  uniq <- sort(unique(vote8[i,]))
  temp <- table(vote8[i,])
  uniq[which.max(temp)]
})
vote9<- matrix(nrow=nrow(vote8), ncol=2)
vote9[,1] <- rownames(vote8)
vote9[,2] <- substr(temp,2,2)
colnames(vote9) <- c('Id','Cover_type')
write.csv(vote9, 'testing_class51.csv', quote = F, row.names=F)
##################  accuracy 0.78807

vote10 <- as.matrix(data.frame( vote1 = testing_class43, vote2 = testing_class26, vote3 = ensembel1_test))
rownames(vote10) <- rownames(testing)
temp <- sapply(1:nrow(vote10), function(i) {
  uniq <- sort(unique(vote10[i,]))
  temp <- table(vote10[i,])
  uniq[which.max(temp)]
})
vote11<- matrix(nrow=nrow(vote10), ncol=2)
vote11[,1] <- rownames(vote10)
vote11[,2] <- substr(temp,2,2)
colnames(vote11) <- c('Id','Cover_type')
write.csv(vote11, 'testing_class52.csv', quote = F, row.names=F)

################################ vote using gbm, rf, extratrees
vote12 <- as.matrix(data.frame( vote1 = testing_class48, vote2 = testing_class26, vote3 = testing_class13))
rownames(vote12) <- rownames(testing)
temp <- sapply(1:nrow(vote12), function(i) {
  uniq <- sort(unique(vote12[i,]))
  temp <- table(vote12[i,])
  uniq[which.max(temp)]
})
vote13<- matrix(nrow=nrow(vote12), ncol=2)
vote13[,1] <- rownames(vote12)
vote13[,2] <- substr(temp,2,2)
colnames(vote13) <- c('Id','Cover_type')
write.csv(vote13, 'testing_class53.csv', quote = F, row.names=F)
######################## accuracy 0.77707
########### Combined extratrees with ensembel1_combine
ensemble1_combined2 <- cbind(ensemble1_combined, pred_extratrees)
colnames(ensemble1_combined2) <- c(paste('C',1:7,'rf', sep=''), 
                                  paste('C',1:7,'gbm', sep=''),
                                  'obs', 'rf_pred','gbm_pred','et_pred')
t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
rf_tuneGrid <- expand.grid(.mtry= 1:10)
modelfit_rf_type6<-train(obs~.,method='rf',data=ensemble1_combined2,
                         trControl = trainControl(method = "repeatedcv", 
                                                  repeats=1),                        
                         ntree=3000, 
                         tuneGrid=rf_tuneGrid)
t2 <- Sys.time()
t2-t1

ensembel1_test_input_combined2 <- cbind(ensembel1_test_input, testing_class13,
                                       testing_class26,testing_class48)
colnames(ensembel1_test_input_combined2) <- c(paste('C',1:7,'rf', sep=''), 
                                             paste('C',1:7,'gbm', sep=''),
                                             'rf_pred','gbm_pred','et_pred')
testing_class54 <- predict(modelfit_rf_type6, ensembel1_test_input_combined2)

temp <- substr(testing_class54,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class54.csv',row.names=F,quote=F)
############# Accuracy  0.79138





########################################################
############# using modelfit_rf_type to predict C1 and C2 
testing_class28 <- as.character(testing_class26)
index <- which((testing_class13=='C1'&testing_class26=='C2')|
                 (testing_class26=='C1'&testing_class13=='C2'))
testing_class28_sub <- predict(modelfit_rf_type2, testing_scaled_sel[index,])
testing_class28[index] <- as.character(testing_class28_sub)

temp <- substr(testing_class28,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class28.csv',row.names=F,quote=F)



################# Use gbm_dummy6 to predict C1 and C2
testing_class29 <- as.character(testing_class26)
index <- which((testing_class13=='C1'&testing_class26=='C1')|
                 (testing_class26=='C2'&testing_class13=='C2'))
testing_class29_sub <- predict(modelfit_gbm_dummy6, testing_scaled_sel[index,])
testing_class29[index] <- as.character(testing_class29_sub)

temp <- substr(testing_class29,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class29.csv',row.names=F,quote=F)


############################################### Tune svm
tuneGrid_svm <- expand.grid(.C = 100,
                            .sigma = 0.1)
t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
modelfit_svm_class2 <- train(Cover_Type~.,method='svmRadial',data=training_scaled_sel,
                             trControl = trainControl(method = "none"), 
                             tuneGrid= tuneGrid_svm)
t2 <- Sys.time()
t2-t1

testing_class31 <- predict(modelfit_svm_class2, testing_scaled_sel)

#### Take C7 from svm
testing_class32 <- testing_class27
testing_class32[testing_class31=='C7'] <- 'C7'

temp <- substr(testing_class32,2,2)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing_class32.csv',row.names=F,quote=F)
################# even worse



require(compiler)
#Based on caret:::twoClassSummary
multiClassSummary <- cmpfun(function (data, lev = NULL, model = NULL){
  #Load Libraries
  require(Metrics)
  require(caret)
  #Check data
  if (!all(levels(data[, "pred"]) == levels(data[, "obs"])))
    stop("levels of observed and predicted data do not match")
  #Calculate custom one-vs-all stats for each class
  prob_stats <- lapply(levels(data[, "pred"]), function(class){
    #Grab one-vs-all data for the class
    pred <- ifelse(data[, "pred"] == class, 1, 0)
    obs <- ifelse(data[, "obs"] == class, 1, 0)
    prob <- data[,class]
    
    #Calculate one-vs-all AUC and logLoss and return
    cap_prob <- pmin(pmax(prob, .000001), .999999)
    prob_stats <- c(auc(obs, prob), logLoss(obs, cap_prob))
    names(prob_stats) <- c('ROC', 'logLoss')
    return(prob_stats)
  })
  prob_stats <- do.call(rbind, prob_stats)
  rownames(prob_stats) <- paste('Class:', levels(data[, "pred"]))
  #Calculate confusion matrix-based statistics
  CM <- confusionMatrix(data[, "pred"], data[, "obs"])
  #Aggregate and average class-wise stats
  #Todo: add weights
  class_stats <- cbind(CM$byClass, prob_stats)
  weight <- c(1, 1, rep(0.5, times=5) )
  weight <- weight/sum(weight)
  class_stats <- apply(sweep(class_stats, 1, weight, '*'), 2, sum)
  #Aggregate overall stats
  overall_stats <- c(CM$overall)
  #Combine overall with class-wise stats and remove some stats we don't want
  stats <- c(overall_stats, class_stats)
  stats <- stats[! names(stats) %in% c('AccuracyNull',
                                       'Prevalence', 'Detection Prevalence')]
  #Clean names and return
  names(stats) <- gsub('[[:blank:]]+', '_', names(stats))
  return(stats)
}) 


# ###############################################################
# #################################################### 
# ############## exam validation error of rf, gbm, and prob based rf and gbm
# set.seed(1990)
# Train1 <- createDataPartition(training_scaled_sel$Cover_Type, p = 0.4, list = F)
# Train2 <- createDataPartition(training_scaled_sel[-Train1,]$Cover_Type, p = 2/3, list = F)
# Train_train <- training_scaled_sel[Train1,]
# Train_valid <- training_scaled_sel[Train2,]
# Train_test <- training_scaled_sel[-c(Train1, Train2),]
# ############## Train rf with Train_train
# t1 <- Sys.time()
# registerDoMC(cores=4)
# set.seed(1990)
# rf_tuneGrid <- expand.grid(.mtry= 18)
# modelfit_rf_class11<-train(Cover_Type~.,method='rf',data=Train_train,
#                            trControl = trainControl(method = "none"), 
#                            ntree=3000, tuneGrid=rf_tuneGrid)
# t2 <- Sys.time()
# t2-t1
# 
# modelfit_rf_class12<-train(Cover_Type~.,method='rf',data=Train_valid,
#                            trControl = trainControl(method = "none"), 
#                            ntree=3000, tuneGrid=rf_tuneGrid)
# 
# 
# valid_rf_prob <- predict(modelfit_rf_class11, Train_valid, type='prob')
# test_rf <- predict(modelfit_rf_class12, Train_test)
# test_rf_prob <- predict(modelfit_rf_class12, Train_test, type='prob')
# 
# ############## Train gbm with Train_train 
# gbmGrid <- expand.grid(.interaction.depth = 45,
#                        .n.trees = 11000,
#                        .shrinkage =  0.01)
# set.seed(1990)
# modelfit_gbm_dummy7 <- train(Cover_Type~., data=Train_train,
#                              method = "gbm",
#                              tuneGrid = gbmGrid,
#                              trControl = trainControl(method = "none"),
#                              ## The gbm() function produces copious amounts
#                              ## of output, so pass in the verbose option
#                              ## to avoid printing a lot to the screen
#                              verbose = F)
# 
# modelfit_gbm_dummy8 <- train(Cover_Type~., data=Train_valid,
#                              method = "gbm",
#                              tuneGrid = gbmGrid,
#                              trControl = trainControl(method = "none"),
#                              ## The gbm() function produces copious amounts
#                              ## of output, so pass in the verbose option
#                              ## to avoid printing a lot to the screen
#                              verbose = F)
# 
# valid_gbm_prob <- predict(modelfit_gbm_dummy7, Train_valid, type='prob')
# test_gbm <- predict(modelfit_gbm_dummy8, Train_test)
# test_gbm_prob <- predict(modelfit_gbm_dummy8, Train_test, type='prob')
# 
# ############################ Train using prob of rf and gbm use rf
# ensemble2 <- cbind(valid_rf_prob, valid_gbm_prob)
# colnames(ensemble2) <- c(paste('C',1:7,'rf', sep=''), paste('C',1:7,'gbm', sep=''))
# ensemble2$obs <- Train_valid$Cover_Type
# 
# t1 <- Sys.time()
# registerDoMC(cores=4)
# set.seed(1990)
# rf_tuneGrid <- expand.grid(.mtry= 2)
# modelfit_rf_type5<-train(obs~.,method='rf',data=ensemble2,
#                          trControl = trainControl(method = 'none'),
#                          ntree=3000, 
#                          tuneGrid=rf_tuneGrid)
# t2 <- Sys.time()
# t2-t1
# 
# ensemble2_test <- cbind(test_rf_prob, test_gbm_prob)
# colnames(ensemble2_test) <- c(paste('C',1:7,'rf', sep=''), paste('C',1:7,'gbm', sep=''))
# 
# rf_ensembel2_prob <- predict(modelfit_rf_type5, ensemble2_test)
# 
# ############################ Train using prob of rf and gbm use gbm
# gbmGrid <- expand.grid(.interaction.depth = 2,
#                        .n.trees = 500,
#                        .shrinkage =  0.01)
# t1 <- Sys.time()
# registerDoMC(cores=4)
# set.seed(1990)
# modelfit_gbm_ensembel2_prob <- train(obs~., data=ensemble2,
#                                      method = "gbm",
#                                      tuneGrid = gbmGrid,
#                                      trControl = trainControl(method = 'none'),
#                                      ## The gbm() function produces copious amounts
#                                      ## of output, so pass in the verbose option
#                                      ## to avoid printing a lot to the screen
#                                      verbose = F)
# t2 <- Sys.time()
# t2-t1
# 
# gbm_ensembel2_prob <- predict(modelfit_gbm_ensembel2_prob, ensemble2_test)
# 
# ####################### compare accuracy
# confusionMatrix(test_rf, Train_test$Cover_Type)
# confusionMatrix(test_gbm, Train_test$Cover_Type)
# confusionMatrix(rf_ensembel2_prob, Train_test$Cover_Type)
# confusionMatrix(gbm_ensembel2_prob, Train_test$Cover_Type)

# Try rpart with factor variables

# The randomForest function has a limitation that all factor predictors must
# not have more than 32 levels.


