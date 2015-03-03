library('caret')
library('doMC')
setwd('../Dropbox/gitrepository/Kaggle/Forest classification/')
training <- read.csv('train.csv',header=T,row.names=1)
testing <- read.csv('test.csv',header=T,row.names=1)
training$Cover_Type <- as.factor(paste('C',training$Cover_Type,sep=''))


# ######### Impute missing feature values for Hillshade_3pm
# training$Hillshade_3pm[training$Hillshade_3pm==0] <- NA
# 
# training <- rfImpute(training[,-55], training[,55])
# training$Cover_Type <- training[,1]
# training <- training[,-1]
# 

###### Create new features 
newfeature1 <- with (training, Elevation- 0.2*Horizontal_Distance_To_Hydrology)
newfeature2 <- with (training, Elevation- Vertical_Distance_To_Hydrology)

training$newfeature1 <- newfeature1
training$newfeature2 <- newfeature2

train_subset2 <- training[,c(1:10, 56,57)]
mean_subset2 <- apply(train_subset2,2,mean)
sd_subset2 <- apply(train_subset2,2,sd)
train_subset2_scaled <- sweep(train_subset2,2,mean_subset2,'-')
train_subset2_scaled <- sweep(train_subset2_scaled,2,sd_subset2,'/')
training_scaled <- training
training_scaled[,c(1:10, 56,57)] <- train_subset2_scaled
training_scaled_sel <- training_scaled[,-c(21,22,29,39)]
testing_scaled_sel <- testing_scaled[,-c(21,22,29,39)]

newfeature1_test <- with (testing, Elevation- 0.2*Horizontal_Distance_To_Hydrology)
newfeature2_test <- with (testing, Elevation- Vertical_Distance_To_Hydrology)

testing$newfeature1 <- newfeature1_test
testing$newfeature2 <- newfeature2_test

testing_subset <- testing[,c(1:10, 55,56) ]
testing_subset_scaled <- sweep(testing_subset,2,mean_subset2,'-')
testing_subset_scaled <- sweep(testing_subset_scaled,2,sd_subset2,'/')
testing_scaled <- testing
testing_scaled[,c(1:10, 55,56)] <- testing_subset_scaled

########################### Tune using optimazed parameter mtry=18
t1 <- Sys.time()
registerDoMC(cores=4)
set.seed(1990)
rf_tuneGrid <- expand.grid(.mtry= 18)
modelfit_rf_class1<-train(Cover_Type~.,method='rf',data=training_scaled_sel,
                          trControl = trainControl(method = "none"), 
                          ntree=3000, 
                          tuneGrid=rf_tuneGrid)
t2 <- Sys.time()
t2-t1

testing_class1 <- predict(modelfit_rf_class1, testing_scaled_sel)

temp <- substr(testing_class1,3,3)
temp <- matrix(temp, nrow = length(temp), ncol = 1)
temp <- cbind(rownames(testing),temp)
colnames(temp) <- c('Id','Cover_type')
write.csv(temp,'testing1.csv',row.names=F,quote=F)


############################ Tune gbm using optimized paramater
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






# #####################
# Wilderness <- factor(c(paste('wilder',1:4,sep='')))
# Soiltype <- factor(c(paste('soil',1:40,sep='')))
# train_subset <- training[,15:54]
# train_subset1 <- training[,11:14]
# index_soil <- apply(train_subset,1,function(i) which(i==1))
# index_wild <- apply(train_subset1,1,function(i) which(i==1))
# training_trans <- training[1:10]
# training_trans$Wild <- Wilderness[index_wild]
# training_trans$Soil <- Soiltype[index_soil]
# training_trans[,1:10] <- train_subset2_scaled
# training_trans$Cover_Type <- training$Cover_Type
# 
