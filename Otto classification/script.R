library('caret')
library('doMC')
# load training and testing dataset
training <- read.csv('train.csv',header=T,row.names=1)
testing <- read.csv('test.csv',header=T,row.names=1)

# pre-examine the dataset
dim(training)
dim(testing)
colnames(training)


# propertion of NA for each feature
NA_prop <- sapply(training,function(i) {sum(is.na(i))/length(i)})
table(NA_prop)

# Propertion of missing values for each feature 
Missing_prop <- sapply(training,function(i) {sum(i=='')/length(i)})
table(Missing_prop)

# find features with high correlation 
corMat <- cor(training[,-94])
Highcorr <- findCorrelation(corMat,cutoff = 0.9)

# find linear combination 
LinearComb <- findLinearCombos(training[,-94])

## First try of rf
# try rf with smaller range of values 
t1 <- Sys.time()
registerDoMC(cores=40)
set.seed(1990)
# construt the paramter grid for tuning 
rf_tuneGrid <- expand.grid(.mtry= seq(1,93,by = 5))
modelfit_rf_class<-train(target~.,method='rf',data=training,
                          trControl = trainControl(method = "repeatedcv", 
                                                   repeats=1), 
                          ntree=1000, tuneGrid=rf_tuneGrid)
t2 <- Sys.time()
t2-t1

#################################################
## 
t1 <- Sys.time()
registerDoMC(cores=40)
set.seed(1990)
# construt the paramter grid for tuning 
rf_tuneGrid <- expand.grid(.mtry= 17:21)
modelfit_rf_class2<-train(target~.,method='rf',data=training,
                         trControl = trainControl(method = "repeatedcv", 
                                                  repeats=1), 
                         ntree=3000, tuneGrid=rf_tuneGrid)
t2 <- Sys.time()
t2-t1

testing_class <- predcit(modelfit_rf_class2, testing)
temp <- model.matrix(~0+testing_class)
temp <-cbind(1:length(testing_class),temp)
colnames(temp) <- c('id', paste('Class',1:9,sep='_'))
write.csv2(temp, 'testing_class.csv',row.names=F,quote=F)

##################################################
### tune gbm




