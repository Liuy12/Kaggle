ggplot() + geom_point(aes(x=Elevation, y= Aspect, colour=Cover_Type), data= training)
ggplot() + geom_point(aes(x=Elevation, y= Horizontal_Distance_To_Roadways, colour=Cover_Type), data= training)
ggplot() + geom_point(aes(x=Elevation, y= Horizontal_Distance_To_Fire_Points, colour=Cover_Type), data= training)
ggplot() + geom_point(aes(x=Elevation, y= Horizontal_Distance_To_Hydrology, colour=Cover_Type), data= training)
ggplot() + geom_point(aes(x=Elevation, y= Vertical_Distance_To_Hydrology, colour=Cover_Type), data= training)

Combined_hdistance <- with(training, Horizontal_Distance_To_Roadways +
                             Horizontal_Distance_To_Fire_Points +
                             Horizontal_Distance_To_Hydrology)

ggplot() + geom_point(aes(x=Elevation, y= Combined_hdistance, colour=Cover_Type), data= training)
 #### Not working 
##############################
ggplot() + geom_point(aes(x=Elevation- 0.2*Horizontal_Distance_To_Hydrology, y= Horizontal_Distance_To_Hydrology, colour=Cover_Type), data= training)

ggplot() + geom_point(aes(x=Elevation- Vertical_Distance_To_Hydrology, y= Vertical_Distance_To_Hydrology, colour=Cover_Type), data= training)

newfeature1 <- scale(with (training, Elevation- 0.2*Horizontal_Distance_To_Hydrology))
newfeature2 <- scale(with (training, Elevation- Vertical_Distance_To_Hydrology))

training1$newfeature1 <- newfeature1
training1$newfeature2 <- newfeature2
filterVarImp(training1[,-55], training1[,55])

#########################################

ggplot() + geom_point(aes(x=Horizontal_Distance_To_Roadways, y= Horizontal_Distance_To_Hydrology, colour=Cover_Type), data= training)



## euclidian distance 
Euclid <- with( training_scaled, sqrt(Horizontal_Distance_To_Hydrology^2 + Vertical_Distance_To_Hydrology^2))
Euclid_scaled <- scale(Euclid)
training1 <- training_scaled
training1$Euclid <- Euclid_scaled
filterVarImp(training1[,-55], training1[,55])



ggplot() + geom_point(aes(x=Elevation, y= Euclid, colour=Cover_Type), data= training)

############################

ggplot() + geom_point(aes(x=Elevation, y= Hillshade_9am, colour=Cover_Type), data= training)
ggplot() + geom_point(aes(x=Elevation, y= Hillshade_Noon, colour=Cover_Type), data= training)
ggplot() + geom_point(aes(x=Elevation, y= Hillshade_3pm, colour=Cover_Type), data= training)

## impute the values with 0 for Hillshade_3pm, has lot of missing values, 
with(training, length(which(Hillshade_3pm==0)))

# 
# [1] "Elevation"                          "Aspect"                            
# [3] "Slope"                              "Horizontal_Distance_To_Hydrology"  
# [5] "Vertical_Distance_To_Hydrology"     "Horizontal_Distance_To_Roadways"   
# [7] "Hillshade_9am"                      "Hillshade_Noon"                    
# [9] "Hillshade_3pm"                      "Horizontal_Distance_To_Fire_Points"
# [11] "Wild"                               "Soil"                              
# [13] "Cover_Type"  
# 




