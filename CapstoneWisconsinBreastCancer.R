## Installing all necessary libraries if not already preloaded
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(ggfortify)) install.packages("ggfortify", repos = "http://cran.us.r-project.org")
if(!require(glmnet)) install.packages("glmnet", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(nnet)) install.packages("nnet", repos = "http://cran.us.r-project.org")
if(!require(funModeling)) install.packages("funModeling", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
library(funModeling)
library(corrplot)

# Loading the Wisconsin Breast Cancer Diagnostic dataset from my github repository
breastcancer <- read.csv("https://raw.githubusercontent.com/jaygopalak/wbcd/master/wbcddata.csv")
# Link to the original dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data
# Link to .csv dataset on kaggle: https://www.kaggle.com/uciml/breast-cancer-wisconsin-data

# Examining the dataset
head(breastcancer)

# Removing column 33 (Invalid for our analysis)
breastcancer <- breastcancer[-c(33)]

## Data Summary Statistics ##
# 569 observations of 32 variables
dim(breastcancer)
head(breastcancer)
# summary statistics for dataset
summary(breastcancer)

# Checking the dataset for any missing values
sum(is.na(breastcancer))

### Data Exploration ###

# Diagnosis is a factor with two levels: "M" (malignant) and "B" (Benign).
# Proportions of the diagnosis (Benign vs Malignant)
prop.table(table(breastcancer$diagnosis))

# Distribution of the  Diagnosis Column
ggplot(breastcancer, aes(x=diagnosis, fill=diagnosis))+geom_bar(alpha=0.5)+labs(title="Distribution of Diagnosis")

# Plotting all numerical variables except for id
plot_num(breastcancer %>% select(-id), bins=10) 

# Checking for correlation of variables excluding id and diagnosis
correlationMatrix <- cor(breastcancer[,3:ncol(breastcancer)])
correlationMatrix

# Since there are a lot of variables, a correlation plot is used to visualize the above matrix
# Correlation matrix is reorderd according to hierarchial clustering and rectangles are added for better visualization
corrplot(correlationMatrix, order = "hclust", tl.cex = 1, addrect = 8)

## Data Cleaning ## 

# Finding highly correlated variables
highCorrelation <- findCorrelation(correlationMatrix, cutoff=0.9)
# Removing correlated variables
breastcancer2 <- breastcancer %>%select(-highCorrelation)
# Number of columns after removing correlated variables (only 22)
ncol(breastcancer2)

## Data Pre-Processing ##

# Principle Component Analysis (PCA)
## Removing id & diagnosis , then scaling and centering the variables
pca_bc <- prcomp(breastcancer %>% select(-id, -diagnosis), scale = TRUE, center = TRUE)
plot(pca_bc, type="l")
summary(pca_bc)

# PCA using transformed dataset
pca_bc2<- prcomp(breastcancer2[,3:ncol(breastcancer2)], center = TRUE, scale = TRUE)
plot(pca_bc2, type="l")
summary(pca_bc2)

# PC's in the transformed dataset2
pca_df <- as.data.frame(pca_bc2$x)
# Plot of pc1 and pc2
g_pc1 <- ggplot(pca_df, aes(x=PC1, fill=breastcancer$diagnosis)) + geom_density(alpha=0.25)  
g_pc2 <- ggplot(pca_df, aes(x=PC2, fill=breastcancer$diagnosis)) + geom_density(alpha=0.25)  
g_pc1
g_pc2


# Linear Discriminant Analysis (LDA)

# Data with LDA
lda_bc <- MASS::lda(diagnosis~., data = breastcancer, center = TRUE, scale = TRUE) 
lda_bc

#Data frame of the LDA for visualization purposes
lda_df_predict <- predict(lda_bc, breastcancer)$x %>% as.data.frame() %>% cbind(diagnosis=breastcancer$diagnosis)
ggplot(lda_df_predict, aes(x=LD1, fill=diagnosis)) + geom_density(alpha=0.5)


## Data Modeling ##


# Creation of the partition 80% and 20%
set.seed(1815) #provare 1234
breastcancer3 <- cbind (diagnosis=breastcancer$diagnosis, breastcancer2)
data_sampling_index <- createDataPartition(breastcancer$diagnosis, times=1, p=0.8, list = FALSE)
train_data <- breastcancer3[data_sampling_index, ]
test_data <- breastcancer3[-data_sampling_index, ]

# trainControl function used to specify cross validation and the number of folds (15)
fitControl <- trainControl(method="cv",    
                           number = 15,    
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)


### Naive Bayes Model

# Creation of Naive Bayes Model
model_nb <- train(diagnosis~.,
                      train_data,
                      method="nb",
                      metric="ROC",
                      preProcess=c('center', 'scale'), 
                      trace=FALSE,
                      trControl=fitControl)

# Prediction
prediction_nb <- predict(model_nb, test_data)
# Confusion matrix
confusionmatrix_nb <- confusionMatrix(prediction_nb, test_data$diagnosis, positive = "M")
confusionmatrix_nb

#Plot of important predictors
plot(varImp(model_nb), top=10, main="NaiveBayes")


### Logistic Regression Model 

# Creation of Logistic Regression Model
model_logregression<- train(diagnosis ~., data = train_data, method = "glm",
                     metric = "ROC",
                     preProcess = c("scale", "center"), 
                     trControl= fitControl)
# Prediction
prediction_logregression<- predict(model_logregression, test_data)

# Confusion matrix
confusionmatrix_logregression <- confusionMatrix(prediction_logregression, test_data$diagnosis, positive = "M")
confusionmatrix_logregression

# Plot of top important variables
plot(varImp(model_logregression), top=10, main="Logistic Regression")


### Random Forest Model

# Creation of Random Forest Model
model_randomforest <- train(diagnosis~.,
                            train_data,
                            method="rf",  
                            metric="ROC",
                            preProcess = c('center', 'scale'),
                            trControl=fitControl)
# Prediction
prediction_randomforest <- predict(model_randomforest, test_data)

# Confusion matrix
confusionmatrix_randomforest <- confusionMatrix(prediction_randomforest, test_data$diagnosis, positive = "M")
confusionmatrix_randomforest

# Plot of top important variables
plot(varImp(model_randomforest), top=10, main="Random Forest")


### K Nearest Neighbor (KNN) Model

# Creation of K Nearest Neighbor (KNN) Model
model_knn <- train(diagnosis~.,
                   train_data,
                   method="knn",
                   metric="ROC",
                   preProcess = c('center', 'scale'),
                   tuneLength=10, 
                   trControl=fitControl)
# Prediction
prediction_knn <- predict(model_knn, test_data)

# Confusion matrix        
confusionmatrix_knn <- confusionMatrix(prediction_knn, test_data$diagnosis, positive = "M")
confusionmatrix_knn

# Plot of top important variables
plot(varImp(model_knn), top=10, main="KNN")

### Neural Network with PCA Model

# Creation of Random Forest Model
# Specifying tune length to try 10 default values for the main parameter
model_neuralnet_pca <- train(diagnosis~.,
                        train_data,
                        method="nnet",
                        metric="ROC",
                        preProcess=c('center', 'scale', 'pca'),
                        tuneLength=10,
                        trace=FALSE,
                        trControl=fitControl)
# Prediction
prediction_neuralnet_pca <- predict(model_neuralnet_pca, test_data)

# Confusion matrix
confusionmatrix_neuralnet_pca <- confusionMatrix(prediction_neuralnet_pca, test_data$diagnosis, positive = "M")
confusionmatrix_neuralnet_pca

# Plot of top important variables
plot(varImp(model_neuralnet_pca), top=8, main="Neural Network PCA")

### Neural Network with LDA Model

# Creation of training set and test set with LDA modified data
train_data_lda <- lda_df_predict[data_sampling_index, ]
test_data_lda <- lda_df_predict[-data_sampling_index, ]

# Creation of Neural Network with LDA Model
model_neuralnet_lda <- train(diagnosis~.,
                        train_data_lda,
                        method="nnet",
                        metric="ROC",
                        preProcess=c('center', 'scale'),
                        tuneLength=10,
                        trace=FALSE,
                        trControl=fitControl)
# Prediction
prediction_neuralnet_lda <- predict(model_neuralnet_lda, test_data_lda)
# Confusion matrix
confusionmatrix_neuralnet_lda <- confusionMatrix(prediction_neuralnet_lda, test_data_lda$diagnosis, positive = "M")
confusionmatrix_neuralnet_lda


## Results ##

# Creation of the list of all models
models_list <- list(Naive_Bayes=model_nb, 
                    Logistic_regr=model_logregression,
                    Random_Forest=model_randomforest,
                    KNN=model_knn,
                    Neural_PCA=model_neuralnet_pca,
                    Neural_LDA=model_neuralnet_lda)                                    
models_results <- resamples(models_list)

# Print the summary of models
summary(models_results)

# Plot of the models results
bwplot(models_results, metric="ROC")

# Confusion matrix of the models
confusionmatrix_list <- list(
  Naive_Bayes=confusionmatrix_nb, 
  Logistic_regr=confusionmatrix_logregression,
  Random_Forest=confusionmatrix_randomforest,
  KNN=confusionmatrix_knn,
  Neural_PCA=confusionmatrix_neuralnet_pca,
  Neural_LDA=confusionmatrix_neuralnet_lda)   
confusionmatrix_list_results <- sapply(confusionmatrix_list, function(x) x$byClass)
confusionmatrix_list_results %>% knitr::kable()

# Find the best result for each metric
confusionmatrix_results_max <- apply(confusionmatrix_list_results, 1, which.is.max)
report <- data.frame(metric=names(confusionmatrix_results_max), 
                            best_model=colnames(confusionmatrix_list_results)
                            [confusionmatrix_results_max],
                            value=mapply(function(x,y) 
                            {confusionmatrix_list_results[x,y]}, 
                            names(confusionmatrix_results_max), 
                            confusionmatrix_results_max))
rownames(report) <- NULL
report



