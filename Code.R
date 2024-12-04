library(tidyverse)
library(corrplot)
library(mltools)
library(cvms)
library(moments)
library(huxtable)
library(MASS)
library(caret)
library(bestNormalize)
library(adabag)
library(xgboost)
library(DiagrammeR)

dataset <- read.csv("wdbc.data")
dataset %>% head()
dataset$M <- as.factor(dataset$M)
dataset %>% colnames()

colnames(dataset) <- c("ID",
                       "Diagnosis",
                       "MeanRadius",
                       "MeanTexture",
                       "MeanPerimeter",
                       "MeanArea",
                       "MeanSmoothness",
                       "MeanCompactness",
                       "MeanConcavity",
                       "MeanConcavePoints",
                       "MeanSymmetry",
                       "MeanFractalDimension",
                       "SERadius",
                       "SETexture",
                       "SEPerimeter",
                       "SEArea",
                       "SESmoothness",
                       "SECompactness",
                       "SEConcavity",
                       "SEConcavePoints",
                       "SESymmetry",
                       "SEFractalDimension",
                       "WorstRadius",
                       "WorstTexture",
                       "WorstPerimeter",
                       "WorstArea",
                       "WorstSmoothness",
                       "WorstCompactness",
                       "WorstConcavity",
                       "WorstConcavePoints",
                       "WorstSymmetry",
                       "WorstFractalDimension")

# creating training and testing data sets, 

numericDiagnosis <- 1 * (dataset$Diagnosis == "M")
dataset$numericDiagnosis <- numericDiagnosis
set.seed(16)
training <- sample(1:nrow(dataset), 0.7*nrow(dataset))
train.set <- dataset[training, -(1)]
test.set <- dataset[-training, -(1)]

#plotting feature distributions

mycols = c(594:598,
           574:578,
           372:376,
           519:523,
           589:593,
           641:645)

par(mfrow = c(6,5))
for(i in (3:32)){
    hist(dataset[, i],
      main = colnames(dataset)[i],
      xlab = "Value",
      col = colors()[mycols[i - 2]])
}

#normality testing

normalitydf <- data.frame(matrix(ncol = 3, nrow = 30))
colnames(normalitydf) <- c('Skewness', 'Kurtosis', 'Jarque-Bera Test P-Value')
rownames(normalitydf) <- colnames(dataset)[3:32]

for(i in (3:32)){
  skew <- skewness(dataset[, i])
  kurt <- kurtosis(dataset[, i])
  jarquepval <- jarque.test(dataset[, i])$p.value
  
  normalitydf[i - 2, 1] <- skew
  normalitydf[i - 2, 2] <- kurt
  normalitydf[i - 2, 3] <- jarquepval
}

ht <- as_huxtable(normalitydf, add_rownames = "Feature")
ht %>% 
  set_bold(1, everywhere, TRUE) %>% 
  set_all_borders(1)
  map_background_color(everywhere, 'Jarque-Bera Test P-Value', by_ranges(c(0, 0.05), "red")) %>% 
  print_screen()

# correlation hypothesis testing and visualization
# pearson doesn't make sense bc features aren't normally dist

corrs <- cor(dataset[,-(1:2)],
             method = "kendall")
pvalmatrix <- cor.mtest(dataset[, -(1:2)], 
                        conf.level = 0.95, 
                        method = "kendall",
                        alternative = "two.sided")
corrplot(corrs, 
         p.mat = pvalmatrix$p,
         sig.level = 0.05,
         type = "upper",
         diag = FALSE,
         col = COL2('PiYG'),
         method = 'color')
 
# model eval matrix

modeleval <- data.frame(matrix(ncol = 4, nrow = 2))
colnames(modeleval) <- c('Accuracy', 'Precision', 'Sensitivity', 'F1')
rownames(modeleval) <- c('Base', 'w/ Box-Cox Transform', 'w/ Yeo-Johnson Transform')

# logistic regression
# plain data set

baselogreg <- glm(numericDiagnosis ~ ., data = train.set, family = binomial)
summary(baselogreg)
baselogregpreds <- predict(baselogreg, test.set)
predictions <- 1 * (baselogregpreds >= 0)
truediagnosis <- test.set[,31] 
cfm <- as_tibble(table(predictions, truediagnosis))
plot_confusion_matrix(cfm,
                      target_col = "truediagnosis",
                      prediction_col = "predictions",
                      counts_col = "n",
                      palette = "Purples")
cm <- confusionMatrix(table(predictions, truediagnosis), mode = "everything")
cm
Accuracy <- cm$overall["Accuracy"]
Precision <- cm$byClass["Precision"]
Sensitivity <- cm$byClass["Sensitivity"]
F1 <- cm$byClass["F1"]

modeleval[1, 1] <- Accuracy
modeleval[1, 2] <- Precision
modeleval[1, 3] <- Sensitivity
modeleval[1, 4] <- F1

# w/ box-cox transforms

boxcoxdata <- dataset
for(i in 3:32){
  if(all(boxcoxdata[, i] > 0)){
    b <- boxcox(lm(boxcoxdata[, i] ~ 1))
    lambda <- b$x[which.max(b$y)]
    boxcoxdata[, i] <- (boxcoxdata[, i] ^ lambda - 1)/lambda
  }
}

boxcoxlogreg <- glm(numericDiagnosis ~ ., data = boxcoxdata[training, -(1:2)], family = binomial)
summary(boxcoxlogreg)
boxcoxlogregpreds <- predict(boxcoxlogreg, test.set)
predictions <- 1 * (boxcoxlogregpreds >= 0)
truediagnosis <- test.set[,31]
cfm <- as_tibble(table(truediagnosis, predictions))
plot_confusion_matrix(cfm,
                      target_col = "truediagnosis",
                      prediction_col = "predictions",
                      counts_col = "n",
                      palette = "Purples")
cm <- confusionMatrix(table(predictions, truediagnosis), mode = "everything")
Accuracy <- cm$overall["Accuracy"]
Precision <- cm$byClass["Precision"]
Sensitivity <- cm$byClass["Sensitivity"]
F1 <- cm$byClass["F1"]

modeleval[2, 1] <- Accuracy
modeleval[2, 2] <- Precision
modeleval[2, 3] <- Sensitivity
modeleval[2, 4] <- F1

# w/ yeo-johnson transform

yeojohnsondata <- dataset
for(i in 3:32){
  yeojohnsondata[, i] <- yeojohnson(yeojohnsondata[, i])
}

yeojohnsonlogreg <- glm(numericDiagnosis ~ ., data = yeojohnsondata[training, -(1:2)], family = binomial)
summary(yeojohnsonlogreg)
yeojohnsonlogregpreds <- predict(yeojohnsonlogreg, test.set)
predictions <- 1 * (yeojohnsonlogregpreds >= 0)
truediagnosis <- test.set[,31]
cfm <- as_tibble(table(truediagnosis, predictions))
plot_confusion_matrix(cfm,
                      target_col = "truediagnosis",
                      prediction_col = "predictions",
                      counts_col = "n",
                      palette = "Purples")
cm <- confusionMatrix(table(predictions, truediagnosis), mode = "everything")
Accuracy <- cm$overall["Accuracy"]
Precision <- cm$byClass["Precision"]
Sensitivity <- cm$byClass["Sensitivity"]
F1 <- cm$byClass["F1"]

modeleval[3, 1] <- Accuracy
modeleval[3, 2] <- Precision
modeleval[3, 3] <- Sensitivity
modeleval[3, 4] <- F1

# w/ PCA

standardize <- function(x){
  stdx <- (x - mean(x))/sqrt(var(x))
  return(stdx)
}

pcadata <- apply(dataset[, -(1:2)], 2, standardize)
pcadata[, 31] <- dataset[, 33]
pcscores <- prcomp(pcadata[, -31], center = TRUE, scale = TRUE)
pcscores
summary(pcscores)
 
# not done yet

# trees
# adaBoost

mfinal <- c(10:40)
maxdepth <- c(3:10)
errormatrix <- matrix(0, length(mfinal), length(maxdepth))

for(i in 1:length(mfinal)){
  for(j in 1:length(maxdepth)){
    adaboostm1 <- boosting(Diagnosis ~ ., data = train.set, mfinal = mfinal[i], coeflearn = "Zhu",
                           control = rpart.control(maxdepth = maxdepth[j]))
    adaboostm1pred <- predict.boosting(adaboostm1, newdata = test.set)
    errormatrix[i, j] <- adaboostm1pred$error
  }
}

m <- errormatrix
min = which(m == min(m), arr.ind = TRUE)
1 - m[min]
min

hist(errormatrix,
     xlab = "Error Rate",
     ylab = "Number of Models",
     main = "AdaBoost CV Errors",
     breaks = seq(from = 0, to = 0.05, by = 0.005),
     col = "purple")


#xgboost

train.set2 <- as.matrix(train.set[, -1])
test.set2 <- as.matrix(test.set[, -1])
bst <- xgboost(data = train.set2[, -31], maxdepth = 20, objective = "binary:logistic", nthread = 8, nrounds = 750, label = train.set2[, 31])
prediction <- as.numeric(predict(bst, test.set2[, -31]) > 0.5)
prediction

cm <- confusionMatrix(table(prediction, test.set2[, 31]), mode = "everything")
cm


xgb.plot.tree(model = bst)
