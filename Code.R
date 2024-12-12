library(tidyverse)
library(corrplot)
library(cvms)
library(moments)
library(huxtable)
library(MASS)
library(caret)
library(adabag)
library(xgboost)

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

set.seed(123)
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

#caclulating skew, kurtosis, and p-val for each feature
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

par(mfrow = c(1,1))
  
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
 
# logistic regression
# plain data set

numericDiagnosis <- 1 * (dataset$Diagnosis == "M")
dataset$numericDiagnosis <- numericDiagnosis
train.set <- dataset[training, -(1:2)]
test.set <- dataset[-training, -(1:2)]
baselogreg <- glm(numericDiagnosis ~ ., data = train.set, family = binomial)
summary(baselogreg)
baselogregpreds <- predict(baselogreg, test.set)

#calculating class prediction 
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

Accuracy
Precision
Sensitivity
F1

# trees
# adaBoost

mfinal <- c(10:40)
maxdepth <- c(3:10)
errormatrix <- matrix(0, length(mfinal), length(maxdepth))

# cross-validation over mfinal and maxdepth parameters

train.set <- dataset[training, -c(1, 33)]
test.set <- dataset[-training, -c(1, 33)]

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
     breaks = seq(from = 0, to = 0.1, by = 0.005),
     col = "purple")


#xgboost
#evaluating metrics at different maxdepths because that is probably the most consequential parameter

xgaccuracies <- data.frame(matrix(ncol = 4, nrow = 8))
colnames(xgaccuracies) <- c('Accuracy', 'Precision', 'Sensitivity', 'F1')
rownames(xgaccuracies) <- c('Max Depth 1:', 'Max Depth 3:', 'Max Depth 5:', 'Max Depth 10:', 'Max Depth 15:', 'Max Depth 20:', 'Max Depth 25:', 'Max Depth 30:')

maxdepths <- c(1, 3, 5, 10, 15, 20, 25, 30)

numericDiagnosis <- 1 * (dataset$Diagnosis == "M")
dataset$numericDiagnosis <- numericDiagnosis
train.set <- dataset[training, -(1:2)]
test.set <- dataset[-training, -(1:2)]
train.set2 <- as.matrix(train.set)
test.set2 <- as.matrix(test.set)

#nthreads determines number of cpu threads used to do computations, so depending on your hardware, you may want to reduce that
#nrounds is set at 750 just to be safe although it is very likely the log loss will converge many rounds before that

for(i in 1:length(maxdepths)){
  bst <- xgboost(data = train.set2[, -31], max.depth = maxdepths[i], objective = "binary:logistic", nthread = 8, nrounds = 750, label = train.set2[, 31])
  prediction <- as.numeric(predict(bst, test.set2[, -31]) > 0.5)
  cm <- confusionMatrix(table(prediction, test.set2[, 31]), mode = "everything")

  xgaccuracies[i, 1] <- cm$overall["Accuracy"]
  xgaccuracies[i, 2] <- cm$byClass["Precision"]
  xgaccuracies[i, 3] <- cm$byClass["Sensitivity"]
  xgaccuracies[i, 4] <- cm$byClass["F1"]
}

xgaccuracies
