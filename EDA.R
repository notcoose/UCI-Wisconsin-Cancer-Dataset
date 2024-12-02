library(tidyverse)
library(corrplot)

dataset <- read.csv("wdbc.data")
dataset %>% head()
dataset$M <- as.factor(dataset$M)
dataset %>% colnames()
colnames(dataset) <- c("ID",
                       "Diagnosis",
                       "Mean Radius",
                       "Mean Texture",
                       "Mean Perimeter",
                       "Mean Area",
                       "Mean Smoothness",
                       "Mean Compactness",
                       "Mean Concavity", 
                       "Mean Concave Points",
                       "Mean Symmetry",
                       "Mean Fractal Dimension",
                       "SE Radius",
                       "SE Texture",
                       "SE Perimeter",
                       "SE Area",
                       "SE Smoothness",
                       "SE Compactness",
                       "SE Concavity", 
                       "SE Concave Points",
                       "SE Symmetry",
                       "SE Fractal Dimension",
                       "Worst Radius",
                       "Worst Texture",
                       "Worst Perimeter",
                       "Worst Area",
                       "Worst Smoothness",
                       "Worst Compactness",
                       "Worst Concavity", 
                       "Worst Concave Points",
                       "Worst Symmetry",
                       "Worst Fractal Dimension")
corrs <- cor(dataset[,-(1:2)])
corrplot(corrs, method = 'color')

