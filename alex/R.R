library(dplyr)

data <- read.csv('C:/Users/Alexander/Documents/DS502_Final/data/train.csv')

names(data)

plot(data$Dir, data$Yards)

glimpse(data)
head(data)
