


##########################################################
# Libraries
##########################################################

library(plyr)
library(tidyverse)
library(caret)
library(data.table)
library(stringr)
library(knitr)
library(purrr)
library(skimr)
library(randomForest)
library(corrplot)
library(rpart)
library(brnn)
library(monomvn)


##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


##########################################################
# Data Exploration
##########################################################

#understanding the fields available to determine predictors

str(edx)
str(validation)
skimmed <- skim(edx)
skimmed[, c(1:4, 8:17)]

#ratings distribution
edx %>% 
  group_by(rating) %>% 
  summarise (count =n() )%>% 
  rename("Rating"=rating, "Frequency"=count) %>% 
  mutate("%"=paste(round(100*Frequency/nrow(edx),2),"%",sep="")) %>% 
  knitr::kable( align = "lcr")

#more rated movies

edx %>% 
  group_by(title) %>% 
  summarise (count =n() ) %>% 
  ungroup() %>% top_n(.,10) %>% 
  rename("# of ratings"=count) %>% 
  knitr::kable()

#evidence that there are movies with low number of ratings

edx %>% 
  group_by(title) %>% 
  summarise (count =n() ) %>% 
  ungroup() %>% top_n(.,-10) %>% 
  tail( .,10)  %>% 
    rename("# of ratings"=count) %>% 
  knitr::kable()

#top active users
edx %>% 
  group_by(userId) %>% 
  summarise (count =n() ) %>% 
  ungroup() %>% top_n(.,10) %>% 
  rename("# of ratings"=count) %>% 
  knitr::kable(align = "lr")

#less active users

edx %>% 
  group_by(userId) %>% 
  summarise (count =n() ) %>% 
  ungroup() %>% top_n(.,-10) %>% 
  tail( .,10)  %>% 
  rename("# of ratings"=count) %>% 
  knitr::kable(align = "lr")


# release year to be use as possible predictor

edx <- edx %>% mutate(release_year = as.numeric(str_extract(str_extract(title, "[/(]\\d{4}[/)]$"), regex("\\d{4}")))) 

edx %>% ggplot() + geom_histogram(aes(release_year))

#ratings values
unique(edx$rating)



##########################################################
# Minimum benchmarks 
##########################################################

#defining a function for RMSE

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# guessing the outcome - benchmark1
y_hat_guess <- sample(c(seq(0.5,5, by=0.5)), length(validation$rating), replace = TRUE)
acc_guessing <-mean(y_hat_guess == validation$rating)
guess_rmse <- RMSE(validation$rating, y_hat_guess)

#naive model:  average rate for all - benchmark 2

mu_hat <- mean(edx$rating)
naive_rmse <- RMSE(validation$rating, mu_hat)
acc_naive <- 1 - ((naive_rmse*(1-acc_guessing))/guess_rmse)

#creating table that will store all results

rmse_results <- data_frame(method = "Guessing_benchmark1", RMSE = guess_rmse, approx_accuracy= acc_guessing)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Simple_Average_benchmark2",
                                     RMSE = naive_rmse, approx_accuracy= acc_naive ))

rmse_results %>% knitr::kable()


##########################################################
# Preprocessing
##########################################################

#Preprocessing: encoding character variables

#defining the encoding function
helmert <- function(n) {
  m <- t((diag(seq(n-1, 0)) - upper.tri(matrix(1, n, n)))[-n,])
  t(apply(m, 1, rev))
}
encode_helmert <- function(df, var) {
  x <- df[[var]]
  x <- unique(x)
  n <- length(x)
  d <- as.data.frame(helmert(n))
  d[[var]] <- rev(x)
  names(d) <- c(paste0(var, 1:(n-1)), var)
  d
}

#genres enconding
d <- encode_helmert(edx, "genres")

d <- d %>% mutate(genres_encoded = rowSums(d[,-797]))

#verifying that all items have unique encoding for genres column
length(unique(d$genres_encoded))

#merging
edx <- inner_join(edx,d[,c("genres","genres_encoded")], by= "genres")

#title encoding
d <- encode_helmert(edx, "title")

d <- d %>% mutate(title_encoded = rowSums(d[,-10676]))

#verifying that all items have unique encoding for title column
length(unique(d$title_encoded))

#merging
edx <- inner_join(edx,d[,c("title","title_encoded")], by= "title")

head(edx)

#removing d object as it's no longer needed
rm(d)


#Preprocessing: convert all the numeric variables to range between 0 and 1

preProcess_range_model <- preProcess(edx[,-c("rating","genres","title")], method='range')
new_edx <- predict(preProcess_range_model, newdata = edx[,-c("rating","genres","title")])

#verifying that limits of potential predictors are between 0 and 1
apply(new_edx, 2, FUN=function(x){c('min'=min(x), 'max'=max(x))})

#exploring the new and old edx datasets
head(new_edx)
head(edx)

#ensuring that there is no alteration of unique items
length(unique(new_edx$userId))
length(unique(edx$userId))

#appending standarised fields
edx$userId_stand <- new_edx$userId
edx$movieId_stand <- new_edx$movieId
edx$timestamp_stand <- new_edx$timestamp
edx$genres_encoded_stand <- new_edx$genres_encoded
edx$title_encoded_stand <- new_edx$title_encoded
edx$release_year_stand <- new_edx$release_year

#removing new_edx and redundant coluns in edx to preserve memory
rm(new_edx)
edx <- edx[,-c("genres_encoded", "title_encoded" )]



#### checking for correlation among the potential predictors #####

cor_edx <- edx[,8:13] %>% cor() 

corrplot(cor_edx, method="number",type= "full",insig = "blank", number.cex = 0.6)

#### Identifying significant variables #####

colnames(edx[,c(8:13,3)])

sample_n(edx[,c(8:13,3)],100000) %>% 
  pivot_longer( userId_stand:release_year_stand ) %>% 
  group_by(as.factor(rating)) %>%  
  filter(n()>=100) %>% ggplot() +
  geom_boxplot(aes(as.factor(rating),value)) +
  facet_wrap(vars(name))


sample_n(edx[,c(8:13,3)],100000) %>% 
  pivot_longer( userId_stand:release_year_stand  ) %>% 
  group_by(as.factor(rating)) %>%  
  filter(n()>=100) %>% ggplot() +
  geom_col(aes(as.factor(rating),value)) +
  facet_wrap(vars(name))


###quantifying variance influence and ranking variables###

#user effect
length(unique(edx$userId))

user_effect <-edx %>% group_by(userId) %>% summarise( mean_uId = mean(rating), u_effect = mean(mu_hat - mean_uId )) 

edx <- left_join(edx, user_effect[,c(1,3)])

ranking_vars <- data.frame( var = "user_effect_mean" , value= mean(user_effect$u_effect))

rm(user_effect)

#movie effect (using movie ID  discarding title as its correlated)
length(unique(edx$movieId))

movie_effect <-edx %>% group_by(movieId) %>% summarize( mean_mId = mean(rating),  m_effect = mean(mu_hat - mean_mId - u_effect))  

edx <- left_join(edx, movie_effect[,c(1,3)])

ranking_vars <- bind_rows(ranking_vars, data_frame(var = "movie_effect_mean" , value= mean(movie_effect$m_effect)))

rm(movie_effect)

#released year effect
length(unique(edx$release_year))

ry_effect <-edx %>% group_by(release_year) %>% summarize( mean_ryId = mean(rating),  ry_effect = mean(mu_hat- mean_ryId - u_effect  - m_effect ))  

edx <- left_join(edx, ry_effect[,c(1,3)])

ranking_vars <- bind_rows(ranking_vars, data_frame(var = "ry_effect_mean" , value= mean(ry_effect$ry_effect)))

rm(ry_effect)

#genres year effect
length(unique(edx$genres))

genres_effect <-edx %>% group_by(genres) %>% summarize( mean_grId = mean(rating),  gr_effect = mean(mu_hat- mean_grId - u_effect  - m_effect - ry_effect))  

edx <- left_join(edx, genres_effect[,c(1,3)])

ranking_vars <- bind_rows(ranking_vars, data_frame(var = "genres_effect_mean" , value= mean(genres_effect$gr_effect)))

rm(genres_effect)

#timestamp effect not considered due to high carnality and no rational connection to ranking decision
length(unique(edx$timestamp))


#visualising the relevance of the predictors

ranking_vars %>% ggplot(aes(x= "" , y=value, fill=var)) +  geom_bar (stat= "identity" , width=.5)




######Model fitting#######

###linear model###

# optimising the regularisation parameter

lambdas <- seq(0, 7, 0.5)

reg_rmse <- sapply( lambdas, function(l){
  mu_hat <- mean(edx$rating)
  m <- edx %>%
    group_by(movieId) %>%
    summarize(reg_m_effect = sum(rating - mu_hat)/(n()+l))
  u <- edx %>% 
    left_join(m, by="movieId") %>%
    group_by(userId) %>%
    summarize(reg_u_effect = sum(rating - reg_m_effect - mu_hat)/(n()+l))
  predicted_ratings <- 
    edx %>% 
    left_join(m, by = "movieId") %>%
    left_join(u, by = "userId") %>%
    mutate(mu = mean(rating), y_hat = mu + reg_m_effect + reg_u_effect) %>%
    .$y_hat
  return(RMSE(predicted_ratings, edx$rating))
})


qplot(lambdas, reg_rmse)  

lambda <- lambdas[which.min(reg_rmse)]
lambda

# predicting with regularised lineal model

reg_rmse_val <- lapply( lambda, function(l){
  mu_hat <- mean(validation$rating)
  m <- validation %>%
    group_by(movieId) %>%
    summarize(reg_m_effect = sum(rating - mu_hat)/(n()+l))
  u <- validation %>% 
    left_join(m, by="movieId") %>%
    group_by(userId) %>%
    summarize(reg_u_effect = sum(rating - reg_m_effect - mu_hat)/(n()+l))
  predicted_ratings <- 
    validation %>% 
    left_join(m, by = "movieId") %>%
    left_join(u, by = "userId") %>%
    mutate(mu = mean(rating), y_hat = mu + reg_m_effect + reg_u_effect) %>%
    .$y_hat
  return(RMSE(predicted_ratings, validation$rating))
})


reglm_naive <- 1 - ((as.numeric(reg_rmse_val)*(1-acc_guessing))/guess_rmse)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularised_Lineal_Model",
                                     RMSE = as.numeric(reg_rmse_val), approx_accuracy= reglm_naive ))

####Evaluating other models #####

##preparing the validation set: convert relevant numeric variables to range between 0 and 1 ####

preProcess_range_model_val <- preProcess(validation[,-c("rating","genres","title")], method='range')
new_val <- predict(preProcess_range_model_val, newdata = validation[,-c("rating","genres","title")])

#verifying that limits of potential predictors are between 0 and 1
apply(new_val, 2, FUN=function(x){c('min'=min(x), 'max'=max(x))})

#appending standarised fields
validation$userId_stand <- new_val$userId
validation$movieId_stand <- new_val$movieId

#exploding amended validation
head(validation)

rm(new_val)


# setting the control parameters
fitControl <- trainControl(
  method = 'cv',                   
  number = 2,                      
  savePredictions = 'final',      
    ) 


# measuring run time for various models


#Bayesian Regularized Neural Networks (method = 'brnn')

brnn_grid <-  expand.grid(neurons = c(2, 3, 4))

set.seed(100)

startTime <- Sys.time()
model_brnn = train(rating ~ userId_stand + movieId_stand , data=sample_n(edx,1000), method='brnn', metric='RMSE', tuneGrid = brnn_grid, trControl = fitControl)
endTime <- Sys.time()

print(endTime - startTime)

startTime <- Sys.time()
predict(model_brnn, sample_n(validation,1000))
endTime <- Sys.time()
print(endTime - startTime)


#CART (method = 'rpart2')

cart_grid <-  expand.grid(maxdepth = c(2, 3, 4, 5))

set.seed(100)

startTime <- Sys.time()
model_cart = train(rating ~ userId_stand + movieId_stand , data=sample_n(edx,1000), method='rpart2', metric='RMSE', tuneGrid = cart_grid, trControl = fitControl)
endTime <- Sys.time()

print(endTime - startTime)

startTime <- Sys.time()
predict(model_cart, sample_n(validation,1000))
endTime <- Sys.time()
print(endTime - startTime)


# Bayesian Ridge Regression (Model Averaged)   method = 'blassoAveraged'

set.seed(100)

startTime <- Sys.time()
model_brr = train(rating ~ userId_stand + movieId_stand , data=sample_n(edx,1000), method='blassoAveraged', metric='RMSE', trControl = fitControl)
endTime <- Sys.time()

print(endTime - startTime)

startTime <- Sys.time()
predict(model_brr, sample_n(validation,1000))
endTime <- Sys.time()
print(endTime - startTime)


#comparing sample models

sample_models_compare <- resamples(list(brnn = model_brnn, 
                                        cart = model_cart, 
                                        brr = model_brr))

# Summary of the sample models performances
summary(sample_models_compare) 


# Draw box plots to compare models
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(sample_models_compare, scales=scales)



####saving data environment for use in Rmd####

save.image (file = "Capstone-MovieLens_Report_MM.RData")






