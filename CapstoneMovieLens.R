# MovieLens Project: Movie recommendation system
# HarvardX: PH125.9x - Data Science: Capstone
# Author: Priscila Trevino Aguilar
# Date: December 2020



###### Required packages installation ######
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(dyplr)) install.packages("dyplr", repos = "http://cran.us.r-project.org")
if(!require(ggthemes)) install.packages("ggthemes", repos = "http://cran.us.r-project.org")
if(!require(ggsci)) install.packages("ggsci", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(knitr))install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(scales))install.packages("scales", repos = "http://cran.us.r-project.org")
if(!require(randomForest))install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(params))install.packages("params", repos = "http://cran.us.r-project.org")



##### Required libraries loading #####
library(tidyverse)
library(caret)
library(data.table)
library(dyplr)
library(ggthemes)
library(ggsci)
library(lubridate)
library(knitr)
library(scales)
library(randomForest)
library(params)



##### Data loading, tidying and creation of edx and validation sets #####

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]  #edx set creation
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>%  # validation set creation
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Removal of objects from memory
rm(dl, ratings, movies, test_index, temp, movielens, removed)



# Methods section #

##### Exploratory data analysis #####

# General data exploration
str(edx, width=80, strict.width="cut")
summary(edx)
sapply(edx, class)

# Number of distinct movies and users 
n_distinct(edx$movieId)
n_distinct(edx$userId)

# No NA values verification
any(is.na(edx))


## Data visualization

# Ratings per movie distribution plot (the movie effect)
edx %>%
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins=15, color="#6699CC", fill="#006669") +
  scale_x_log10() +
  xlab("Ratings") +
  ylab("Movies") + 
  ggtitle("Ratings per movie") + 
  theme_economist() + 
  theme(plot.title = element_text(size=14, face="bold", hjust=0.5),
        axis.title.x = element_text(size=12, face="bold"),
        axis.title.y = element_text(size=12, face="bold"),
        axis.text.x = element_text(face="bold", size=10),
        axis.text.y = element_text(face="bold", size=10))


# Ratings per user distribution plot (the user effect)
edx %>%
  count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins=15, color="#6699CC", fill="#006669") +
  scale_x_log10() +
  xlab("Ratings") +
  ylab("Users") + 
  ggtitle("Ratings per users") + 
  theme_economist() + 
  theme(plot.title = element_text(size=14, face="bold", hjust=0.5),
        axis.title.x = element_text(size=12, face="bold"),
        axis.title.y = element_text(size=12, face="bold"),
        axis.text.x = element_text(face="bold", size=10),
        axis.text.y = element_text(face="bold", size=10))


# Ratings per top 10 genres plot (the genre effect)
genre_plot <- edx %>% 
  separate_rows(genres, sep = "\\|") %>%  # genres separation 
  group_by(genres) %>%
  summarize(count = n()) %>%
  arrange(desc(count)) %>% 
  top_n(10)

genre_plot %>%  # the top 10 most rated movie genres were plotted 
  ggplot(aes(x=reorder(genres, count), y=count)) + 
  geom_bar(aes(fill=genres), stat = "identity") +
  coord_flip() +  
  theme_economist() + 
  scale_fill_aaas() + 
  ggtitle("Rating per genre distribution") +
  ylab("Number of ratings") + 
  theme(legend.position = "none",  
        plot.title = element_text(size=14, face="bold", hjust=0.5),
        axis.title.y = element_blank(),
        axis.title.x = element_text(size=12, face="bold"),
        axis.text.x = element_text(face="bold", size=10),
        axis.text.y = element_text(face="bold", size=10))

# Ratings per movie release year plot (the release year effect)
year_plot <- edx %>% mutate(releaseYear = as.numeric(str_extract(str_extract(title, "[/(]\\d{4}[/)]$"), 
                            regex("\\d{4}"))),
                            title = str_remove(title, "[/(]\\d{4}[/)]$"))  # the year was extracted from...
                                                                           # ... the title column 

year_plot %>% group_by(releaseYear) %>%  # movie rating per release year was plotted
  summarize(rating = mean(rating)) %>%
  ggplot(aes(releaseYear, rating)) +
  geom_point(color="#006669") +
  geom_smooth(linetype="dashed", color="blue", fill="#99CCCC") + 
  theme_economist() + 
  ggtitle("Rating per movie release year") + 
  xlab("Release year") + 
  ylab("Rating") + 
  theme(plot.title = element_text(size=14, face="bold", hjust=0.5),
        axis.title.x = element_text(size=12, face="bold"),
        axis.title.y = element_text(size=12, face="bold"),
        axis.text.x = element_text(face="bold", size=10),
        axis.text.y = element_text(face="bold", size=10))




##### Data wrangling #####

# Function to extract the movie release year from the 'title' column
year_sep <- function(z){
  
  z <- z %>% mutate(releaseYear = as.numeric(str_extract(str_extract(title, "[/(]\\d{4}[/)]$"), 
                                                         regex("\\d{4}"))), 
                    title = str_remove(title, "[/(]\\d{4}[/)]$"))
}

edx <- year_sep(edx)  # the function was applied to the edx and validation sets
validation <- year_sep(validation)


# Function to separate the movie genres
genre_sep <- function(x){
  
  x <- x %>% separate_rows(genres, sep = "\\|") 
}

edx <- genre_sep(edx)  # the function was applied to the edx and validation sets
validation <- genre_sep(validation)


# The sets were transformed back to data frames to facilitate analysis
edx <- as.data.frame(edx) 
validation <- as.data.frame(validation)

# New data set observation
head(edx)



  
##### Data partition and creation of train and test sets #####

set.seed(1)
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, 
                                    list = FALSE)
test_set <- edx[test_index,]
train_set <- edx[-test_index,]

# Ensuring that the corresponding movieId and userIds are included in all sets
test_set <- test_set %>%  
    semi_join(train_set, by = "movieId") %>%
    semi_join(train_set, by = "userId")
  
validation <- validation %>% 
    semi_join(train_set, by = "movieId") %>%
    semi_join(train_set, by = "userId")  




##### Modeling approach #####

# RMSE function creation for models evaluation
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


# Just the average model (naive)
mu <- mean(train_set$rating) 

result_mu <- RMSE(test_set$rating, mu)
print(result_mu)  # result was computed and saved


# Movie effect model
mu <- mean(train_set$rating)
movie_effect <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_m = mean(rating - mu))  # added movie effect

model_1 <- test_set %>% 
  left_join(movie_effect, by="movieId") %>%
  mutate(pred = mu + b_m) %>% 
  pull(pred)

result_1 <- RMSE(test_set$rating, model_1)
print(result_1)  # result was computed and saved


# Movie and user effect model
user_effect <- train_set %>% 
  left_join(movie_effect, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_m))  # added movie + user effect

model_2 <- test_set %>% 
  left_join(movie_effect, by="movieId") %>%
  left_join(user_effect, by="userId") %>%
  mutate(pred = mu + b_m + b_u) %>%
  pull(pred)

result_2 <- RMSE(test_set$rating, model_2)
print(result_2)  # result was computed and saved


# Movie, user and genre effect model
genre_effect <- train_set %>% 
  left_join(movie_effect, by="movieId") %>%
  left_join(user_effect, by="userId") %>%
  group_by(genres) %>% 
  summarize(b_g = mean(rating - mu - b_m - b_u))  # added movie + user + genre effect

model_3 <- test_set %>% 
  left_join(movie_effect, by="movieId") %>%
  left_join(user_effect, by="userId") %>%
  left_join(genre_effect, by="genres") %>%
  mutate(pred = mu + b_m + b_u + b_g) %>%
  pull(pred)

result_3 <- RMSE(test_set$rating, model_3)
print(result_3)  # result was computed and saved


# Movie, user, genre and year effect model
year_effect <- train_set %>% 
  left_join(movie_effect, by="movieId") %>%
  left_join(user_effect, by="userId") %>%
  left_join(genre_effect, by="genres") %>%
  group_by(releaseYear) %>% 
  summarize(b_y = mean(rating - mu - b_m - b_u - b_g))  # added movie + user + genre + year effect

model_4 <- test_set %>% 
  left_join(movie_effect, by="movieId") %>%
  left_join(user_effect, by="userId") %>%
  left_join(genre_effect, by="genres") %>%
  left_join(year_effect, by="releaseYear") %>%
  mutate(pred = mu + b_m + b_u + b_g + b_y) %>%
  pull(pred)

result_4 <- RMSE(test_set$rating, model_4)
print(result_4)  # result was computed and saved



# Regularization and cross validation to find optimal lambda (tuning parameter) value 

lambdas <- seq(0, 10, 0.25) # tuning parameter range

mu <- mean(train_set$rating)
 
rmses <- sapply(lambdas, function(l){  # function to perform cross validation
  
  b_m <- train_set %>%  # adjust mean by penalizing movie effect
    group_by(movieId) %>%
    summarize(b_m = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>% 
    left_join(b_m, by="movieId") %>%  # adjust mean by penalizing movie and user effect
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_m - mu)/(n()+l))  
  
  b_g <- train_set %>%  # adjust mean by penalizing movie, user and genre effect
    left_join(b_m, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_m - b_u - mu)/(n()+l)) 
  
  b_y <- train_set %>%  # adjust mean by penalizing movie, user, genre and year effect
    left_join(b_m, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(b_g, by="genres") %>%
    group_by(releaseYear) %>%
    summarize(b_y = sum(rating - b_m - b_u - b_g - mu)/(n()+l))
  
  predicted_ratings <- test_set %>%  # computed predictions 
    left_join(b_m, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    left_join(b_y, by = "releaseYear") %>%
    mutate(pred = mu + b_m + b_u + b_g + b_y) %>%
    pull(pred)
  
  return(RMSE(test_set$rating, predicted_ratings))
})

# Lambdas vs RMSEs plot
qplot(lambdas, rmses)

# optimal lambda value
lambda <- lambdas[which.min(rmses)]  
print(lambda)

result_reg <- min(rmses)
print(result_reg) # cross validation result was saved 




# Final hold-out test set (validation) using the optimal lambda value 

lambda <- 3.75 # optimal lambda value obtained in previous cross validation

b_m <- train_set %>% 
  group_by(movieId) %>%
  summarize(b_m = sum(rating - mu)/(n()+lambda))

b_u <- train_set %>% 
  left_join(b_m, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_m - mu)/(n()+lambda))

b_g <- train_set %>% 
  left_join(b_m, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - b_m - b_u - mu)/(n()+lambda))

b_y <- train_set %>% 
  left_join(b_m, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  left_join(b_g, by="genres") %>%
  group_by(releaseYear) %>%
  summarize(b_y = sum(rating - b_m - b_u - b_g - mu)/(n()+lambda))

predicted_ratings <- validation %>%  # only call for validation set, final test
  left_join(b_m, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%   
  left_join(b_g, by = "genres") %>%
  left_join(b_y, by = "releaseYear") %>%
  mutate(pred = mu + b_m + b_u + b_g + b_y) %>%
  pull(pred)


result_5 <- RMSE(validation$rating, predicted_ratings)
print(result_5) # final model result was saved




##### Random forest model attempt #####

mysample <- edx[sample(1:nrow(edx), 10000,  # a random sample sample of length 10 thousand...
                       replace=FALSE),]     #... is taken from the edx set

# Data partition of the sample to create train and test sets 
set.seed(1)
test_index <- createDataPartition(y = mysample$rating, times = 1, p = 0.2, 
                                  list = FALSE)

test_rf <- mysample[test_index,]
train_rf <-mysample[-test_index,]

test_rf <- test_rf %>%                     # ensuring that the corresponding movieId and ...
  semi_join(train_rf, by = "movieId") %>%  # ... userId are included in the test set
  semi_join(train_rf, by = "userId")

# Fitting the random forest model 
rf_model <- randomForest(rating ~ movieId + userId,  # random forest fit
                         data = train_rf, 
                         type = "regression", 
                         proximity = TRUE)

# Variable importance plot
varImpPlot(rf_model)

# Random forest model evaluation through RMSE
rf_pred <- predict(rf_model, test_rf)
result_rf <- RMSE(test_rf$rating, rf_pred)




# Results section #

##### Results table #####
results_df <- data.frame(Model = c("Just the average",
                                  "Movie effect", 
                               "Movie + user effect", 
                               "Movie + user + genre effect", 
                               "Movie + user + genre + year effect", 
                               "Regularized movie + user + genre + year cv test",
                               "Regularized movie + user + genre + year validation final",
                               "Random Forest Test"), 
                          RMSE = c(result_mu, result_1, result_2, result_3, result_4, result_reg, result_5, result_rf))

  kable(results_df, caption = "Movielens rating prediction results")  


  

# Appendix #

# Environment
print("Operating System:")
version
  


