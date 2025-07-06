library(rpart)          # Regresyon Ağacı
library(randomForest)   # Random Forest
library(ipred)          # Bagging
library(Metrics)  
library(ggplot2)
library(reshape2)
library(patchwork)
library(dplyr)
library(magrittr)
library(pROC)
library(rpart)
library(ipred)
library(MASS)             
library(glmnet)
library(caret)

#checking dataset
df <- read.csv("/Users/sim/Downloads/2019.csv")
head(df)
#delete to "overall rank" column
df <- subset(df, select = -Overall.rank)
head(df)
str(df)
summary(df)
dim(df)
#is there any null column?
colSums(is.na(df))
#how many are there different countries?
length(unique(df$Country.or.region))
#names
unique(df$Country.or.region)

ggplot(df, aes(x = Score)) +
  geom_histogram(bins = 30, fill = "pink", color = "black") +
  ggtitle("Histogram of Score")


numeric_vars <- c("Score", "GDP.per.capita", "Social.support", "Healthy.life.expectancy",
                  "Freedom.to.make.life.choices", "Generosity", "Perceptions.of.corruption")

cor_matrix <- cor(df[, numeric_vars])
print(cor_matrix)


p1 <- ggplot(df, aes(x = GDP.per.capita, y = Score)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  ggtitle("GDP per Capita vs Score")

p2 <- ggplot(df, aes(x = Social.support, y = Score)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  ggtitle("Social Support vs Score")

p1 + p2


mean_scores <- aggregate(Score ~ Country.or.region, data = df, FUN = mean)
mean_scores <- mean_scores[order(-mean_scores$Score), ]
print(mean_scores)

top10 <- head(mean_scores, 10)
top10

ggplot(top10, aes(x = reorder(Country.or.region, Score), y = Score, fill = Country.or.region)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  ggtitle("Top 10 Countries by Happiness Score") +
  xlab("Country") +
  ylab("Average Score") +
  theme(legend.position = "none")  


least10 <- tail(mean_scores, 10)
least10

ggplot(least10, aes(x = reorder(Country.or.region, Score), y = Score,  fill = Country.or.region)) +
         geom_bar(stat = "identity") +
  coord_flip() +
  ggtitle("Least 10 Countries by Happiness Score") +
  xlab("Country") +
  ylab("Average Score") +
  theme(legend.position = "none")

#### spliting to data
all_countries <- factor(df$Country.or.region)
set.seed(37) #student id
n <- nrow(df)
train_indices <- sample(1:n, size = 0.7 * n)
train_data <- df[train_indices, ]
test_data <- df[-train_indices, ]

train_data$Country.or.region <- factor(train_data$Country.or.region, levels = levels(all_countries))
test_data$Country.or.region <- factor(test_data$Country.or.region, levels = levels(all_countries))

#  One-hot encoding
ohe_train <- model.matrix(~ Country.or.region - 1, data = train_data)
ohe_test <- model.matrix(~ Country.or.region - 1, data = test_data)

missing_cols <- setdiff(colnames(ohe_train), colnames(ohe_test))
for (col in missing_cols) {
  ohe_test <- cbind(ohe_test, rep(0, nrow(ohe_test)))
  colnames(ohe_test)[ncol(ohe_test)] <- col
}
ohe_test <- ohe_test[, colnames(ohe_train)]

train_data_no_country <- train_data[, !(names(train_data) %in% "Country.or.region")]
test_data_no_country <- test_data[, !(names(test_data) %in% "Country.or.region")]

train_final <- cbind(train_data_no_country, ohe_train)
test_final <- cbind(test_data_no_country, ohe_test)

clean_names <- function(x) {
  x <- gsub(" ", "_", x)          
  x <- gsub("[^[:alnum:]_]", "", x)  
  return(x)
}

colnames(train_final) <- clean_names(colnames(train_final))
colnames(test_final)  <- clean_names(colnames(test_final))

# Linear Regression
lm_model <- lm(Score ~ ., data = train_final)
lm_pred <- predict(lm_model, newdata = test_final)

# Regression Tree
rt_model <- rpart(Score ~ ., data = train_final, method = "anova")
rt_pred <- predict(rt_model, newdata = test_final)

# Bagging
brt_model <- bagging(Score ~ ., data = train_final, nbagg = 50)
brt_pred <- predict(brt_model, newdata = test_final)

# Random Forest
rfr_model <- randomForest(Score ~ ., data = train_final, ntree = 100, mtry = 3, importance = TRUE)
rfr_pred <- predict(rfr_model, newdata = test_final)


actual <- test_final$Score

calc_metrics <- function(actual, predicted) {
  rmse_val <- rmse(actual, predicted)
  mae_val <- mae(actual, predicted)
  r2_val <- 1 - sum((actual - predicted)^2) / sum((actual - mean(actual))^2)
  return(c(RMSE = rmse_val, MAE = mae_val, R2 = r2_val))
}

results <- data.frame(
  Model = c("Linear Regression", "Regression Tree", "Bagging", "Random Forest"),
  t(rbind(
    calc_metrics(actual, lm_pred),
    calc_metrics(actual, rt_pred),
    calc_metrics(actual, brt_pred),
    calc_metrics(actual, rfr_pred)
  ))
)

colnames(results)[2:4] <- c("RMSE", "MAE", "R2")

print(results)

results_long <- melt(results, id.vars = "Model", variable.name = "Metric", value.name = "Value")

ggplot(results_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  facet_wrap(~ Metric, scales = "free_y") +
  theme_minimal() +
  ggtitle("Model Performans Metrikleri") +
  ylab("Değer") +
  xlab("Model") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


#Turkiye icin bagging yontem ile deneme
new_obs <- data.frame(
  GDP.per.capita = 1.1,
  Social.support = 0.8,
  Healthy.life.expectancy = 0.6,
  Freedom.to.make.life.choices = 0.4,
  Generosity = 0.1,
  Perceptions.of.corruption = 0.2
)

country_cols <- grep("^Country\\.or\\.region", colnames(df), value = TRUE)
country_ohe <- as.data.frame(matrix(0, nrow = 1, ncol = length(country_cols)))
colnames(country_ohe) <- country_cols
country_ohe[1, "Country.or.regionTurkey"] <- 1  # Örneğin Turkey

colnames(country_ohe) <- clean_names(colnames(country_ohe))
colnames(new_obs) <- clean_names(colnames(new_obs))

new_input <- cbind(new_obs, country_ohe)

missing_cols <- setdiff(colnames(train_final), colnames(new_input))
for (col in missing_cols) {
  new_input[, col] <- 0
}
new_input <- new_input[, colnames(train_final)]

predicted_score <- predict(brt_model, newdata = new_input)
print(paste("Happiness Score that predicted for Turkiye:", round(predicted_score, 3)))

df[df$Country.or.region == "Turkey", c("Country.or.region", "Score")]

### Finland icin tahmin
new_obs <- data.frame(
  GDP.per.capita = 1.5,
  Social.support = 1.3,
  Healthy.life.expectancy = 1,
  Freedom.to.make.life.choices = 0.9,
  Generosity = 0.2,
  Perceptions.of.corruption = 0.1
)

# Ülke sütunlarını 0'la doldurup Finland'i 1 yapmipi secmek
country_cols <- grep("^Country\\.or\\.region", colnames(df), value = TRUE)
country_ohe <- as.data.frame(matrix(0, nrow = 1, ncol = length(country_cols)))
colnames(country_ohe) <- country_cols
country_ohe[1, "Country.or.regionFinland"] <- 1  # Finland olarak işaretlemek

# Sütun adlarını temizlemek (-,_,*,: vs.)
colnames(country_ohe) <- clean_names(colnames(country_ohe))
colnames(new_obs) <- clean_names(colnames(new_obs))

new_input <- cbind(new_obs, country_ohe)

missing_cols <- setdiff(colnames(train_final), colnames(new_input))
for (col in missing_cols) {
  new_input[, col] <- 0
}
new_input <- new_input[, colnames(train_final)]  

# Tahmin yapmak (gercek vs tahmin)
predicted_score <- predict(brt_model, newdata = new_input)
print(paste("Happiness Score that predicted for Finland:", round(predicted_score, 3)))

df[df$Country.or.region == "Finland", c("Country.or.region", "Score")]






####### Soru 2
# 
# 1. Threshold 
set.seed(37)
threshold_init <- median(train_final$Score)

train_final$Score_Class <- ifelse(train_final$Score >= threshold_init, 1, 0)
test_final$Score_Class  <- ifelse(test_final$Score >= threshold_init, 1, 0)
train_final$Score <- NULL
test_final$Score  <- NULL
table(train_final$Score_Class)

ct_model <- rpart(Score_Class ~ ., data = train_final_clean, method = "class")
ct_pred <- predict(ct_model, newdata = test_final_clean, type = "prob")[,2]
roc_ct <- roc(test_final$Score_Class, ct_pred)
cat("CT Model AUC:", auc(roc_ct), "\n")
plot(roc_ct, main = "ROC Curve - CT Model")

###bagging model
bct_model <- bagging(Score_Class ~ ., data = train_final, nbagg = 50)

bct_pred <- predict(bct_model, newdata = test_final, type = "prob")
head(bct_pred)

roc_bct <- roc(test_final$Score_Class, bct_pred)
cat("Bagginf Model AUC:", auc(roc_bct), "\n")
plot(roc_bct)


#random forrest
train_final$Score_Class <- as.factor(train_final$Score_Class)
rfc_model <- randomForest(Score_Class ~ ., data = train_final, ntree = 100)
rfc_pred <- predict(rfc_model, newdata = test_final, type = "prob")[,2]
roc_rfc <- roc(test_final$Score_Class, rfc_pred)
auc_rfc <- auc(roc_rfc)

cat("Random Forest AUC:", auc_rfc, "\n")


#logistic regression
x_train <- model.matrix(Score_Class ~ ., data = train_final)[, -1]
y_train <- train_final$Score_Class  
cv_ridge <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 0)
x_test <- model.matrix(Score_Class ~ ., data = test_final)[, -1]
lr_pred <- predict(cv_ridge, newx = x_test, s = "lambda.min", type = "response")
lr_pred <- as.vector(lr_pred)  # vektöre çevir
roc_lr <- roc(test_final$Score_Class, lr_pred)
auc_lr <- auc(roc_lr)
cat("Lojistik Regresyon AUC:", auc_lr, "\n")
plot(roc_lr, col = "red", main = "ROC Eğrisi - Lojistik Regresyon")


#LDA
lda_model <- lda(Score_Class ~ ., data = train_final_clean)
lda_pred <- predict(lda_model, newdata = test_final_clean)$posterior[,2]  # sınıf 1 için olasılık
roc_lda <- roc(test_final_clean$Score_Class, lda_pred)
auc_lda <- auc(roc_lda)

cat("LDA AUC:", auc_lda, "\n")
plot(roc_lda, col = "blue", main = "ROC Eğrisi - LDA")

#qda
top_vars <- c("GDPpercapita", "Socialsupport", "Healthylifeexpectancy")
threshold_init <- median(train_data$Score)
train_data$Score_Class <- ifelse(train_data$Score >= threshold_init, 1, 0)
test_data$Score_Class  <- ifelse(test_data$Score >= threshold_init, 1, 0)

test_final$Score_Class <- test_data$Score_Class
train_qda <- dplyr::select(train_final, all_of(c(top_vars, "Score_Class")))
qda_model <- qda(Score_Class ~ ., data = train_qda)

qda_pred <- predict(qda_model, newdata = dplyr::select(test_final, all_of(top_vars)))
qda_prob <- qda_pred$posterior[, 2]

roc_qda <- roc(test_final$Score_Class, qda_prob)
cat("QDA Model AUC:", auc(roc_qda), "\n")
plot(roc_qda, main = "ROC Curve - QDA Model")

plot(roc(test_final$Score_Class, ct_pred), col = "black", lwd = 2, main = "ROC Curves for All Models")

lines(roc(test_final$Score_Class, bct_pred), col = "purple", lwd = 2)
lines(roc(test_final$Score_Class, rfc_pred), col = "green", lwd = 2)
lines(roc(test_final$Score_Class, lr_pred), col = "red", lwd = 2)
lines(roc(test_final$Score_Class, lda_pred), col = "blue", lwd = 2)
lines(roc(test_final$Score_Class, qda_prob), col = "orange", lwd = 2)

# AUC
auc_ct <- auc(roc(test_final$Score_Class, ct_pred))
auc_bct <- auc(roc(test_final$Score_Class, bct_pred))
auc_rfc <- auc(roc(test_final$Score_Class, rfc_pred))
auc_lr <- auc(roc(test_final$Score_Class, lr_pred))
auc_lda <- auc(roc(test_final$Score_Class, lda_pred))
auc_qda <- auc(roc(test_final$Score_Class, qda_prob))

# Grafik içine legend ekleyelim
legend("topright",
       legend = c(
         paste0("CT (AUC=", round(auc_ct, 3), ")"),
         paste0("Bagging (AUC=", round(auc_bct, 3), ")"),
         paste0("Random Forest (AUC=", round(auc_rfc, 3), ")"),
         paste0("Logistic Reg (AUC=", round(auc_lr, 4), ")"),
         paste0("LDA (AUC=", round(auc_lda, 3), ")"),
         paste0("QDA (AUC=", round(auc_qda, 3), ")")
       ),
       col = c("black", "purple", "green", "red", "blue", "orange"),
       lwd = 2,
       cex = 0.8)



cat("AUC Values:\n")
cat("CT:", auc(roc_ct), "\n")
cat("Bagging:", auc(roc_bct), "\n")
cat("Random Forest:", auc(roc_rfc), "\n")
cat("Logistic Regression:", auc(roc_lr), "\n")
cat("LDA:", auc(roc_lda), "\n")
cat("QDA:", auc(roc_qda), "\n")



# Random Forest tahmini
rfc_pred_prob <- predict(rfc_model, newdata = test_final, type = "prob")[, 2]

threshold_rf <- median(train_final$Score_Class)  # ama Score_Class 0/1 olduğu için median = 0.5 gibi olabilir, ama 0/1 dizisinde median 0.5'dir

rfc_pred_class <- ifelse(rfc_pred_prob >= 0.58, 1, 0)  

confusionMatrix(factor(rfc_pred_class), factor(test_final$Score_Class))

ct_class <- ifelse(ct_pred >= 0.23, 1, 0)
confusionMatrix(factor(ct_class), factor(test_final$Score_Class))

bct_class <- ifelse(bct_pred >= 0.5155516, 1, 0)
confusionMatrix(factor(bct_class), factor(test_final$Score_Class))

lr_class <- ifelse(lr_pred >= 0.51, 1, 0)
confusionMatrix(factor(lr_class), factor(test_final$Score_Class))

lda_class <- ifelse(lda_pred >= 0.48, 1, 0)
confusionMatrix(factor(lda_class), factor(test_final$Score_Class))

qda_class <- ifelse(qda_prob >= 0.5844883, 1, 0)
confusionMatrix(factor(qda_class), factor(test_final$Score_Class))


roc_obj <- roc(test_final$Score_Class, rfc_pred_prob)

# Youden indeksine göre optimal eşik
coords(roc_obj, "best", best.method = "youden", ret = c("threshold", "sensitivity", "specificity"))

library(pROC)

get_best_threshold_metrics <- function(roc_obj) {
  coords_df <- coords(roc_obj, "all", ret = c("threshold", "sensitivity", "specificity"))
  coords_df$youden <- coords_df$sensitivity + coords_df$specificity - 1
  best_idx <- which.max(coords_df$youden)
  best_threshold <- coords_df$threshold[best_idx]
  best_sensitivity <- coords_df$sensitivity[best_idx]
  best_specificity <- coords_df$specificity[best_idx]
  
  list(
    threshold = best_threshold,
    sensitivity = best_sensitivity,
    specificity = best_specificity,
    youden = coords_df$youden[best_idx]
  )
}


best_ct <- get_best_threshold_metrics(roc_ct)
best_bagging <- get_best_threshold_metrics(roc_bct)
best_rf <- get_best_threshold_metrics(roc_rfc)
best_lr <- get_best_threshold_metrics(roc_lr)
best_lda <- get_best_threshold_metrics(roc_lda)
best_qda <- get_best_threshold_metrics(roc_qda)

best_ct
best_bagging
best_rf
best_lr
best_lda
best_qda

library(knitr)

results_thresholds <- data.frame(
  Model = c("CT", "Bagging", "Random Forest", "Logistic Regression", "LDA", "QDA"),
  Best_Threshold = c(best_ct$threshold, best_bagging$threshold, best_rf$threshold, best_lr$threshold, best_lda$threshold, best_qda$threshold)
)

kable(results_thresholds, caption = "Modellerin Optimum Eşik Değerleri (Threshold)")






##### Random forest ile deneme ####

new_obs <- data.frame(
  GDP.per.capita = 1.1,
  Social.support = 0.8,
  Healthy.life.expectancy = 0.6,
  Freedom.to.make.life.choices = 0.4,
  Generosity = 0.1,
  Perceptions.of.corruption = 0.2
)
country_cols <- grep("^Country\\.or\\.region", colnames(train_final), value = TRUE)
country_ohe <- as.data.frame(matrix(0, nrow = 1, ncol = length(country_cols)))
colnames(country_ohe) <- country_cols

country_ohe[1, "Country.or.regionTurkey"] <- 1  

clean_names <- function(x) {
  x <- gsub(" ", "_", x)
  x <- gsub("[^[:alnum:]_]", "", x)
  return(x)
}

colnames(country_ohe) <- clean_names(colnames(country_ohe))
colnames(new_obs) <- clean_names(colnames(new_obs))

new_input <- cbind(new_obs, country_ohe)

missing_cols <- setdiff(colnames(train_final)[!colnames(train_final) %in% "Score_Class"], colnames(new_input))
for (col in missing_cols) {
  new_input[, col] <- 0
}

new_input <- new_input[, colnames(train_final)[!colnames(train_final) %in% "Score_Class"]]

predicted_score <- predict(rfr_model, newdata = new_input)
print(paste("Random Forest ile tahmin edilen Score:", round(predicted_score, 3)))
predicted_prob <- predict(rfc_model, newdata = new_input, type = "prob")[,2]

# Optimal threshold'u kullanarak sınıf tahmini
threshold <- 0.58 
predicted_class <- ifelse(predicted_prob >= threshold, 1, 0)
print(paste("Tahmin edilen sınıf:", predicted_class))
print(paste("Türkiye için tahmin edilen skor:", round(predicted_score, 3)))

turkey_real_score <- df[df$Country.or.region == "Turkey", "Score"]
turkey_real_class <- ifelse(turkey_real_score >= median(df$Score), 1, 0)
cat("Türkiye gerçek sınıf:", turkey_real_class, "\n")
cat("Türkiye gerçek skor:", turkey_real_score, "\n")



finland_real_score <- df[df$Country.or.region == "Finland", "Score"]
finland_real_class <- ifelse(finland_real_score >= median(df$Score), 1, 0)

cat("Finlandiya gerçek sınıf:", finland_real_class, "\n")
cat("Finlandiya gerçek skor:", finland_real_score, "\n")


# Finlandiya için yeni gözlem
new_obs_finland <- data.frame(
  GDP.per.capita = 2,
  Social.support = 1.7,
  Healthy.life.expectancy = 1.5,
  Freedom.to.make.life.choices = 1.6,
  Generosity = 0.4,
  Perceptions.of.corruption = 0.1
)

# Ülke one-hot encoding kolonları
country_ohe_finland <- as.data.frame(matrix(0, nrow = 1, ncol = length(country_cols)))
colnames(country_ohe_finland) <- country_cols

# Finlandiya=1
country_ohe_finland[1, "Country.or.regionFinland"] <- 1

colnames(country_ohe_finland) <- clean_names(colnames(country_ohe_finland))
colnames(new_obs_finland) <- clean_names(colnames(new_obs_finland))

new_input_finland <- cbind(new_obs_finland, country_ohe_finland)

missing_cols_finland <- setdiff(colnames(train_final)[!colnames(train_final) %in% "Score_Class"], colnames(new_input_finland))
for (col in missing_cols_finland) {
  new_input_finland[, col] <- 0
}

new_input_finland <- new_input_finland[, colnames(train_final)[!colnames(train_final) %in% "Score_Class"]]

predicted_score_finland <- predict(rfr_model, newdata = new_input_finland)
print(paste("Random Forest ile tahmin edilen Score (Finlandiya):", round(predicted_score_finland, 3)))

predicted_prob_finland <- predict(rfc_model, newdata = new_input_finland, type = "prob")[,2]
threshold <- 0.58
predicted_class_finland <- ifelse(predicted_prob_finland >= threshold, 1, 0)
print(paste("Tahmin edilen sınıf (Finlandiya):", predicted_class_finland))

finland_real_score <- df[df$Country.or.region == "Finland", "Score"]
finland_real_class <- ifelse(finland_real_score >= median(df$Score), 1, 0)

cat("Finlandiya gerçek sınıf:", finland_real_class, "\n")
cat("Finlandiya gerçek skor:", finland_real_score, "\n")
