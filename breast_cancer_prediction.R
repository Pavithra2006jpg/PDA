# Breast cancer diagnosis using logistic regression (base R only)
data <- read.csv("breast_cancer.csv")
data$diagnosis <- factor(data$diagnosis, levels = c("Benign", "Malignant"))

str(data)
print(table(data$diagnosis))

# ---- Train / test split (70 : 30) ----
set.seed(2024)
idx   <- sample(seq_len(nrow(data)), size = round(0.7 * nrow(data)))
train <- data[idx, ]
test  <- data[-idx, ]

# ---- Logistic regression model ----
model <- glm(diagnosis ~ ., data = train, family = binomial)
print(summary(model)$coefficients)

# ---- Prediction and evaluation ----
prob <- predict(model, newdata = test, type = "response")
pred <- factor(ifelse(prob > 0.5, "Malignant", "Benign"),
               levels = c("Benign", "Malignant"))

cm <- table(Predicted = pred, Actual = test$diagnosis)
print(cm)

TP <- cm["Malignant", "Malignant"]; TN <- cm["Benign", "Benign"]
FP <- cm["Malignant", "Benign"];    FN <- cm["Benign", "Malignant"]
cat("Accuracy   :", round((TP + TN) / sum(cm), 4), "\n")
cat("Sensitivity:", round(TP / (TP + FN), 4), "\n")
cat("Specificity:", round(TN / (TN + FP), 4), "\n")
cat("Precision  :", round(TP / (TP + FP), 4), "\n")

# ---- ROC curve (computed manually) ----
th  <- sort(unique(c(0, prob, 1)), decreasing = TRUE)
tpr <- sapply(th, function(t) mean(prob[test$diagnosis == "Malignant"] >= t))
fpr <- sapply(th, function(t) mean(prob[test$diagnosis == "Benign"] >= t))
auc <- sum(diff(fpr) * (head(tpr, -1) + tail(tpr, -1)) / 2)
cat("AUC        :", round(auc, 4), "\n")

# ---- Graphs ----
png("q7_eda.png", width = 1400, height = 550, res = 130)
par(mfrow = c(1, 2))
barplot(table(data$diagnosis), col = c("steelblue", "firebrick"),
        main = "Benign vs Malignant cases", ylab = "Number of patients")
boxplot(radius_mean ~ diagnosis, data = data, col = c("steelblue", "firebrick"),
        main = "Mean radius by diagnosis", ylab = "Mean radius")
dev.off()

png("q7_roc.png", width = 1000, height = 700, res = 130)
plot(fpr, tpr, type = "l", lwd = 2, col = "darkgreen",
     xlab = "False positive rate", ylab = "True positive rate",
     main = paste("ROC curve (AUC =", round(auc, 3), ")"))
abline(0, 1, lty = 2)
dev.off()
