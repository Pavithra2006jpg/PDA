# Forecasting demand for hourly employees with Holt-Winters (base R)
# Monthly demand, Jan 2023 - Sep 2026 (sample data)
demand <- c(279, 272, 293, 308, 321, 326, 323, 317, 316, 336, 355, 359,
            305, 310, 322, 326, 343, 348, 349, 345, 336, 358, 380, 388,
            336, 333, 341, 358, 366, 378, 373, 373, 365, 381, 403, 415,
            357, 351, 374, 381, 401, 408, 396, 401, 387)
ts_data <- ts(demand, start = c(2023, 1), frequency = 12)
print(ts_data)

# ---- Step 1: trend and seasonality ----
png("q8_decompose.png", width = 1200, height = 900, res = 130)
plot(decompose(ts_data))
dev.off()

# ---- Step 2: check accuracy on the last 12 months ----
train  <- window(ts_data, end = c(2025, 9))
test   <- window(ts_data, start = c(2025, 10))
fit_tr <- HoltWinters(train)
fc_tr  <- predict(fit_tr, n.ahead = 12)

rmse <- sqrt(mean((test - fc_tr)^2))
mape <- mean(abs((test - fc_tr) / test)) * 100
cat("Test RMSE:", round(rmse, 2), "\n")
cat("Test MAPE:", round(mape, 2), "%\n")

png("q8_test.png", width = 1200, height = 700, res = 130)
ts.plot(train, test, fc_tr, col = c("black", "blue", "red"), lwd = 2,
        ylab = "Hourly employees needed",
        main = "Holt-Winters: actual vs forecast (last 12 months)")
legend("topleft", c("Training", "Actual", "Forecast"),
       col = c("black", "blue", "red"), lwd = 2)
dev.off()

# ---- Step 3: final model on all data and next-month forecast ----
fit <- HoltWinters(ts_data)
cat("alpha =", round(fit$alpha, 3), " beta =", round(fit$beta, 3),
    " gamma =", round(fit$gamma, 3), "\n")

next_month <- predict(fit, n.ahead = 1)
cat("Forecast for next month (Oct 2026):\n")
print(round(next_month, 1))

fc6 <- predict(fit, n.ahead = 6, prediction.interval = TRUE)
print(round(fc6, 1))

png("q8_forecast.png", width = 1200, height = 700, res = 130)
plot(fit, fc6, ylab = "Hourly employees needed",
     main = "Next 6 months forecast with prediction interval")
dev.off()
