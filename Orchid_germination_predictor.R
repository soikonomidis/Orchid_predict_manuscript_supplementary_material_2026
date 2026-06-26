# ==============================================================================
# ORCHID GERMINATION PREDICTIVE MODELING
# Target: Final Germination Percentage (FG%)
# ==============================================================================

# Load libraries
library(tidyverse)
library(caret)
library(xgboost)
library(scales)
library(e1071)
library(kernlab)

# 2. Set Working Directory & Load Data =========================================

data <- readxl::read_xlsx('analysis_dataset_references_ES.xlsx')

# Create a dictionary of Species to E:S ratios to use later for the test dataset
es_dict <- data %>%
  select(Species, `E:S`) %>%
  rename(ES_ratio = `E:S`) %>%
  drop_na(ES_ratio) %>%
  distinct(Species, .keep_all = TRUE)

# 3. Data Cleaning, Preprocessing & WEIGHTING ==================================

# Define the classification function based on breakpoints
categorize_germination <- function(x) {
  cut(x, 
      breaks = c(-Inf, 30, 50, 80, Inf), 
      labels = c("Low", "Mid", "High", "Max"), 
      right = TRUE) # 'right = TRUE' means brackets are (a, b], so 0-30 includes exactly 30
}


training_summary <- data %>%
  rename(
    FG_perc = `FG%`,
    Pretreatment_duration_min = `Duration (min) of pretreatment`
  ) %>%
  mutate(Pretreatment_duration_min = as.numeric(Pretreatment_duration_min),
         FG_perc = as.numeric(FG_perc)) %>%
  group_by(Species, Pretreatment_duration_min, Subfamily, `Growth habit`, Habitat, `Climate zone`) %>%
  summarise(
    N_Petris = n(),
    Mean_FG = mean(FG_perc, na.rm = TRUE),
    SD_FG = ifelse(is.na(sd(FG_perc, na.rm = TRUE)), 0, sd(FG_perc, na.rm = TRUE)),
    SE_FG = ifelse(N_Petris > 1, SD_FG / sqrt(N_Petris), 0),
    .groups = "drop"
  ) %>%
  arrange(desc(Mean_FG))

# Identify Low Germinating Species from Training Data (< 30%)
low_germinating_training <- training_summary %>%
  filter(Mean_FG < 30)


analysis_data <- data %>%
  rename(
    FG_perc = `FG%`,
    ES_ratio = `E:S`,
    Growth_habit = `Growth habit`,
    Climate_zone = `Climate zone`,
    Pretreatment_duration_min = `Duration (min) of pretreatment` 
  ) %>%
  select(Species, FG_perc, ES_ratio, Subfamily, Growth_habit, Habitat, Climate_zone, Pretreatment_duration_min) %>%
  mutate(
    FG_perc = as.numeric(FG_perc),
    ES_ratio = as.numeric(ES_ratio),
    Pretreatment_duration_min = as.numeric(Pretreatment_duration_min),
    Subfamily = as.factor(Subfamily),
    Growth_habit = as.factor(Growth_habit),
    Habitat = as.factor(Habitat),
    Climate_zone = as.factor(Climate_zone),
    # Pre-categorize to apply training weights
    Germ_Class = categorize_germination(FG_perc) 
  ) %>%
  drop_na(FG_perc, ES_ratio, Subfamily, Growth_habit, Habitat, Climate_zone, Pretreatment_duration_min)

# Calculate inverse frequency weights to force the model to prioritize Low germination species
weight_df <- analysis_data %>%
  count(Germ_Class) %>%
  mutate(Weight = sum(n) / (n() * n)) %>%
  select(Germ_Class, Weight)

analysis_data <- analysis_data %>%
  left_join(weight_df, by = "Germ_Class")

if(nrow(analysis_data) == 0) stop("No data left! Check your CSV file column names.")

# 4. Define Iterative Feature Sets =============================================
formulas <- list(
  "Step 1: ES_ratio" = FG_perc ~ ES_ratio,
  "Step 2: +Subfamily" = FG_perc ~ ES_ratio + Subfamily,
  "Step 3: +Growth_habit" = FG_perc ~ ES_ratio + Subfamily + Growth_habit,
  "Step 4: +Habitat" = FG_perc ~ ES_ratio + Subfamily + Growth_habit + Habitat,
  "Step 5: +Climate_zone" = FG_perc ~ ES_ratio + Subfamily + Growth_habit + Habitat + Climate_zone,
  "Step 6: +Pretreatment_duration" = FG_perc ~ ES_ratio + Subfamily + Growth_habit + Habitat + Climate_zone + Pretreatment_duration_min
)

# 5. Training Setup ============================================
train_control <- trainControl(method = "cv", number = 5, savePredictions = "final", verboseIter = FALSE)
set.seed(123)

# ==============================================================================
# INDEPENDENT MODEL TRAINING BLOCKS
# Models are saved into lists for later prediction on the test set
# Sample Weights are applied to LM, RPART, RF, and XGB
# ==============================================================================

# 6. MODEL 1: LINEAR REGRESSION (lm) ===========================================
results_lm_list <- list()
models_lm <- list()
for (f_name in names(formulas)) {
  fit <- train(formulas[[f_name]], data = analysis_data, method = "lm", trControl = train_control, 
               preProcess = c("center", "scale"), weights = analysis_data$Weight)
  models_lm[[f_name]] <- fit
  results_lm_list[[f_name]] <- data.frame(Model = "lm", Feature_Set = f_name, RMSE = fit$results$RMSE, Rsquared = fit$results$Rsquared, MAE = fit$results$MAE)
}
df_lm <- bind_rows(results_lm_list)

# 7. MODEL 2: K-NEAREST NEIGHBORS (knn) ========================================
results_knn_list <- list()
models_knn <- list()
knn_grid <- expand.grid(k = c(1, 3, 5, 7, 9))
for (f_name in names(formulas)) {
  fit <- suppressWarnings(train(formulas[[f_name]], data = analysis_data, method = "knn", trControl = train_control, tuneGrid = knn_grid, preProcess = c("center", "scale")))
  models_knn[[f_name]] <- fit
  res <- fit$results[rownames(fit$bestTune), ]
  results_knn_list[[f_name]] <- data.frame(Model = "knn", Feature_Set = f_name, RMSE = res$RMSE, Rsquared = res$Rsquared, MAE = res$MAE)
}
df_knn <- bind_rows(results_knn_list)

# 8. MODEL 3: DECISION TREE (rpart) ============================================
results_rpart_list <- list()
models_rpart <- list()
rpart_grid <- expand.grid(cp = c(0.001, 0.005, 0.01, 0.02, 0.05, 0.1))
for (f_name in names(formulas)) {
  fit <- suppressWarnings(train(formulas[[f_name]], data = analysis_data, method = "rpart", trControl = train_control, 
                                tuneGrid = rpart_grid, preProcess = c("center", "scale"), weights = analysis_data$Weight))
  models_rpart[[f_name]] <- fit
  res <- fit$results[rownames(fit$bestTune), ]
  results_rpart_list[[f_name]] <- data.frame(Model = "rpart", Feature_Set = f_name, RMSE = res$RMSE, Rsquared = res$Rsquared, MAE = res$MAE)
}
df_rpart <- bind_rows(results_rpart_list)

# 9. MODEL 4: SUPPORT VECTOR MACHINE (svmRadial) ===============================
results_svm_list <- list()
models_svm <- list()
for (f_name in names(formulas)) {
  fit <- suppressWarnings(train(formulas[[f_name]], data = analysis_data, method = "svmRadial", trControl = train_control, tuneLength = 10, preProcess = c("center", "scale")))
  models_svm[[f_name]] <- fit
  res <- fit$results[rownames(fit$bestTune), ]
  results_svm_list[[f_name]] <- data.frame(Model = "svmRadial", Feature_Set = f_name, RMSE = res$RMSE, Rsquared = res$Rsquared, MAE = res$MAE)
}
df_svm <- bind_rows(results_svm_list)

# 10. MODEL 5: RANDOM FOREST (rf) ==============================================
results_rf_list <- list()
models_rf <- list()
for (f_name in names(formulas)) {
  fit <- suppressWarnings(train(formulas[[f_name]], data = analysis_data, method = "rf", trControl = train_control, 
                                tuneLength = 7, preProcess = c("center", "scale"), weights = analysis_data$Weight))
  models_rf[[f_name]] <- fit
  res <- fit$results[rownames(fit$bestTune), ]
  results_rf_list[[f_name]] <- data.frame(Model = "rf", Feature_Set = f_name, RMSE = res$RMSE, Rsquared = res$Rsquared, MAE = res$MAE)
}
df_rf <- bind_rows(results_rf_list)

# 11. MODEL 6: XGBOOST (Native xgboost package) ===================================
results_xgb_list <- list()
xgb_trained_models <- list() 

y <- analysis_data$FG_perc
var_y <- var(y)

for (f_name in names(formulas)) {
  f <- formulas[[f_name]]
  
  X <- model.matrix(f, data = analysis_data)[, -1, drop = FALSE]
  

  dtrain <- xgb.DMatrix(data = X, label = y, weight = analysis_data$Weight)
  
  xgb_params <- list(
    objective = "reg:squarederror",
    eta = 0.05,             
    max_depth = 4,          
    alpha = 0.1,            
    lambda = 0.5,           
    subsample = 0.8,        
    colsample_bytree = 0.8  
  )
  
  set.seed(123)
  cv_xgb <- xgb.cv(
    params = xgb_params, metrics = list("rmse", "mae"), data = dtrain,
    nrounds = 300, nfold = 5, early_stopping_rounds = 15, verbose = 0, showsd = FALSE
  )
  
  best_iter <- cv_xgb$best_iteration
  if (is.null(best_iter) || length(best_iter) == 0) best_iter <- which.min(cv_xgb$evaluation_log$test_rmse_mean)
  
  best_rmse <- as.numeric(cv_xgb$evaluation_log$test_rmse_mean[best_iter])
  best_mae <- as.numeric(cv_xgb$evaluation_log$test_mae_mean[best_iter])
  best_r2 <- 1 - ((best_rmse^2) / var_y)
  
  results_xgb_list[[f_name]] <- data.frame(Model = "xgbTree", Feature_Set = f_name, RMSE = best_rmse, Rsquared = best_r2, MAE = best_mae)
  
  final_xgb <- xgb.train(params = xgb_params, data = dtrain, nrounds = best_iter, verbose = 0)
  xgb_trained_models[[f_name]] <- final_xgb
}
df_xgb <- bind_rows(results_xgb_list)


# ==============================================================================
# 12. Summarize and Format Training Results
# ==============================================================================
all_results <- bind_rows(df_lm, df_knn, df_rpart, df_svm, df_rf, df_xgb)

r2_table <- all_results %>% select(Feature_Set, Model, Rsquared) %>% pivot_wider(names_from = Model, values_from = Rsquared)
rmse_table <- all_results %>% select(Feature_Set, Model, RMSE) %>% pivot_wider(names_from = Model, values_from = RMSE)

print(as.data.frame(r2_table), row.names = FALSE)


# ==============================================================================
# 13. Test Dataset: Processing & Summarization
# ==============================================================================

test_raw <- readxl::read_xlsx('Orchid_germination_analysis.xlsx')

# Calculate raw percentages, clean temperature strings, and summarize by groups WITH STATS
test_summarized <- test_raw %>%
  mutate(
    Germination_Percentage = (Number_of_Seeds_Rhizoids / Number_of_Seeds_in_Dish) * 100,
    Temperature = as.numeric(gsub("[^0-9.]", "", Temperature))
  ) %>%
  group_by(Species, Location, Temperature, Treatment_duration_min) %>%
  summarize(
    N_Petris = n(),
    Mean_FG = mean(Germination_Percentage, na.rm = TRUE),
    SD_FG = ifelse(is.na(sd(Germination_Percentage, na.rm = TRUE)), 0, sd(Germination_Percentage, na.rm = TRUE)),
    SE_FG = ifelse(N_Petris > 1, SD_FG / sqrt(N_Petris), 0),
    Actual_FG_perc = max(Germination_Percentage, na.rm = TRUE),
    Subfamily = first(Subfamily),
    Growth_habit = first(`Growth habit`),
    Habitat = first(Habitat),
    Climate_zone = first(`Climate zone`),
    .groups = "drop"
  ) %>%
  rename(Pretreatment_duration_min = Treatment_duration_min)

# Process New Armenia Test Data from CSV
armenia_raw <- readxl::read_xlsx('Orchid_Germination_Armenia.xlsx')

armenia_summarized <- armenia_raw %>%
  rename(Subfamily = Sufamily) %>% # Fix column typo
  mutate(
    Germination_Percentage = (Number_of_Seeds_Rhizoids / Number_of_Seeds_in_Dish) * 100,
    Temperature = as.numeric(Temperature),
    Treatment_duration_min = as.numeric(Treatment_duration_min)
  ) %>%
  group_by(Species, Location, Temperature, Treatment_duration_min) %>%
  summarize(
    N_Petris = n(),
    Mean_FG = mean(Germination_Percentage, na.rm = TRUE),
    SD_FG = ifelse(is.na(sd(Germination_Percentage, na.rm = TRUE)), 0, sd(Germination_Percentage, na.rm = TRUE)),
    SE_FG = ifelse(N_Petris > 1, SD_FG / sqrt(N_Petris), 0),
    Actual_FG_perc = max(Germination_Percentage, na.rm = TRUE),
    Subfamily = first(Subfamily),
    Growth_habit = first(Growth_habit),
    Habitat = first(Habitat),
    Climate_zone = first(Climate_zone),
    .groups = "drop"
  ) %>%
  rename(Pretreatment_duration_min = Treatment_duration_min)

# Combine both test datasets 
test_summarized_combined <- bind_rows(test_summarized, armenia_summarized)

# Load Morphometry dataset to calculate Test E:S ratios
morph_data <- readxl::read_xlsx('Orchid_seed_Morphometry_new_records_Armenia.xlsx')
colnames(morph_data) <- trimws(colnames(morph_data))

es_dict_morph <- morph_data %>%
  mutate(
    Testa_Length_mm = as.numeric(Testa_Length_mm),
    Embryo_Length_mm = as.numeric(Embryo_Length_mm),
    Calculated_ES = Embryo_Length_mm / Testa_Length_mm
  ) %>%
  group_by(Species) %>%
  summarize(ES_ratio_morph = mean(Calculated_ES, na.rm = TRUE), .groups = "drop")

# Combine dictionaries and format test data
es_dict_combined <- es_dict %>%
  full_join(es_dict_morph, by = "Species") %>%
  mutate(ES_ratio = coalesce(ES_ratio_morph, ES_ratio)) %>%
  select(Species, ES_ratio)

# Format test data for Model Predictions
test_analysis_data <- test_summarized %>%
  inner_join(es_dict_combined, by = "Species") %>%
  rename(FG_perc = Actual_FG_perc) %>% 
  mutate(
    Subfamily = factor(Subfamily, levels = levels(analysis_data$Subfamily)),
    Growth_habit = factor(Growth_habit, levels = levels(analysis_data$Growth_habit)),
    Habitat = factor(Habitat, levels = levels(analysis_data$Habitat)),
    Climate_zone = factor(Climate_zone, levels = levels(analysis_data$Climate_zone)),
    Pretreatment_duration_min = as.numeric(Pretreatment_duration_min),
    ES_ratio = as.numeric(ES_ratio)
  ) %>%
  drop_na(ES_ratio, Subfamily, Growth_habit, Habitat, Climate_zone, Pretreatment_duration_min)


# ==============================================================================
# 14. Test Dataset: All Models & All Steps Predictions
# ==============================================================================

test_comparisons <- test_analysis_data %>%
  select(Species, Location, Temperature, Pretreatment_duration_min, FG_perc) %>%
  rename(Actual_Max_Germination = FG_perc)

# Iterate through all formulas to generate a complete view of how predictions change
for (f_name in names(formulas)) {
  f <- formulas[[f_name]]
  step_num <- gsub("[^0-9]", "", f_name) 
  
  # Prepare XGBoost matrix (no weights needed for predicting)
  X_test <- model.matrix(f, data = test_analysis_data)[, -1, drop = FALSE]
  dtest <- xgb.DMatrix(data = X_test)
  
  # Predict across all 6 models, bounding results between 0% and 100%
  preds_lm    <- predict(models_lm[[f_name]], test_analysis_data)
  preds_knn   <- predict(models_knn[[f_name]], test_analysis_data)
  preds_rpart <- predict(models_rpart[[f_name]], test_analysis_data)
  preds_svm   <- predict(models_svm[[f_name]], test_analysis_data)
  preds_rf    <- predict(models_rf[[f_name]], test_analysis_data)
  preds_xgb   <- predict(xgb_trained_models[[f_name]], dtest)
  
  # Attach columns to the dataframe
  test_comparisons[[paste0("Step", step_num, "_LM")]]    <- pmax(0, pmin(100, round(preds_lm, 2)))
  test_comparisons[[paste0("Step", step_num, "_KNN")]]   <- pmax(0, pmin(100, round(preds_knn, 2)))
  test_comparisons[[paste0("Step", step_num, "_RPART")]] <- pmax(0, pmin(100, round(preds_rpart, 2)))
  test_comparisons[[paste0("Step", step_num, "_SVM")]]   <- pmax(0, pmin(100, round(preds_svm, 2)))
  test_comparisons[[paste0("Step", step_num, "_RF")]]    <- pmax(0, pmin(100, round(preds_rf, 2)))
  test_comparisons[[paste0("Step", step_num, "_XGB")]]   <- pmax(0, pmin(100, round(preds_xgb, 2)))
}

# ==============================================================================
# 15. Classification & Accuracy Analysis
# ==============================================================================

# Apply classification to actuals and all predictions in the comparisons dataframe
test_classifications <- test_comparisons %>%
  mutate(across(c(Actual_Max_Germination, starts_with("Step")), categorize_germination))

# Calculate Accuracy 
accuracy_results <- list()
for (f_name in names(formulas)) {
  step_num <- gsub("[^0-9]", "", f_name)
  for (mod in c("LM", "KNN", "RPART", "SVM", "RF", "XGB")) {
    col_name <- paste0("Step", step_num, "_", mod)
    
    # Calculate Overall Accuracy
    acc_overall <- mean(test_classifications[[col_name]] == test_classifications$Actual_Max_Germination, na.rm = TRUE)
    
    # Calculate Accuracy Specifically for "Low" Class to track improvements
    is_low <- test_classifications$Actual_Max_Germination == "Low"
    if (sum(is_low, na.rm = TRUE) > 0) {
      acc_low <- mean(test_classifications[[col_name]][is_low] == "Low", na.rm = TRUE)
    } else {
      acc_low <- NA
    }
    
    accuracy_results[[length(accuracy_results) + 1]] <- data.frame(
      Step = paste("Step", step_num),
      Model = mod,
      Overall_Accuracy = acc_overall,
      Low_Class_Accuracy = acc_low
    )
  }
}

accuracy_df <- bind_rows(accuracy_results)






# ==============================================================================
# 17. Orchids of Greece and Armenia Morphometry Summary
# ==============================================================================

morph_data <- readxl::read_xlsx('Orchid_seed_Morphometry_new_records_Armenia.xlsx')
colnames(morph_data) <- make.names(trimws(colnames(morph_data)), unique = TRUE)

morph_processed <- morph_data %>%
  mutate(
    Testa_Length_mm = as.numeric(Testa_Length_mm),
    Testa_Width_mm = as.numeric(Testa_Width_mm),
    Embryo_Length_mm = as.numeric(Embryo_Length_mm),
    Embryo_Width_mm = as.numeric(Embryo_Width_mm),
    ES_Ratio = Embryo_Length_mm / Testa_Length_mm,
    Species = trimws(Species)
  ) %>%
  drop_na(Testa_Length_mm, Embryo_Length_mm, ES_Ratio)

morph_summary <- morph_processed %>%
  group_by(Species, Year_of_collection, Population_Location) %>%
  summarize(
    Seed_Count = n(),
    Mean_Testa_Length_mm = round(mean(Testa_Length_mm, na.rm = TRUE), 4),
    SD_Testa_Length_mm = round(sd(Testa_Length_mm, na.rm = TRUE), 4),
    Mean_Testa_Width_mm = round(mean(Testa_Width_mm, na.rm = TRUE), 4),
    SD_Testa_Width_mm = round(sd(Testa_Width_mm, na.rm = TRUE), 4),
    Mean_Embryo_Length_mm = round(mean(Embryo_Length_mm, na.rm = TRUE), 4),
    SD_Embryo_Length_mm = round(sd(Embryo_Length_mm, na.rm = TRUE), 4),
    Mean_Embryo_Width_mm = round(mean(Embryo_Width_mm, na.rm = TRUE), 4),
    SD_Embryo_Width_mm = round(sd(Embryo_Width_mm, na.rm = TRUE), 4),
    Mean_ES_Ratio = round(mean(ES_Ratio, na.rm = TRUE), 4),
    SD_ES_Ratio = round(sd(ES_Ratio, na.rm = TRUE), 4),
    .groups = "drop"
  ) %>%
  arrange(Species, Year_of_collection, Population_Location)


# ==============================================================================
# 18. Identify the unique orchid species in the datasets
# ==============================================================================

train_raw <- readxl::read_xlsx('analysis_dataset_references_ES.xlsx') 
test_raw <- readxl::read_xlsx('Orchid_germination_analysis.xlsx')

exclude_species <- unique(tolower(trimws(c(train_raw$Species, test_raw$Species))))

# Process Morphometry Dataset
morph_data <- readxl::read_xlsx('Orchid_seed_Morphometry_new_records_Armenia.xlsx')
colnames(morph_data) <- make.names(trimws(colnames(morph_data)), unique = TRUE)

morph_summary_pred <- morph_data %>%
  mutate(
    Species = trimws(Species),
    Testa_Length_mm = as.numeric(Testa_Length_mm),
    Embryo_Length_mm = as.numeric(Embryo_Length_mm),
    ES_Ratio_Ind = Embryo_Length_mm / Testa_Length_mm
  ) %>%
  group_by(Species) %>%
  summarize(
    ES_ratio = mean(ES_Ratio_Ind, na.rm = TRUE),
    Subfamily = substr(first(Subfamily...12), 1, 1),
    Habitat = first(Habitat),
    Growth_habit = first(Growth.Habit),
    Climate_zone = first(Climate.Zone),
    .groups = "drop"
  )

# Process SE Dimensions Dataset
tab1_data <- readxl::read_xlsx('Orchidaceae_SE_dim.xlsx')
colnames(tab1_data) <- c("Species", "Seed_L", "Seed_W", "Emb_L", "Emb_W", "Habitat", "Growth_habit", "Climate_zone", "Subfamily", "Reference")

tab1_processed <- tab1_data %>%
  mutate(
    Species = trimws(Species),
    ES_ratio = as.numeric(Emb_L) / as.numeric(Seed_L)
  ) %>%
  select(Species, ES_ratio, Subfamily, Growth_habit, Habitat, Climate_zone)

# Combine and Filter
combined_new_species <- bind_rows(morph_summary_pred, tab1_processed) %>%
  distinct(Species, .keep_all = TRUE)

filtered_species_pool <- combined_new_species %>%
  filter(!(tolower(Species) %in% exclude_species)) %>%
  drop_na(ES_ratio, Subfamily, Growth_habit, Habitat, Climate_zone)

write.csv(filtered_species_pool, file.path("Figures", "Prepared_New_Species_Pool.csv"), row.names = FALSE)


# ==============================================================================
# 19. Identify the unique orchid species in the datasets (Re-Train Step 4)
# ==============================================================================

train_processed <- train_raw %>%
  rename(
    FG_perc = `FG%`,
    ES_ratio = `E:S`,
    Growth_habit = `Growth habit`,
    Climate_zone = `Climate zone`
  ) %>%
  mutate(
    FG_perc = as.numeric(FG_perc),
    ES_ratio = as.numeric(ES_ratio),
    Subfamily = as.factor(Subfamily),
    Growth_habit = as.factor(Growth_habit),
    Habitat = as.factor(Habitat),
    Climate_zone = as.factor(Climate_zone),
    Germ_Class = categorize_germination(FG_perc)
  ) %>%
  drop_na(FG_perc, ES_ratio, Subfamily, Growth_habit, Habitat)

weight_df <- train_processed %>%
  count(Germ_Class) %>%
  mutate(Weight = sum(n) / (n() * n)) %>%
  select(Germ_Class, Weight)

train_processed <- train_processed %>% left_join(weight_df, by = "Germ_Class")

# Refocused on Step 4
form_step4 <- FG_perc ~ ES_ratio + Subfamily + Growth_habit + Habitat
train_control <- trainControl(method = "cv", number = 5)

set.seed(123)
knn_grid <- expand.grid(k = c(1, 3, 5, 7, 9))
knn_model <- train(form_step4, data = train_processed, method = "knn",
                   trControl = train_control, tuneGrid = knn_grid,
                   preProcess = c("zv", "center", "scale"))

set.seed(123)
svm_model <- suppressWarnings(
  train(form_step4, data = train_processed, method = "svmRadial",
        trControl = train_control, tuneLength = 10,
        preProcess = c("zv", "center", "scale"), weights = train_processed$Weight)
)

new_species_pool <- readxl::read_xlsx('Species_predi.xlsx') %>%
  mutate(
    Subfamily = factor(Subfamily, levels = levels(train_processed$Subfamily)),
    Growth_habit = factor(Growth_habit, levels = levels(train_processed$Growth_habit)),
    Habitat = factor(Habitat, levels = levels(train_processed$Habitat)),
    Climate_zone = factor(Climate_zone, levels = levels(train_processed$Climate_zone))
  ) %>%
  drop_na() 

predictions_df <- new_species_pool %>%
  mutate(
    Pred_KNN_Raw = round(pmax(0, pmin(100, predict(knn_model, new_species_pool))), 1),
    Pred_SVM_Raw = round(pmax(0, pmin(100, predict(svm_model, new_species_pool))), 1),
    Class_KNN = categorize_germination(Pred_KNN_Raw),
    Class_SVM = categorize_germination(Pred_SVM_Raw)
  ) %>%
  arrange(Species)

write.csv(predictions_df, file.path("Figures", "Final_New_Species_Predictions.csv"), row.names = FALSE)