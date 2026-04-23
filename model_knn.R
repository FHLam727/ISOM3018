# ============================================================
# k-NN — BNPL default classification
# Uses shared train/test data from preprocessing output
# Tunes k with CV on training data only
# Main report version: modeling_knn_logistic.Rmd
# ============================================================

library(tidymodels)
library(tidyverse)

if (!file.exists("train_df.rds") || !file.exists("test_df.rds")) {
  stop("Missing train_df.rds or test_df.rds. Run data_preprocessing.R from the project root.")
}

train_df <- readRDS("train_df.rds")
test_df  <- readRDS("test_df.rds")

nm_tr <- names(train_df)
nm_te <- names(test_df)
if (!setequal(nm_tr, nm_te)) {
  stop("train_df and test_df do not contain the same columns. Recreate both RDS files using data_preprocessing.R.")
}
if (!identical(nm_tr, nm_te)) {
  test_df <- test_df %>% select(all_of(nm_tr))
  message("Aligned test_df column order to train_df.")
}

event_lvl <- "second"

# Normalize numeric predictors for distance-based model
rec_knn <- recipe(default_flag ~ ., data = train_df) %>%
  step_normalize(all_numeric_predictors())

knn_spec <- nearest_neighbor(neighbors = tune()) %>%
  set_engine("kknn") %>%
  set_mode("classification")

wf_knn <- workflow() %>%
  add_recipe(rec_knn) %>%
  add_model(knn_spec)

# Candidate odd k values to reduce tie risk
k_grid <- tibble(neighbors = seq(3, 101, by = 2))

set.seed(42)
folds <- vfold_cv(train_df, v = 5, strata = default_flag)

set.seed(42)
tune_res <- tune_grid(
  wf_knn,
  resamples = folds,
  grid = k_grid,
  metrics = metric_set(roc_auc, accuracy),
  control = control_grid(save_pred = FALSE, verbose = FALSE)
)

best_k <- tune_res %>% select_best(metric = "roc_auc")
cat("\n=== k-NN — CV selection (max roc_auc) ===\n")
print(best_k)

wf_knn_final <- finalize_workflow(wf_knn, best_k)
set.seed(42)
fit_knn <- fit(wf_knn_final, data = train_df)

# Generate aligned prediction output for evaluation teammate
test_aug <- augment(fit_knn, test_df)

pred_cols <- names(test_aug)[startsWith(names(test_aug), ".pred_")]
pred_export <- test_aug %>%
  mutate(row_id = row_number(), .before = 1) %>%
  select(row_id, default_flag, all_of(pred_cols))

write_csv(pred_export, "predictions_knn_test.csv")

cat("\n=== k-NN — Test set metrics (sanity check) ===\n")
cat("(Positive class: Default, event_level = second)\n\n")

m_roc  <- roc_auc(test_aug, truth = default_flag, .pred_Default, event_level = event_lvl)
m_acc  <- accuracy(test_aug, truth = default_flag, .pred_class)
m_sens <- yardstick::sens(test_aug, truth = default_flag, estimate = .pred_class, event_level = event_lvl)
m_spec <- yardstick::spec(test_aug, truth = default_flag, estimate = .pred_class, event_level = event_lvl)

metrics_tbl <- bind_rows(m_roc, m_acc, m_sens, m_spec)
print(metrics_tbl)

cm <- conf_mat(test_aug, truth = default_flag, estimate = .pred_class)
cat("\n=== Confusion matrix (test) ===\n")
print(cm)
