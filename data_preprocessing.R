
#import libraries
library(tidyverse)    
library(lubridate)    
library(themis)       
library(recipes)      
library(rsample)      

df_raw <- read_csv("Buy_Now_Pay_Later_BNPL_CreditRisk_Dataset.csv")


#remove columns

df_clean <- df_raw %>%
  select(-user_id, -transaction_date, -customer_segment, -risk_score, -monthly_income)


# data transformation
df_clean <- df_clean %>%
  mutate(
    default_flag = factor(default_flag,
                          levels = c(0, 1),
                          labels = c("No_Default", "Default")),
    
    # transform categorical variables to factor
    employment_type  = factor(employment_type),
    product_category = factor(product_category),
    location         = factor(location),
    bnpl_installments = factor(bnpl_installments, levels = c(3, 6, 9, 12))
  )

df_clean <- df_clean %>%
  filter(
    debt_to_income_ratio <= quantile(debt_to_income_ratio, 0.99),
    missed_payments      <= quantile(missed_payments, 0.99),
    repayment_delay_days <= quantile(repayment_delay_days, 0.99)
  )

print(table(df_clean$default_flag))
print(prop.table(table(df_clean$default_flag)))


#Train/Test Split

set.seed(42)

split <- initial_split(df_clean, prop = 0.80, strata = default_flag)

train_raw <- training(split)
test_df   <- testing(split)


# apply SMOTE on training dataset

recipe_smote <- recipe(default_flag ~ ., data = train_raw) %>%
  step_dummy(all_nominal_predictors(), one_hot = FALSE) %>%
  step_smote(default_flag, over_ratio = 1, neighbors = 5)

prep_smote  <- prep(recipe_smote, training = train_raw)
train_df    <- bake(prep_smote, new_data = NULL)


# only do dummy encoding on testing dataset

recipe_test <- recipe(default_flag ~ ., data = train_raw) %>%
  step_dummy(all_nominal_predictors(), one_hot = FALSE)

prep_test <- prep(recipe_test, training = train_raw)
test_df   <- bake(prep_test, new_data = test_df)

saveRDS(train_df, "train_df.rds")
saveRDS(test_df,  "test_df.rds")

write_csv(
  tibble(column = names(train_df)),
  "column_names.csv"
)

