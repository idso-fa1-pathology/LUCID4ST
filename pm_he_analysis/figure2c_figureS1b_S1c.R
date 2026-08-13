library(tidyverse)
library(caret)
library(stringr)


####Evaluation of LUCID-Fig2C, FigS1B, 1C#####
setwd('/Volumes/yuan_lab/TIER2/anthracosis/0AI_PM_ST_code/pm_he_analysis')
load('../data/LUCIDEval.RData')
path_all <- merge(path1_all, path2_all, by='file_name')
path_all$patient_id <- str_extract(path_all$file_name, "(N|NS|NS |MS)0\\d+") #"NS0" or "N0" or "MS0" or 'NS 0"
path_all$slide_id <- str_remove(path_all$file_name, "_region_.*")
path_all$slide_id <- str_remove(path_all$slide_id, "^\\d+_\\d+_")


# Function to compute mean and CI
compute_ci <- function(x, conf.level = 0.95) {
  x <- x[!is.na(x)]
  n <- length(x)
  
  if (n == 0) {
    return(tibble(
      mean_value = NA_real_,
      lowerCI   = NA_real_,
      upperCI   = NA_real_
    ))
  }
  m <- mean(x)
  if (n == 1 || sd(x) == 0) {
    return(tibble(
      mean_value = m,
      lowerCI   = m,
      upperCI   = m
    ))
  }
  
  test <- t.test(x, conf.level = conf.level)
  
  tibble(
    mean_value = unname(test$estimate),
    lowerCI   = test$conf.int[1],
    upperCI   = test$conf.int[2]
  )
}


#bootstrap Function to compute mean and CI- patche based
compute_boot_ci <- function(x, conf.level = 0.95,
                            n_boot = 5000,
                            seed = 123) {
  x <- x[!is.na(x)]
  n <- length(x)
  
  if (n == 0) {
    return(tibble(
      mean_value = NA_real_,
      lowerCI   = NA_real_,
      upperCI   = NA_real_
    ))
  }
  
  m <- mean(x)
  s <- sd(x)
  
  if (n == 1 || is.na(s) || s < .Machine$double.eps) {
    return(tibble(
      mean_value = m,
      lowerCI   = m,
      upperCI   = m
    ))
  }
  
  set.seed(seed)
  
  boot_means <- replicate(n_boot, {
    mean(sample(x, size = n, replace = TRUE))
  })
  
  alpha <- 1 - conf.level
  
  tibble(
    mean_value = m,
    lowerCI   = unname(quantile(boot_means, probs = alpha / 2)),
    upperCI   = unname(quantile(boot_means, probs = 1 - alpha / 2))
  )
}


####Fig2C, dice
df_long <- path_all %>%
  pivot_longer(cols = c(dice1, dice2),
               names_to = "pathologist",
               values_to = "dice") %>%
  mutate(pathologist = ifelse(pathologist == "dice1", "Pathologist 1", "Pathologist 2"),
         pgmn_status = case_when(
           pathologist == "Pathologist 1" ~ path1_pgmn,
           pathologist == "Pathologist 2" ~ path2_pgmn
         ))


df_summary <- df_long %>%
  group_by(pathologist, pgmn_status) %>%
  summarise(compute_ci(dice), .groups = "drop")

df_all <- df_long %>%
  group_by(pathologist) %>%
  summarise(compute_ci(dice), .groups = "drop") %>%
  mutate(pgmn_status = "All")


df_summary <- bind_rows(df_summary, df_all)
df_summary$pgmn_status <- factor(df_summary$pgmn_status, levels = c("All", "Yes", "No"))

fig2c <- ggplot(df_summary, aes(x = pathologist, y = mean_value, fill = pgmn_status, group = pgmn_status)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8),  width = 0.7) +
  geom_errorbar(aes(ymin = lowerCI, ymax = upperCI),
                width = 0.2, position = position_dodge(width = 0.8)) +
  labs(x = NULL, y = "Mean Dice Score (95% CI)", fill = "PGMN Status") +
  theme_classic() +
  scale_y_continuous(breaks = seq(0, 1.2, by = 0.2)) +
  scale_fill_manual(values = c("Yes" = "#7fbf7b", "No" = "#d9f0d3", "All" = "#1b7837"))
print(fig2c) 
ggsave(fig2c, file = "./figure2/fig2c-pgmnEvaldice.pdf",width = 7.5, height = 4, units = "cm")  


####FigS2C, precision
df_long <- path_all %>%
  pivot_longer(cols = c(precision1, precision2),
               names_to = "pathologist",
               values_to = "precision") %>%
  mutate(pathologist = ifelse(pathologist == "precision1", "Pathologist 1", "Pathologist 2"),
         pgmn_status = case_when(
           pathologist == "Pathologist 1" ~ path1_pgmn,
           pathologist == "Pathologist 2" ~ path2_pgmn
         ))


df_summary <- df_long %>%
  group_by(pathologist, pgmn_status) %>%
  summarise(compute_ci(precision), .groups = "drop")

df_all <- df_long %>%
  group_by(pathologist) %>%
  summarise(compute_ci(precision), .groups = "drop") %>%
  mutate(pgmn_status = "All")


df_summary <- bind_rows(df_summary, df_all)
df_summary$pgmn_status <- factor(df_summary$pgmn_status, levels = c("All", "Yes", "No"))

figS2c1 <- ggplot(df_summary, aes(x = pathologist, y = mean_value, fill = pgmn_status, group = pgmn_status)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8),  width = 0.7) +
  geom_errorbar(aes(ymin = lowerCI, ymax = upperCI),
                width = 0.2, position = position_dodge(width = 0.8)) +
  labs(x = NULL, y = "Mean Precision Score (95% CI)", fill = "PGMN Status") +
  theme_classic() +
  scale_y_continuous(breaks = seq(0, 1.2, by = 0.2)) +
  scale_fill_manual(values = c("Yes" = "#d8b365", "No" = "#f6e8c3", "All" = "#8c510a"))
print(figS2c1) 
ggsave(figS2c1, file = "./figure2/figS1c1-pgmnEvalprecision.pdf",width = 7.5, height = 4, units = "cm")  



####FigS2C, recall
df_long <- path_all %>%
  pivot_longer(cols = c(recall1, recall2),
               names_to = "pathologist",
               values_to = "recall") %>%
  mutate(pathologist = ifelse(pathologist == "recall1", "Pathologist 1", "Pathologist 2"),
         pgmn_status = case_when(
           pathologist == "Pathologist 1" ~ path1_pgmn,
           pathologist == "Pathologist 2" ~ path2_pgmn
         ))


df_summary <- df_long %>%
  group_by(pathologist, pgmn_status) %>%
  summarise(compute_ci(recall), .groups = "drop")

df_all <- df_long %>%
  group_by(pathologist) %>%
  summarise(compute_ci(recall), .groups = "drop") %>%
  mutate(pgmn_status = "All")


df_summary <- bind_rows(df_summary, df_all)
df_summary$pgmn_status <- factor(df_summary$pgmn_status, levels = c("All", "Yes", "No"))

figS2c2 <- ggplot(df_summary, aes(x = pathologist, y = mean_value, fill = pgmn_status, group = pgmn_status)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8),  width = 0.7) +
  geom_errorbar(aes(ymin = lowerCI, ymax = upperCI),
                width = 0.2, position = position_dodge(width = 0.8)) +
  labs(x = NULL, y = "Mean Precision Score (95% CI)", fill = "PGMN Status") +
  theme_classic() +
  scale_y_continuous(breaks = seq(0, 1.2, by = 0.2)) +
  scale_fill_manual(values = c("Yes" = "#5ab4ac", "No" = "#c7eae5", "All" = "#01665e"))
print(figS2c2) 
ggsave(figS2c2, file = "./figure2/figS1c2-pgmnEvalrecall.pdf",width = 7.5, height = 4, units = "cm")  



###FigS1B,pie chart for training set
df <- data.frame(
  category = c("PM-positive", "PM-negative"),
  count = c(6619, 2286)
)

figs1b1 <- ggplot(df, aes(x = "", y = count, fill = category)) +
  geom_col(width = 1) +
  coord_polar(theta = "y") +
  geom_text(aes(label = paste0(round(count/sum(count)*100, 1), "%")), 
            position = position_stack(vjust = 0.5)) +
  scale_fill_manual(values = c("#BF9BDDFF", "#F8B150FF")) +
  theme_void()
print(figs1b1)
ggsave(figs1b1, file='./figure2/figS1b1_lucid_train.pdf', width = 7, height = 7, units = "cm") 


###FigS1B, pie charts for nesting set
library(scales)
library(dplyr)
test_data <- tribble(
  ~ring,           ~category,      ~count,
  "path1", "PM-positive",   5560,
  "path1", "PM-negative",   3801,
  "path2", "PM-positive",   4847,
  "path2", "PM-negative",   4514
) %>%
  mutate(
    ring_id  = ifelse(ring == "path1", 1, 2),
    category = factor(category, levels = c("PM-negative", "PM-positive"))
  )

figs1b2 <- ggplot(test_data, aes(x = ring_id, y = count, fill = category)) +
  geom_bar(stat = "identity", width=1, color = "black") +
  coord_polar(theta = "y") +
  geom_text(aes(label = count),
            position = position_stack(vjust = 0.5),
            size = 3.5, fontface = "bold") +
  scale_fill_manual(values = c("PM-positive" = "#F8B150FF", "PM-negative" = "#BF9BDDFF"),
                    name = NULL) +
  theme_void()
print(figs1b2)
ggsave(figs1b2, file='./figure2/figs1b2_lucid_test.pdf', width = 7, height = 7, units = "cm")
