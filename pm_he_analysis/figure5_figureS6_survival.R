library(reshape2)
library(dplyr)
library(tidyr)
library(fst)
library(RColorBrewer)
library(Rtsne)
library(ggplot2)
library(ggbeeswarm)
library(ggbiplot)
library(gtable)
library(cowplot)
library(GGally)
library(rlist)
library(ggrepel)
library(data.table)
library(ggpubr)
library(lmerTest)
library(ggfittext)
library(ggalluvial)
library(survival)
library("survminer")
library("caret") #confusionMatrix
library(pROC) #AUC
library(cowplot) #multi panels within a figure
library(survC1)
library(corrplot)
library(stringr)
library(forestmodel)
library(readxl)
library(purrr)
library(stringr)

setwd('/Volumes/yuan_lab/TIER2/anthracosis/0AI_PM_ST_code/pm_he_analysis')

#==============================================================================
# In-house cohort
#==============================================================================
in_house <- read.csv('../data/in_house_all.csv')
###univariate, continuous, FigS6a
ADC <- in_house %>%
  filter(pgmn_tbed >0) %>%
  filter(rec_time >= 0) 


# features to test
run_uni <- function(dat, smoke_label, resolution) {
  
  features <- c(
    paste0("inflam_mh_tval_ses_", resolution)
  )
  
  map_dfr(features, function(v) {
    
    dat2 <- dat %>%
      filter(
        !is.na(dfs_death_time),
        !is.na(dfs_death_event),
        !is.na(.data[[v]])
      )
    
    form <- as.formula(
      paste0("Surv(dfs_death_time, dfs_death_event) ~ ", v)
    )
    
    fit <- coxph(form, data = dat2)
    s <- summary(fit)
    
    tibble(
      Feature = v,
      N = nrow(dat2),
      Events = sum(dat2$dfs_death_event == 1, na.rm = TRUE),
      HR = s$conf.int[1, "exp(coef)"],
      lower95 = s$conf.int[1, "lower .95"],
      upper95 = s$conf.int[1, "upper .95"],
      p_wald = s$coefficients[1, "Pr(>|z|)"],
      p_lrt = s$logtest["pvalue"],
      p_score = s$sctest["pvalue"],
      Smoke = smoke_label,
      Resolution = resolution
    )
  })
}

uni_res_NS20 <- run_uni(ADC %>% filter(type == "never"), "NS", "20")
uni_res_S20 <- run_uni(ADC %>% filter(type == "smoker"), "S", "20")
uni_res_NS30 <- run_uni(ADC %>% filter(type == "never"), "NS", "30")
uni_res_S30 <- run_uni(ADC %>% filter(type == "smoker"), "S", "30")
uni_all <- bind_rows(uni_res_NS20, uni_res_S20, uni_res_NS30, uni_res_S30)


format_p <- function(p) {
  if (is.na(p)) return(NA_character_)
  if (p >= 0.05) return("NS")
  if (p < 0.001) {
    e <- floor(log10(p))
    m <- round(p / 10^e, 1)
    return(paste0(m, " × 10^", e))
  }
  sprintf("%.3f", p)
}

uni_all <- uni_all %>%
  rowwise() %>%
  mutate(
    p_label = format_p(p_score),
    PlotGroup = paste0(Resolution, ".", Smoke)
  ) %>%
  ungroup()

uni_allre <- uni_all %>%
  filter(
    str_starts(Feature, "inflam_mh_tval_ses"),
    Resolution %in% c("20", "30"),
    Smoke %in% c("NS", "S")
  ) %>%
  mutate(
    PlotGroup = factor(
      PlotGroup,
      levels = c("20.NS", "30.NS", "20.S", "30.S")
    )
  )

p_log <- ggplot(uni_allre, aes(x = PlotGroup, y = HR, color = Smoke)) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "gray") +
  geom_errorbar(
    aes(ymin = lower95, ymax = upper95),
    width = 0.18,
    linewidth = 0.7
  ) +
  geom_point(size = 1, shape = 16) +
  scale_y_log10() +
  scale_color_manual(
    values = c(
      "NS" = "#009E73",
      "S"  = "#E69F00"
    )
  ) +
  labs(
    x = NULL,
    y = "Hazard ratio (log scale)"
  ) +
  theme_classic(base_size = 14) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.background = element_blank(),
    strip.text = element_text(face = "bold"),
    legend.position = "none"
  )

p_log
ggsave("./figure5/figS6a_inhouse_20_30_NS_S.pdf", p_log, width = 4, height = 4,  units = "cm")



#####for never-smoker alone, FigS6h
ADC <- in_house %>%
  filter(pgmn_tbed >0) %>%
  filter(rec_time >= 0) %>%
  filter(type== 'never') %>%
  mutate(pgmn2tumor_perc = pgmn2tumor * 100) %>%
  mutate(inflam_Tper_perc = inflam_Tper * 100) %>%
  mutate(
    inflam_Tper.bin = ntile(inflam_Tper, 2),
    inflam_Tper.bin = factor(
      inflam_Tper.bin,
      levels = c(1, 2),
      labels = c("Low", "High")),
    
    pgmn2tumor.bin = ntile(pgmn2tumor, 2),
    pgmn2tumor.bin = factor(
      pgmn2tumor.bin,
      levels = c(1, 2),
      labels = c("Low", "High")),
    
    inflam_ses_sign_20_0.75 = case_when(
      inflam_mh_tval_ses_20 >quantile(inflam_mh_tval_ses_20, 0.75, na.rm = TRUE) ~ "colocalization",
      inflam_mh_tval_ses_20 <=quantile(inflam_mh_tval_ses_20, 0.75, na.rm = TRUE) ~ "noncolocal",
      TRUE ~ NA_character_
    ),
    inflam_ses_sign_20_0.75 = factor(inflam_ses_sign_20_0.75, levels=c("noncolocal", "colocalization")),
    
    inflam_ses_sign_20_0.5 = case_when(
      inflam_mh_tval_ses_20 >quantile(inflam_mh_tval_ses_20, 0.5, na.rm = TRUE) ~ "colocalization",
      inflam_mh_tval_ses_20 <=quantile(inflam_mh_tval_ses_20, 0.5, na.rm = TRUE) ~ "noncolocal",
      TRUE ~ NA_character_
    ),
    inflam_ses_sign_20_0.5 = factor(inflam_ses_sign_20_0.5, levels=c("noncolocal", "colocalization")),
    
    inflam_ses_sign_30_0.75 = case_when(
      inflam_mh_tval_ses_30 >quantile(inflam_mh_tval_ses_30, 0.75, na.rm = TRUE) ~ "colocalization",
      inflam_mh_tval_ses_30 <=quantile(inflam_mh_tval_ses_30, 0.75, na.rm = TRUE) ~ "noncolocal",
      TRUE ~ NA_character_
    ),
    inflam_ses_sign_30_0.75 = factor(inflam_ses_sign_30_0.75, levels=c("noncolocal", "colocalization")),
    
    inflam_ses_sign_30_0.5 = case_when(
      inflam_mh_tval_ses_30 >quantile(inflam_mh_tval_ses_30, 0.5, na.rm = TRUE) ~ "colocalization",
      inflam_mh_tval_ses_30 <=quantile(inflam_mh_tval_ses_30, 0.5, na.rm = TRUE) ~ "noncolocal",
      TRUE ~ NA_character_
    ),
    inflam_ses_sign_30_0.5 = factor(inflam_ses_sign_30_0.5, levels=c("noncolocal", "colocalization"))
  ) 
ADC_MDA <- ADC[c('patient_id', 'inflam_mh_tval_ses_20', 'inflam_mh_tval_ses_30', 'pgmn2tumor','inflam_Tper','Race_')]

run_uni <- function(dat, smoke_label, resolution) {
  
  features <- c(
    "inflam_Tper.bin",
    "pgmn2tumor.bin",
    paste0("inflam_mh_tval_ses_", resolution)
  )
  
  map_dfr(features, function(v) {
    
    dat2 <- dat %>%
      filter(
        !is.na(dfs_death_time),
        !is.na(dfs_death_event),
        !is.na(.data[[v]])
      )
    
    form <- as.formula(
      paste0("Surv(dfs_death_time, dfs_death_event) ~ ", v)
    )
    
    fit <- coxph(form, data = dat2)
    s <- summary(fit)
    
    tibble(
      Feature = v,
      N = nrow(dat2),
      Events = sum(dat2$dfs_death_event == 1, na.rm = TRUE),
      HR = s$conf.int[1, "exp(coef)"],
      lower95 = s$conf.int[1, "lower .95"],
      upper95 = s$conf.int[1, "upper .95"],
      p_wald = s$coefficients[1, "Pr(>|z|)"],
      p_lrt = s$logtest["pvalue"],
      p_score = s$sctest["pvalue"],
      Smoke = smoke_label,
      Resolution = resolution
    )
  })
}

uni_res_NS20 <- run_uni(ADC %>% filter(type == "never"), "NS", "20")
uni_res_NS30 <- run_uni(ADC %>% filter(type == "never"), "NS", "30")


# format labels
uni_res20_plot <- uni_res_NS20 %>%
  mutate(
    Feature = factor(Feature, levels = rev(Feature)),
    HR_CI = sprintf("%.2f (%.2f–%.2f)", HR, lower95, upper95),
    p_label = case_when(
      is.na(p_score) ~ NA_character_,
      p_score >= 0.05 ~ "NS",
      p_score < 0.001 ~ format(p_score, scientific = TRUE, digits = 2),
      TRUE ~ sprintf("%.3f", p_score)
    )
  ) %>%
  slice(-n())

# forest plot, recommended log scale because CI may be wide
p_log20 <- ggplot(uni_res20_plot, aes(x = HR, y = Feature)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  geom_errorbarh(aes(xmin = lower95, xmax = upper95), height = 0.18, linewidth = 0.7) +
  geom_point(size = 3) +
  scale_x_log10() +
  labs(
    x = "Hazard ratio (univariate Cox, log scale)",
    y = NULL,
    title = "20 µm"
  ) +
  theme_classic(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    axis.text.y = element_text(size = 12)
  )

p_log20
ggsave("./figure5/figS6h_inhouse_MDA_inflam_pgmn2tumor_binary.pdf", p_log20, width = 9, height = 5,  units = "cm")


####KM curves @20um-0.75
pdf('./figure5/fig5e_inhouse_ns_inflam_colocal_ses20_75per.pdf', height = 4, width = 5, onefile = FALSE)
fit <- survfit(Surv(dfs_death_time, dfs_death_event)~ inflam_ses_sign_20_0.75, data =ADC)
p <- ggsurvplot(fit, data = ADC, conf.int = FALSE, 
                pval = TRUE, pval.size = 5, pval.coord = c(0.2, 0.1),
                linetype = "solid",
                surv.plot.height = 0.7, palette = c( "#80A1C1", "#C94277"),
                risk.table = TRUE,
                risk.table.col = "black", break.time.by = 500,
                tables.height = 0.25, 
                tables.theme = theme_cleantable(),
                risk.table.title = "Number at Risk",
                tables.x.text = "none", xlim = c(0, 4200), ylim=c(0,1),
                xlab = "Days to Recurrence", ylab = "Disease-free Survival")
print(p)
dev.off()

p$plot <- p$plot + coord_cartesian(xlim = c(0, 4200))
p$table <- p$table + coord_cartesian(xlim = c(0, 4200))

pdf("./figure5/fig5e_inhouse_ns_inflam_colocal_ses20_75per_4200.pdf", width = 6, height = 5, onefile = FALSE)
print(p)
dev.off()


####KM curves @30um-0.75
pdf('./figure5/fig5e_inhouse_ns_inflam_colocal_ses30_75per.pdf', height = 4, width = 5, onefile = FALSE)
fit <- survfit(Surv(dfs_death_time, dfs_death_event)~ inflam_ses_sign_30_0.75, data =ADC)
p <- ggsurvplot(fit, data = ADC, conf.int = FALSE, 
                pval = TRUE, pval.size = 5, pval.coord = c(0.2, 0.1),
                linetype = "solid",
                surv.plot.height = 0.7, palette = c( "#80A1C1", "#C94277"),
                risk.table = TRUE,
                risk.table.col = "black", break.time.by = 500,
                tables.height = 0.25, 
                tables.theme = theme_cleantable(),
                risk.table.title = "Number at Risk",
                tables.x.text = "none", xlim = c(0, 4200), ylim=c(0,1),
                xlab = "Days to Recurrence", ylab = "Disease-free Survival")
print(p)
dev.off()

p$plot <- p$plot + coord_cartesian(xlim = c(0, 4200))
p$table <- p$table + coord_cartesian(xlim = c(0, 4200))

pdf("./figure5/fig5e_inhouse_ns_inflam_colocal_ses30_75per_4200.pdf", width = 6, height = 5, onefile = FALSE)
print(p)
dev.off()


####KM curves @20um-0.5
pdf('./figure5/fig5e_inhouse_ns_inflam_colocal_ses20_50per.pdf', height = 4, width = 5, onefile = FALSE)
fit <- survfit(Surv(dfs_death_time, dfs_death_event)~ inflam_ses_sign_20_0.5, data =ADC)
p <- ggsurvplot(fit, data = ADC, conf.int = FALSE, 
                pval = TRUE, pval.size = 5, pval.coord = c(0.2, 0.1),
                linetype = "solid",
                surv.plot.height = 0.7, palette = c( "#80A1C1", "#C94277"),
                risk.table = TRUE,
                risk.table.col = "black", break.time.by = 500,
                tables.height = 0.25, 
                tables.theme = theme_cleantable(),
                risk.table.title = "Number at Risk",
                tables.x.text = "none", xlim = c(0, 4200), ylim=c(0,1),
                xlab = "Days to Recurrence", ylab = "Disease-free Survival")
print(p)
dev.off()

p$plot <- p$plot + coord_cartesian(xlim = c(0, 4200))
p$table <- p$table + coord_cartesian(xlim = c(0, 4200))

pdf("./figure5/fig5e_inhouse_ns_inflam_colocal_ses20_50per_4200.pdf", width = 6, height = 5, onefile = FALSE)
print(p)
dev.off()


####KM curves @30um-0.5
pdf('./figure5/fig5e_inhouse_ns_inflam_colocal_ses30_50per.pdf', height = 4, width = 5, onefile = FALSE)
fit <- survfit(Surv(dfs_death_time, dfs_death_event)~ inflam_ses_sign_30_0.5, data =ADC)
p <- ggsurvplot(fit, data = ADC, conf.int = FALSE, 
                pval = TRUE, pval.size = 5, pval.coord = c(0.2, 0.1),
                linetype = "solid",
                surv.plot.height = 0.7, palette = c( "#80A1C1", "#C94277"),
                risk.table = TRUE,
                risk.table.col = "black", break.time.by = 500,
                tables.height = 0.25, 
                tables.theme = theme_cleantable(),
                risk.table.title = "Number at Risk",
                tables.x.text = "none", xlim = c(0, 4200), ylim=c(0,1),
                xlab = "Days to Recurrence", ylab = "Disease-free Survival")
print(p)
dev.off()

p$plot <- p$plot + coord_cartesian(xlim = c(0, 4200))
p$table <- p$table + coord_cartesian(xlim = c(0, 4200))

pdf("./figure5/fig5e_inhouse_ns_inflam_colocal_ses30_50per_4200.pdf", width = 6, height = 5, onefile = FALSE)
print(p)
dev.off()




#==============================================================================
# TCGA cohort
#==============================================================================
tcga <- read.csv('../data/tcga_all.csv')
ADC <- tcga %>%
  filter(pgmn_tbed >0) %>%
  filter(PFI.time >= 0)

  
# features to test
run_uni <- function(dat, smoke_label, resolution) {
  
  features <- c(
    paste0("inflam_mh_tval_ses_", resolution)
  )
  
  map_dfr(features, function(v) {
    
    dat2 <- dat %>%
      filter(
        !is.na(PFI.time),
        !is.na(PFI),
        !is.na(.data[[v]])
      )
    
    form <- as.formula(
      paste0("Surv(PFI.time, PFI) ~ ", v)
    )
    
    fit <- coxph(form, data = dat2)
    s <- summary(fit)
    
    tibble(
      Feature = v,
      N = nrow(dat2),
      Events = sum(dat2$PFI == 1, na.rm = TRUE),
      HR = s$conf.int[1, "exp(coef)"],
      lower95 = s$conf.int[1, "lower .95"],
      upper95 = s$conf.int[1, "upper .95"],
      p_wald = s$coefficients[1, "Pr(>|z|)"],
      p_lrt = s$logtest["pvalue"],
      p_score = s$sctest["pvalue"],
      Smoke = smoke_label,
      Resolution = resolution
    )
  })
}

uni_res_NS20 <- run_uni(ADC %>% filter(type == "never"), "NS", "20")
uni_res_S20 <- run_uni(ADC %>% filter(type == "smoker"), "S", "20")
uni_res_NS30 <- run_uni(ADC %>% filter(type == "never"), "NS", "30")
uni_res_S30 <- run_uni(ADC %>% filter(type == "smoker"), "S", "30")
uni_all <- bind_rows(uni_res_NS20, uni_res_S20, uni_res_NS30, uni_res_S30)


format_p <- function(p) {
  if (is.na(p)) return(NA_character_)
  if (p >= 0.05) return("NS")
  if (p < 0.001) {
    e <- floor(log10(p))
    m <- round(p / 10^e, 1)
    return(paste0(m, " × 10^", e))
  }
  sprintf("%.3f", p)
}

uni_all <- uni_all %>%
  rowwise() %>%
  mutate(
    p_label = format_p(p_score),
    PlotGroup = paste0(Resolution, ".", Smoke)
  ) %>%
  ungroup()

uni_allre <- uni_all %>%
  filter(
    str_starts(Feature, "inflam_mh_tval_ses"),
    Resolution %in% c("20", "30"),
    Smoke %in% c("NS", "S")
  ) %>%
  mutate(
    PlotGroup = factor(
      PlotGroup,
      levels = c("20.NS", "30.NS", "20.S", "30.S")
    )
  )

p_log <- ggplot(uni_allre, aes(x = PlotGroup, y = HR, color = Smoke)) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "gray") +
  geom_errorbar(
    aes(ymin = lower95, ymax = upper95),
    width = 0.18,
    linewidth = 0.7
  ) +
  geom_point(size = 1, shape=16) +
  scale_y_log10() +
  scale_color_manual(
    values = c(
      "NS" = "#009E73",
      "S"  = "#E69F00"
    )
  ) +
  labs(
    x = NULL,
    y = "Hazard ratio (log scale)"
  ) +
  theme_classic(base_size = 14) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.background = element_blank(),
    strip.text = element_text(face = "bold"),
    legend.position = "none"
  )

p_log
ggsave("./figure5/figS6a_tcga_20_30_NS_S.pdf", p_log, width = 4, height = 4,  units = "cm")



#####for never-smoker alone, FigS6h
ADC <- tcga %>%
  filter(pgmn_tbed >0) %>%
  filter(PFI.time >= 0) %>%
  filter(type== 'never') %>%
  mutate(pgmn2tumor_perc = pgmn2tumor * 100) %>%
  mutate(inflam_Tper_perc = inflam_Tper * 100) %>%
  mutate(
    inflam_Tper.bin = ntile(inflam_Tper, 2),
    inflam_Tper.bin = factor(
      inflam_Tper.bin,
      levels = c(1, 2),
      labels = c("Low", "High")),
    
    pgmn2tumor.bin = ntile(pgmn2tumor, 2),
    pgmn2tumor.bin = factor(
      pgmn2tumor.bin,
      levels = c(1, 2),
      labels = c("Low", "High")),
    
    inflam_ses_sign_20_0.75 = case_when(
      inflam_mh_tval_ses_20 >quantile(inflam_mh_tval_ses_20, 0.75, na.rm = TRUE) ~ "colocalization",
      inflam_mh_tval_ses_20 <=quantile(inflam_mh_tval_ses_20, 0.75, na.rm = TRUE) ~ "noncolocal",
      TRUE ~ NA_character_
    ),
    inflam_ses_sign_20_0.75 = factor(inflam_ses_sign_20_0.75, levels=c("noncolocal", "colocalization")),
    
    inflam_ses_sign_20_0.5 = case_when(
      inflam_mh_tval_ses_20 >quantile(inflam_mh_tval_ses_20, 0.5, na.rm = TRUE) ~ "colocalization",
      inflam_mh_tval_ses_20 <=quantile(inflam_mh_tval_ses_20, 0.5, na.rm = TRUE) ~ "noncolocal",
      TRUE ~ NA_character_
    ),
    inflam_ses_sign_20_0.5 = factor(inflam_ses_sign_20_0.5, levels=c("noncolocal", "colocalization")),
    
    inflam_ses_sign_30_0.75 = case_when(
      inflam_mh_tval_ses_30 >quantile(inflam_mh_tval_ses_30, 0.75, na.rm = TRUE) ~ "colocalization",
      inflam_mh_tval_ses_30 <=quantile(inflam_mh_tval_ses_30, 0.75, na.rm = TRUE) ~ "noncolocal",
      TRUE ~ NA_character_
    ),
    inflam_ses_sign_30_0.75 = factor(inflam_ses_sign_30_0.75, levels=c("noncolocal", "colocalization")),
    
    inflam_ses_sign_30_0.5 = case_when(
      inflam_mh_tval_ses_30 >quantile(inflam_mh_tval_ses_30, 0.5, na.rm = TRUE) ~ "colocalization",
      inflam_mh_tval_ses_30 <=quantile(inflam_mh_tval_ses_30, 0.5, na.rm = TRUE) ~ "noncolocal",
      TRUE ~ NA_character_
    ),
    inflam_ses_sign_30_0.5 = factor(inflam_ses_sign_30_0.5, levels=c("noncolocal", "colocalization"))
  ) 


run_uni <- function(dat, smoke_label, resolution) {
  features <- c(
    "inflam_Tper.bin",
    "pgmn2tumor.bin",
    paste0("inflam_mh_tval_ses_", resolution)
  )
  
  map_dfr(features, function(v) {
    
    dat2 <- dat %>%
      filter(
        !is.na(PFI.time),
        !is.na(PFI),
        !is.na(.data[[v]])
      )
    
    form <- as.formula(
      paste0("Surv(PFI.time, PFI) ~ ", v)
    )
    
    fit <- coxph(form, data = dat2)
    s <- summary(fit)
    
    tibble(
      Feature = v,
      N = nrow(dat2),
      Events = sum(dat2$PFI == 1, na.rm = TRUE),
      HR = s$conf.int[1, "exp(coef)"],
      lower95 = s$conf.int[1, "lower .95"],
      upper95 = s$conf.int[1, "upper .95"],
      p_wald = s$coefficients[1, "Pr(>|z|)"],
      p_lrt = s$logtest["pvalue"],
      p_score = s$sctest["pvalue"],
      Smoke = smoke_label,
      Resolution = resolution
    )
  })
}

uni_res_NS20 <- run_uni(ADC %>% filter(type == "never"), "NS", "20")
uni_res_NS30 <- run_uni(ADC %>% filter(type == "never"), "NS", "30")


# format labels
uni_res20_plot <- uni_res_NS20 %>%
  mutate(
    Feature = factor(Feature, levels = rev(Feature)),
    HR_CI = sprintf("%.2f (%.2f–%.2f)", HR, lower95, upper95),
    p_label = case_when(
      is.na(p_score) ~ NA_character_,
      p_score >= 0.05 ~ "NS",
      p_score < 0.001 ~ format(p_score, scientific = TRUE, digits = 2),
      TRUE ~ sprintf("%.3f", p_score)
    )
  ) %>%
  slice(-n())

# forest plot, recommended log scale because CI may be wide
p_log20 <- ggplot(uni_res20_plot, aes(x = HR, y = Feature)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  geom_errorbarh(aes(xmin = lower95, xmax = upper95), height = 0.18, linewidth = 0.7) +
  geom_point(size = 3) +
  scale_x_log10() +
  labs(
    x = "Hazard ratio (univariate Cox, log scale)",
    y = NULL,
    title = "20 µm"
  ) +
  theme_classic(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    axis.text.y = element_text(size = 12)
  )

p_log20
ggsave("./figure5/figS6h_tcga_inflam_pgmn2tumor_binary.pdf", p_log20, width = 9, height = 5,  units = "cm")



####KM curves @20um-0.75
pdf('./figure5/fig5e_tcga_ns_inflam_colocal_ses20_75per.pdf', height = 4, width = 5, onefile = FALSE)
fit <- survfit(Surv(PFI.time, PFI)~ inflam_ses_sign_20_0.75, data =ADC)
p <- ggsurvplot(fit, data = ADC, conf.int = FALSE, 
                pval = TRUE, pval.size = 5, pval.coord = c(0.2, 0.1),
                linetype = "solid",
                surv.plot.height = 0.7, palette = c( "#80A1C1", "#C94277"),
                risk.table = TRUE,
                risk.table.col = "black", break.time.by = 500,
                tables.height = 0.25, 
                tables.theme = theme_cleantable(),
                risk.table.title = "Number at Risk",
                tables.x.text = "none", xlim = c(0, 2500), ylim=c(0,1),
                xlab = "Days to Recurrence", ylab = "Disease-free Survival")
print(p)
dev.off()


####KM curves @30um-0.75
pdf('./figure5/fig5e_tcga_ns_inflam_colocal_ses30_75per.pdf', height = 4, width = 5, onefile = FALSE)
fit <- survfit(Surv(PFI.time, PFI)~ inflam_ses_sign_30_0.75, data =ADC)
p <- ggsurvplot(fit, data = ADC, conf.int = FALSE, 
                pval = TRUE, pval.size = 5, pval.coord = c(0.2, 0.1),
                linetype = "solid",
                surv.plot.height = 0.7, palette = c( "#80A1C1", "#C94277"),
                risk.table = TRUE,
                risk.table.col = "black", break.time.by = 500,
                tables.height = 0.25, 
                tables.theme = theme_cleantable(),
                risk.table.title = "Number at Risk",
                tables.x.text = "none", xlim = c(0, 2500), ylim=c(0,1),
                xlab = "Days to Recurrence", ylab = "Disease-free Survival")
print(p)
dev.off()

####KM curves @20um-0.5
pdf('./figure5/fig5e_tcga_ns_inflam_colocal_ses20_50per.pdf', height = 4, width = 5, onefile = FALSE)
fit <- survfit(Surv(PFI.time, PFI)~ inflam_ses_sign_20_0.5, data =ADC)
p <- ggsurvplot(fit, data = ADC, conf.int = FALSE, 
                pval = TRUE, pval.size = 5, pval.coord = c(0.2, 0.1),
                linetype = "solid",
                surv.plot.height = 0.7, palette = c( "#80A1C1", "#C94277"),
                risk.table = TRUE,
                risk.table.col = "black", break.time.by = 500,
                tables.height = 0.25, 
                tables.theme = theme_cleantable(),
                risk.table.title = "Number at Risk",
                tables.x.text = "none", xlim = c(0, 2500), ylim=c(0,1),
                xlab = "Days to Recurrence", ylab = "Disease-free Survival")
print(p)
dev.off()


####KM curves @30um-0.5
pdf('./figure5/fig5e_tcga_ns_inflam_colocal_ses30_50per.pdf', height = 4, width = 5, onefile = FALSE)
fit <- survfit(Surv(PFI.time, PFI)~ inflam_ses_sign_30_0.5, data =ADC)
p <- ggsurvplot(fit, data = ADC, conf.int = FALSE, 
                pval = TRUE, pval.size = 5, pval.coord = c(0.2, 0.1),
                linetype = "solid",
                surv.plot.height = 0.7, palette = c( "#80A1C1", "#C94277"),
                risk.table = TRUE,
                risk.table.col = "black", break.time.by = 500,
                tables.height = 0.25, 
                tables.theme = theme_cleantable(),
                risk.table.title = "Number at Risk",
                tables.x.text = "none", xlim = c(0, 2500), ylim=c(0,1),
                xlab = "Days to Recurrence", ylab = "Disease-free Survival")
print(p)
dev.off()








#==============================================================================
# cptac cohort
#==============================================================================
cptac <- read.csv('../data/cptac_all.csv')
cptac <- cptac %>%
  mutate(
    recurrence_cc = case_when(
      death_or_recurrence == "Yes" ~ 1,
      death_or_recurrence == "No"  ~ 0,
      death_or_recurrence%in% c("Unknown", "") ~ NA_real_,
      is.na(death_or_recurrence) ~ NA_real_
    )
  ) %>%
  mutate(
    os_cc = case_when(
      vital_status == "Deceased" ~ 1,
      vital_status == "Living"   ~ 0,
      vital_status == ""         ~ NA_real_,
      is.na(vital_status)        ~ NA_real_
    )
  )


###univariate, continuous, FigS6a
ADC <- cptac %>%
  filter(pgmn_tbed >0) 

# features to test
run_uni <- function(dat, smoke_label, resolution) {
  
  features <- c(
    paste0("inflam_mh_tval_ses_", resolution)
  )
  
  map_dfr(features, function(v) {
    
    dat2 <- dat %>%
      filter(
        !is.na(days_to_last_contact_or_death_or_recurrence),
        !is.na(recurrence_cc),
        !is.na(.data[[v]])
      )
    
    form <- as.formula(
      paste0("Surv(days_to_last_contact_or_death_or_recurrence, recurrence_cc) ~ ", v)
    )
    
    fit <- coxph(form, data = dat2)
    s <- summary(fit)
    
    tibble(
      Feature = v,
      N = nrow(dat2),
      Events = sum(dat2$recurrence_cc == 1, na.rm = TRUE),
      HR = s$conf.int[1, "exp(coef)"],
      lower95 = s$conf.int[1, "lower .95"],
      upper95 = s$conf.int[1, "upper .95"],
      p_wald = s$coefficients[1, "Pr(>|z|)"],
      p_lrt = s$logtest["pvalue"],
      p_score = s$sctest["pvalue"],
      Smoke = smoke_label,
      Resolution = resolution
    )
  })
}

uni_res_NS20 <- run_uni(ADC %>% filter(type == "never"), "NS", "20")
uni_res_S20 <- run_uni(ADC %>% filter(type == "smoker"), "S", "20")
uni_res_NS30 <- run_uni(ADC %>% filter(type == "never"), "NS", "30")
uni_res_S30 <- run_uni(ADC %>% filter(type == "smoker"), "S", "30")
uni_all <- bind_rows(uni_res_NS20, uni_res_S20, uni_res_NS30, uni_res_S30)


format_p <- function(p) {
  if (is.na(p)) return(NA_character_)
  if (p >= 0.05) return("NS")
  if (p < 0.001) {
    e <- floor(log10(p))
    m <- round(p / 10^e, 1)
    return(paste0(m, " × 10^", e))
  }
  sprintf("%.3f", p)
}

uni_all <- uni_all %>%
  rowwise() %>%
  mutate(
    p_label = format_p(p_score),
    PlotGroup = paste0(Resolution, ".", Smoke)
  ) %>%
  ungroup()

uni_allre <- uni_all %>%
  filter(
    str_starts(Feature, "inflam_mh_tval_ses"),
    Resolution %in% c("20", "30"),
    Smoke %in% c("NS", "S")
  ) %>%
  mutate(
    PlotGroup = factor(
      PlotGroup,
      levels = c("20.NS", "30.NS", "20.S", "30.S")
    )
  )

p_log <- ggplot(uni_allre, aes(x = PlotGroup, y = HR, color=Smoke)) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "gray") +
  geom_errorbar(
    aes(ymin = lower95, ymax = upper95),
    width = 0.18,
    linewidth = 0.7
  ) +
  geom_point(size = 1, shape = 16) +
  scale_y_log10() +
  scale_color_manual(
    values = c(
      "NS" = "#009E73",
      "S"  = "#E69F00"
    )
  ) +
  labs(
    x = NULL,
    y = "Hazard ratio (log scale)"
  ) +
  theme_classic(base_size = 14) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.background = element_blank(),
    strip.text = element_text(face = "bold"),
    legend.position = "none"
  )

p_log
ggsave("./figure5/figS6a_cptac_20_30_NS_S.pdf", p_log, width = 4, height = 4,  units = "cm")



#####for never-smoker alone, FigS6h
ADC <- cptac %>%
  filter(pgmn_tbed >0) %>%
  filter(type== 'never') %>%
  mutate(pgmn2tumor_perc = pgmn2tumor * 100) %>%
  mutate(inflam_Tper_perc = inflam_Tper * 100) %>%
  mutate(
    inflam_Tper.bin = ntile(inflam_Tper, 2),
    inflam_Tper.bin = factor(
      inflam_Tper.bin,
      levels = c(1, 2),
      labels = c("Low", "High")),
    
    pgmn2tumor.bin = ntile(pgmn2tumor, 2),
    pgmn2tumor.bin = factor(
      pgmn2tumor.bin,
      levels = c(1, 2),
      labels = c("Low", "High")),
    
    inflam_ses_sign_20_0.75 = case_when(
      inflam_mh_tval_ses_20 >quantile(inflam_mh_tval_ses_20, 0.75, na.rm = TRUE) ~ "colocalization",
      inflam_mh_tval_ses_20 <=quantile(inflam_mh_tval_ses_20, 0.75, na.rm = TRUE) ~ "noncolocal",
      TRUE ~ NA_character_
    ),
    inflam_ses_sign_20_0.75 = factor(inflam_ses_sign_20_0.75, levels=c("noncolocal", "colocalization")),
    
    inflam_ses_sign_20_0.5 = case_when(
      inflam_mh_tval_ses_20 >quantile(inflam_mh_tval_ses_20, 0.5, na.rm = TRUE) ~ "colocalization",
      inflam_mh_tval_ses_20 <=quantile(inflam_mh_tval_ses_20, 0.5, na.rm = TRUE) ~ "noncolocal",
      TRUE ~ NA_character_
    ),
    inflam_ses_sign_20_0.5 = factor(inflam_ses_sign_20_0.5, levels=c("noncolocal", "colocalization")),
    
    inflam_ses_sign_30_0.75 = case_when(
      inflam_mh_tval_ses_30 >quantile(inflam_mh_tval_ses_30, 0.75, na.rm = TRUE) ~ "colocalization",
      inflam_mh_tval_ses_30 <=quantile(inflam_mh_tval_ses_30, 0.75, na.rm = TRUE) ~ "noncolocal",
      TRUE ~ NA_character_
    ),
    inflam_ses_sign_30_0.75 = factor(inflam_ses_sign_30_0.75, levels=c("noncolocal", "colocalization")),
    
    inflam_ses_sign_30_0.5 = case_when(
      inflam_mh_tval_ses_30 >quantile(inflam_mh_tval_ses_30, 0.5, na.rm = TRUE) ~ "colocalization",
      inflam_mh_tval_ses_30 <=quantile(inflam_mh_tval_ses_30, 0.5, na.rm = TRUE) ~ "noncolocal",
      TRUE ~ NA_character_
    ),
    inflam_ses_sign_30_0.5 = factor(inflam_ses_sign_30_0.5, levels=c("noncolocal", "colocalization"))
  ) 
ADC_cptac <- ADC[c('patient_id', 'inflam_mh_tval_ses_20', 'inflam_mh_tval_ses_30', 'pgmn2tumor','inflam_Tper','Race_')]

run_uni <- function(dat, smoke_label, resolution) {
  
  features <- c(
    "inflam_Tper.bin",
    "pgmn2tumor.bin",
    paste0("inflam_mh_tval_ses_", resolution)
  )
  
  map_dfr(features, function(v) {
    
    dat2 <- dat %>%
      filter(
        !is.na(days_to_last_contact_or_death_or_recurrence),
        !is.na(recurrence_cc),
        !is.na(.data[[v]])
      )
    
    form <- as.formula(
      paste0("Surv(days_to_last_contact_or_death_or_recurrence, recurrence_cc) ~ ", v)
    )
    
    fit <- coxph(form, data = dat2)
    s <- summary(fit)
    
    tibble(
      Feature = v,
      N = nrow(dat2),
      Events = sum(dat2$recurrence_cc == 1, na.rm = TRUE),
      HR = s$conf.int[1, "exp(coef)"],
      lower95 = s$conf.int[1, "lower .95"],
      upper95 = s$conf.int[1, "upper .95"],
      p_wald = s$coefficients[1, "Pr(>|z|)"],
      p_lrt = s$logtest["pvalue"],
      p_score = s$sctest["pvalue"],
      Smoke = smoke_label,
      Resolution = resolution
    )
  })
}

uni_res_NS20 <- run_uni(ADC %>% filter(type == "never"), "NS", "20")
uni_res_NS30 <- run_uni(ADC %>% filter(type == "never"), "NS", "30")


# format labels
uni_res20_plot <- uni_res_NS20 %>%
  mutate(
    Feature = factor(Feature, levels = rev(Feature)),
    HR_CI = sprintf("%.2f (%.2f–%.2f)", HR, lower95, upper95),
    p_label = case_when(
      is.na(p_score) ~ NA_character_,
      p_score >= 0.05 ~ "NS",
      p_score < 0.001 ~ format(p_score, scientific = TRUE, digits = 2),
      TRUE ~ sprintf("%.3f", p_score)
    )
  ) %>%
  slice(-n())

# forest plot, recommended log scale because CI may be wide
p_log20 <- ggplot(uni_res20_plot, aes(x = HR, y = Feature)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  geom_errorbarh(aes(xmin = lower95, xmax = upper95), height = 0.18, linewidth = 0.7) +
  geom_point(size = 3) +
  scale_x_log10() +
  labs(
    x = "Hazard ratio (univariate Cox, log scale)",
    y = NULL,
    title = "20 µm"
  ) +
  theme_classic(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    axis.text.y = element_text(size = 12)
  )

p_log20
ggsave("./figure5/figS6h_cptac_inflam_pgmn2tumor_binary.pdf", p_log20, width = 9, height = 5,  units = "cm")



####KM curves @20um-0.75
pdf('./figure5/fig5e_cptac_ns_inflam_colocal_ses20_75per.pdf', height = 4, width = 5, onefile = FALSE)
fit <- survfit(Surv(days_to_last_contact_or_death_or_recurrence, recurrence_cc)~ inflam_ses_sign_20_0.75, data =ADC)
p <- ggsurvplot(fit, data = ADC, conf.int = FALSE, 
                pval = TRUE, pval.size = 5, pval.coord = c(0.2, 0.1),
                linetype = "solid",
                surv.plot.height = 0.7, palette = c( "#80A1C1", "#C94277"),
                risk.table = TRUE,
                risk.table.col = "black", break.time.by = 500,
                tables.height = 0.25, 
                tables.theme = theme_cleantable(),
                risk.table.title = "Number at Risk",
                tables.x.text = "none", xlim = c(0, 2000), ylim=c(0,1),
                xlab = "Days to Recurrence", ylab = "Disease-free Survival")
print(p)
dev.off()



####KM curves @30um-0.75
pdf('./figure5/fig5e_cptac_ns_inflam_colocal_ses30_75per.pdf', height = 4, width = 5, onefile = FALSE)
fit <- survfit(Surv(days_to_last_contact_or_death_or_recurrence, recurrence_cc)~ inflam_ses_sign_30_0.75, data =ADC)
p <- ggsurvplot(fit, data = ADC, conf.int = FALSE, 
                pval = TRUE, pval.size = 5, pval.coord = c(0.2, 0.1),
                linetype = "solid",
                surv.plot.height = 0.7, palette = c( "#80A1C1", "#C94277"),
                risk.table = TRUE,
                risk.table.col = "black", break.time.by = 500,
                tables.height = 0.25, 
                tables.theme = theme_cleantable(),
                risk.table.title = "Number at Risk",
                tables.x.text = "none", xlim = c(0, 2000), ylim=c(0,1),
                xlab = "Days to Recurrence", ylab = "Disease-free Survival")
print(p)
dev.off()




####KM curves @20um-0.5
pdf('./figure5/fig5e_cptac_ns_inflam_colocal_ses20_50per.pdf', height = 4, width = 5, onefile = FALSE)
fit <- survfit(Surv(days_to_last_contact_or_death_or_recurrence, recurrence_cc)~ inflam_ses_sign_20_0.5, data =ADC)
p <- ggsurvplot(fit, data = ADC, conf.int = FALSE, 
                pval = TRUE, pval.size = 5, pval.coord = c(0.2, 0.1),
                linetype = "solid",
                surv.plot.height = 0.7, palette = c( "#80A1C1", "#C94277"),
                risk.table = TRUE,
                risk.table.col = "black", break.time.by = 500,
                tables.height = 0.25, 
                tables.theme = theme_cleantable(),
                risk.table.title = "Number at Risk",
                tables.x.text = "none", xlim = c(0, 2000), ylim=c(0,1),
                xlab = "Days to Recurrence", ylab = "Disease-free Survival")
print(p)
dev.off()



####KM curves @30um-0.5
pdf('./figure5/fig5e_cptac_ns_inflam_colocal_ses30_50per.pdf', height = 4, width = 5, onefile = FALSE)
fit <- survfit(Surv(days_to_last_contact_or_death_or_recurrence, recurrence_cc)~ inflam_ses_sign_30_0.5, data =ADC)
p <- ggsurvplot(fit, data = ADC, conf.int = FALSE, 
                pval = TRUE, pval.size = 5, pval.coord = c(0.2, 0.1),
                linetype = "solid",
                surv.plot.height = 0.7, palette = c( "#80A1C1", "#C94277"),
                risk.table = TRUE,
                risk.table.col = "black", break.time.by = 500,
                tables.height = 0.25, 
                tables.theme = theme_cleantable(),
                risk.table.title = "Number at Risk",
                tables.x.text = "none", xlim = c(0, 2000), ylim=c(0,1),
                xlab = "Days to Recurrence", ylab = "Disease-free Survival")
print(p)
dev.off()




###FigS6e-distribution
ADC_cptac$cohort <- 'CPTAC'
ADC_MDA$cohort <- 'In-house'

ADCre <- rbind(ADC_MDA, ADC_cptac)

ADCre$cohort <- factor(
  ADCre$cohort,
  levels = c("In-house", "CPTAC")
)

# color scheme
cohort_cols <- c(
  "In-house" = "#1B9E77",
  "CPTAC"    = "#7570B3"
)

cut_df_long <- ADCre %>%
  group_by(cohort) %>%
  summarise(
    median_cutoff = median(inflam_mh_tval_ses_30, na.rm = TRUE), #to adjust parameter
    q75_cutoff = quantile(inflam_mh_tval_ses_30, 0.75, na.rm = TRUE),  #to adjust parameter
    .groups = "drop"
  ) %>%
  pivot_longer(
    cols = c(median_cutoff, q75_cutoff),
    names_to = "cutoff_type",
    values_to = "cutoff"
  ) %>%
  mutate(
    cutoff_type = factor(
      cutoff_type,
      levels = c("median_cutoff", "q75_cutoff"),
      labels = c("Median", "75th percentile")
    ),
    cutoff_label = paste0(cutoff_type, " = ", round(cutoff, 2))
  )

p_hist <- ggplot(ADCre, aes(x = inflam_mh_tval_ses_30, fill = cohort)) +  #to adjust parameter
  geom_histogram(
    bins = 20,
    color = "black",
    alpha = 0.5
  ) +
  geom_vline(
    data = cut_df_long,
    aes(xintercept = cutoff, linetype = cutoff_type),
    linewidth = 0.8,
    color = "black"
  ) +
  geom_text(
    data = cut_df_long,
    aes(
      x = cutoff,
      y = Inf,
      label = cutoff_label
    ),
    inherit.aes = FALSE,
    angle = 90,
    vjust = -0.4,
    hjust = 1.05,
    size = 3,
    color = "black"
  ) +
  facet_wrap(~ cohort) +
  scale_fill_manual(values = cohort_cols) +
  scale_linetype_manual(
    values = c(
      "Median" = "dashed",
      "75th percentile" = "dotted"
    )
  ) +
  coord_cartesian(clip = "off") +
  theme_test(base_size = 14) +
  labs(
    x="",
    y = "Number of patients",
    fill = NULL,
    linetype = NULL
  ) +
  theme(
    legend.position = "top",
    strip.text = element_text(face = "bold"),
    axis.text = element_text(color = "black"),
    axis.title = element_text(color = "black"),
    #plot.margin = margin(10, 25, 10, 10)
  )

p_hist
ggsave(filename = "./figure5/figS6e_inflam_mh_tval_ses_30.pdf", plot = p_hist, width = 12, height =8,units = "cm")  #to adjust parameter












