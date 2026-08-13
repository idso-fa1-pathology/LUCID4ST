#####anthracosis figures for three cohorts#######
library(reshape2)
library(tibble)
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
library(ggnewscale)  # lets us use two separate fill scales
library(patchwork)  # for combining plots
library(ggtext)        # nice facet labels
library(scales)        # percent formatter
library(ComplexHeatmap)
library(tidyverse)
#if (!require("BiocManager", quietly = TRUE))
#  install.packages("BiocManager")
#BiocManager::install("ComplexHeatmap")

library(circlize) # for custom color palettes
library(ggpattern)    # draws the red diagonal on “NA” legend keys
library(stringr)
library(forestmodel)
library(car) #Anova
library(patchwork)

setwd('/Volumes/yuan_lab/TIER2/anthracosis/0AI_PM_ST_code/pm_he_analysis')
in_house <- read.csv('../data/in_house_all.csv')
tcga <- read.csv('../data/tcga_all.csv')
cptac <-read.csv('../data/cptac_all.csv')

in_house$EGFR[in_house$EGFR=='WT'] <- 'WTe'
in_house$KRAS[in_house$KRAS=='WT'] <- 'WTk'

tcga$EGFR[tcga$EGFR=='none'] <- 'WTe'
tcga$KRAS[tcga$KRAS=='none'] <- 'WTk'

cptac$EGFR[cptac$EGFR=='none'] <- 'WTe'
cptac$KRAS[cptac$KRAS=='none'] <- 'WTk'



##### tile plot for H&E of Fig1c H&E cohort overview

#######to count slides for each patient
in_house_slide <- read.csv('../data/in_house_allslideID_keep.csv')
in_house_slidere <- in_house_slide %>%
  filter(patient_id %in% in_house$patient_id)
in_house_slidere$type[in_house_slidere$type==TRUE] <- 'T'
in_house_slidere$comment <- NULL
in_house_slidere$sk_type <- NULL
length(unique(in_house_slidere$patient_id)) #212

in_house_count <- in_house_slidere %>%
  group_by(patient_id, type) %>%
  summarise(n = n(), .groups = "drop") %>%
  pivot_wider(names_from = type, values_from = n, values_fill = 0) %>%
  rename(n_Tslide = T, n_Lslide = L)

in_house <- in_house %>%
  left_join(in_house_count, by = "patient_id")
in_house$n_total <- in_house$n_Tslide + in_house$n_Lslide
sum(in_house$n_Tslide) #340


tcga_slide <- read.csv('../data/tcga_allslideID_keep.csv')
tcga_slidere <- tcga_slide %>%
  filter(patient_id %in% tcga$patient_id) 
tcga_slidere$type[tcga_slidere$type==TRUE] <- 'T'
tcga_slidere$comment <- NULL
length(unique(tcga_slidere$patient_id)) #384

tcga_count <- tcga_slidere %>%
  group_by(patient_id, type) %>%
  summarise(n = n(), .groups = "drop") %>%
  pivot_wider(names_from = type, values_from = n, values_fill = 0) %>%
  rename(n_Tslide = T)
tcga_count$n_Lslide <- 0

tcga <- tcga %>%
  left_join(tcga_count, by = "patient_id")
tcga$n_total <- tcga$n_Tslide +tcga$n_Lslide
sum(tcga$n_Tslide) #433

cptac_slide <- read.csv('../data/cptac_allslideID_keep.csv')
cptac_slidere <- cptac_slide %>%
  filter(patient_id %in% cptac$patient_id) 
length(unique(cptac_slidere$patient_id)) #187

cptac_count <- cptac_slidere %>%
  group_by(patient_id, type) %>%
  summarise(n = n(), .groups = "drop") %>%
  pivot_wider(names_from = type, values_from = n, values_fill = 0) %>%
  rename(n_Tslide = T, n_Lslide = L)
sum(cptac_count$n_Tslide) #562

cptac <- cptac %>%
  left_join(cptac_count, by = "patient_id")
cptac$n_total <- cptac$n_Tslide +cptac$n_Lslide


in_house_sub <- in_house[c('patient_id','type', 'Age', 'SEX', 'stage', 'Race_', 'EGFR', 'KRAS', 'n_Tslide', 'n_total')]
tcga_sub <- tcga[c('patient_id', 'type', 'Age' ,'stage', 'gender', 'Race_', 'EGFR', 'KRAS', 'n_Tslide', 'n_total')]
cptac_sub <- cptac[c('patient_id', 'type', 'Age',  'stage', 'gender', 'Race_', 'EGFR', 'KRAS', 'n_Tslide', 'n_total')]

in_house_sub <- in_house_sub %>%
  mutate(gender = ifelse(SEX == "M", 'male', 'female'))
in_house_sub$SEX <- NULL
tcga_sub$gender <- tolower(tcga_sub$gender)

in_house_sub$cohort <- 'internal'
tcga_sub$cohort <- 'tcga'
cptac_sub$cohort <- 'cptac'
data <- rbind(in_house_sub, tcga_sub, cptac_sub)
data$type <- factor(data$type, levels = c("never", "smoker"))
data$cohort <- factor(data$cohort, levels = c("internal","tcga", "cptac"))
data$gender <- factor(data$gender)
data$stage <- factor(data$stage)
data$Race_ <- factor(data$Race_)
data$EGFR <- factor(data$EGFR)
data$KRAS <- factor(data$KRAS)
data$Age <- as.numeric(data$Age)

pal_cat <- c("internal" = "#ff0000", "tcga"= "#0000ff", "cptac"= "#ffff00",
             "never" = "#009E73", "smoker" = "#E69F00", "female"="#FBB4AE", "male"= "#B3CDE3",
             "I"= "#CBC9E2", "II"= "#9E9AC8", "III"= "#6A51A3", "IV"="#54278F",
             "African American"="#6495ED", "Asian"= "#de9ed6", "White"="#fd8d3c", "Hispanic"="#8dd3c7", "WOAmerican Indian or Alaska Native"="#8dd3c7",
             "WOther"="#8dd3c7",
             ## EGFR / KRAS status
             "EGFR"="#33A02C", "WTe"= "#C7E9C0", "KRAS"="#ffd92f", "WTk"="#FFFF99",
             "T" = "#BC6892", "L" = "#DFABC9",  # T and L slide tiles
             "NA" = "grey"
)

age_pal <- rev(colorRampPalette(brewer.pal(8, "RdYlBu"))(100))

data <- data %>%
  arrange(cohort, type, gender, stage, Race_, EGFR, KRAS) %>%           # << order tumours (Smoker first)
  mutate(x = row_number())


#three cohorts together- must use this to ensure three cohorts can share the same color legend
row_order <- c("cohort", "type", "Age", "gender", "stage", "Race_", "EGFR", "KRAS")
lab_vec <- c(
  cohort ="Cohort", type = "Type", Age = "Age", gender = "Sex", stage = "Stage",
  Race_ = "Race", EGFR = "EGFR", KRAS = "KRAS"
)

long <- data %>%
  mutate(across(c(cohort, type, gender, stage, Race_, EGFR, KRAS, Age), as.character)) %>%
  pivot_longer(
    cols      = c(cohort, type, gender, stage, Age, Race_, EGFR, KRAS),
    names_to  = "variable",
    values_to = "value"
  ) %>%
  mutate(
    value = ifelse(variable == "Age",
                   as.numeric(value),   # "NA" → NA_real_
                   value))


## split numeric-vs-categorical for two fill scales
cat_long  <- long %>% filter(variable != "Age")
age_long  <- long %>% filter(variable == "Age") %>%
  mutate(value = as.numeric(value))
range(age_long$value, na.rm = TRUE) #35, 87
#without #slides
fig1b <- ggplot() +
  geom_tile(
    data   = cat_long,
    aes(x = x,
        y = factor(variable, levels = row_order),
        fill = value),
    colour = "white", linewidth = 0.08, height = 0.9
  ) +
  scale_fill_manual(
    values      = pal_cat,      # your named palette
    drop        = FALSE,
    na.value    = "grey",     # colour used for true NA entries
    name        = NULL
  ) +
  
  ggnewscale::new_scale_fill() +
  geom_tile(
    data = age_long,
    aes(x = x, y = factor(variable, levels = row_order), fill = value),
    colour = "white", linewidth = 0.08, height = 0.9
  ) +
  scale_fill_gradientn(
    colours = age_pal,
    name    = "Age",
    na.value = "grey"   # colour for missing ages, if any
  ) +
  scale_y_discrete(
    limits = rev(row_order),          # ensures the order you specified
    labels = lab_vec[row_order]       # applies the readable labels
  ) +
  theme_void(base_size = 6) +
  theme(
    axis.text.y     = element_text(hjust = 0),
    legend.position = "none"
  )
print(fig1b)


p_bar <- ggplot(data, aes(x = x)) +  # use same `x` as fig1b
  geom_col(aes(y = n_total),  fill = "#DFABC9", linewidth = 0.08, width = 0.9, position = "identity") +
  geom_col(aes(y = n_Tslide), fill = "#BC6892", linewidth = 0.08, width = 0.9, position = "identity") +
  labs(y = NULL, x = NULL) +
  theme_void(base_size = 6) +
  theme(
    axis.text.y  = element_text(size = 6),         # show y-axis text
    axis.ticks.y = element_line(size = 0.2),       # show ticks
    axis.title.y       = element_text(size = 6),
    plot.margin        = margin(0, 5, 0, 5),
    legend.position = "none"
  )
print(p_bar)
combined_plot <- fig1b / p_bar + plot_layout(heights = c(8, 1))  # adjust ratio as needed
print(combined_plot)
ggsave(combined_plot, file = "./figure1/fig1c_3Adeno.pdf",width = 36, height = 9, units = "cm")



# fig1c precancer cohort
#######to count slides for each patient
load('../data/MP_precancer_Aug23.RData')
compre <- comp[c('Stages', 'PatientID')]
#compre$Stages <- factor(compre$Stages, levels = c("ADC", "MIA", "AIS", "AAH", "Normal"))
precancer_count <- compre %>%
  group_by(PatientID, Stages) %>%
  summarise(n = n(), .groups = "drop") %>%
  pivot_wider(names_from = Stages, values_from = n, values_fill = 0) %>%
  dplyr::select(PatientID, ADC, MIA, AIS, AAH, Normal)

precancer_count$total <- precancer_count$ADC + precancer_count$MIA + precancer_count$AIS + precancer_count$AAH + precancer_count$Normal
precancer <- clinic %>%
  left_join(precancer_count, by = "PatientID")

precancer$type <- 'smoker'
precancer$type[precancer$SmokerType=='Never'] <- 'never'
precancer$n_ADC <- precancer$ADC
precancer$n_MIA <- precancer$ADC + precancer$MIA
precancer$n_AIS <- precancer$ADC + precancer$MIA + precancer$AIS
precancer$n_AAH <- precancer$ADC + precancer$MIA + precancer$AIS ++ precancer$AAH
precancer$n_Normal <- precancer$ADC + precancer$MIA + precancer$AIS ++ precancer$AAH+ precancer$Normal
precancer_re <- precancer[c('PatientID', 'Age', 'Gender', 'type', 'Race', "ADC", "MIA", "AIS", "AAH", "Normal")]
precancer_re$type <- factor(precancer_re$type, levels = c("never", "smoker"))
precancer_re$Gender <- factor(precancer_re$Gender)
precancer_re$Race <- factor(precancer_re$Race)

pal_cat <- c("never" = "#009E73", "smoker" = "#E69F00", "Female"="#FBB4AE", "Male"= "#B3CDE3",
             "African American"="#6495ED", "Asian"= "#de9ed6", "White"="#fd8d3c", "Hispanic"="#8dd3c7", "WOAmerican Indian or Alaska Native"="#8dd3c7",
             "WOther"="#8dd3c7",
             "n_ADC" = "#BC6892", "n_MIA"="#172869", "n_AIS"="#088BBE", "n_AAH"="#1BB6AF" ,"n_Normal" = "#DFABC9",  # T and L slide tiles
             "NA" = "grey"
)

age_pal <- rev(colorRampPalette(brewer.pal(8, "RdYlBu"))(100))

precancer_re <- precancer_re %>%
  arrange(type, Gender, Race) %>%           # << order tumours (Smoker first)
  mutate(x = row_number())

row_order <- c("type", "Age", "Gender", "Race")
lab_vec <- c(
  type = "Type", Age = "Age", Gender = "Sex", Race = "Race"
)

long <- precancer_re %>%
  mutate(across(c(type, Gender,  Race, Age), as.character)) %>%
  pivot_longer(
    cols      = c(type, Gender, Age, Race),
    names_to  = "variable",
    values_to = "value"
  ) %>%
  mutate(
    value = ifelse(variable == "Age",
                   as.numeric(value),   # "NA" → NA_real_
                   value))


## split numeric-vs-categorical for two fill scales
cat_long  <- long %>% filter(variable != "Age")
age_long  <- long %>% filter(variable == "Age") %>%
  mutate(value = as.numeric(value))
range(age_long$value, na.rm = TRUE) #
#without #slides
fig1b <- ggplot() +
  geom_tile(
    data   = cat_long,
    aes(x = x,
        y = factor(variable, levels = row_order),
        fill = value),
    colour = "white", linewidth = 0.08, height = 0.9
  ) +
  scale_fill_manual(
    values      = pal_cat,      # your named palette
    drop        = FALSE,
    na.value    = "grey",     # colour used for true NA entries
    name        = NULL
  ) +
  
  ggnewscale::new_scale_fill() +
  geom_tile(
    data = age_long,
    aes(x = x, y = factor(variable, levels = row_order), fill = value),
    colour = "white", linewidth = 0.08, height = 0.9
  ) +
  scale_fill_gradientn(
    colours = age_pal,
    name    = "Age",
    limits  = c(35, 87),
    na.value = "grey"   # colour for missing ages, if any
  ) +
  scale_y_discrete(
    limits = rev(row_order),          # ensures the order you specified
    labels = lab_vec[row_order]       # applies the readable labels
  ) +
  theme_void(base_size = 6) +
  theme(
    axis.text.y     = element_text(hjust = 0),
    legend.position = "none"
  )
print(fig1b)

#bar plot
lesion_order <- c("Normal", "AAH", "AIS", "MIA", "ADC")
pal_roi <- c(
  "ADC"   = "#BC6892",
  "MIA"   = "#172869",
  "AIS"   = "#088BBE",
  "AAH"   = "#1BB6AF",
  "Normal"= "#DFABC9"
)

roi_long <- precancer_re %>%
  dplyr::select(x, PatientID, all_of(lesion_order)) %>%
  replace_na(as.list(setNames(rep(0, length(lesion_order)), lesion_order))) %>%
  pivot_longer(cols = all_of(lesion_order),
               names_to = "lesion", values_to = "n") %>%
  mutate(
    lesion = factor(lesion, levels = lesion_order),
    n = as.numeric(n)
  )

p_bar <- ggplot(roi_long, aes(x = x, y = n, fill = lesion)) +
  geom_col(width = 0.9) +
  scale_fill_manual(values = pal_roi, breaks = lesion_order) +
  scale_y_continuous(expand = c(0,0)) +
  labs(x = NULL, y = "#ROIs", fill = NULL) +
  theme_void(base_size = 6) +
  theme(
    legend.position = "none",
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.text.y  = element_text(size = 6),         # show y-axis text
    axis.ticks.y = element_line(size = 0.5),       # show ticks
    plot.margin = margin(t = 2, r = 2, b = 2, l = 2)
  )
print(p_bar)

combined_plot <- fig1b / p_bar + plot_layout(heights = c(4, 1))  # adjust ratio as needed
print(combined_plot)
ggsave(combined_plot, file = "./figure1/fig1c_precancer.pdf",width = 5, height = 5, units = "cm")



###fig4a tile plot for ST data
###MDA1 ST
MDA1 <- read.csv('../data/MDA1st_clinical.csv')

###MDA2 ST
MDA2 <- read.csv('../data/MDA2st_clinical.csv')

data_st <- rbind(MDA1, MDA2)

pal_cat <- c("MDA1" = "#FDDFA4", "MDA2"= "#2c3778", 
             "never" = "#009E73", "smoker" = "#E69F00", "F"="#FBB4AE", "M"= "#B3CDE3",
             "I"= "#CBC9E2", "II"= "#9E9AC8", "III"= "#6A51A3", "IV"="#54278F",
             "African American"="#6495ED", "Asian"= "#de9ed6", "Caucasian"="#fd8d3c", "Hispanic"="#8dd3c7", "WOAmerican Indian or Alaska Native"="#8dd3c7",
             "WOther"="#8dd3c7",
             ## EGFR / KRAS status
             "EGFR"="#33A02C", "WTe"= "#C7E9C0", "KRAS"="#ffd92f", "WTk"="#FFFF99",
             ## placeholder for missing values – real colour hidden by stripe pattern
             "NA" = "grey"
)

age_pal <- rev(colorRampPalette(brewer.pal(8, "RdYlBu"))(100))

data_st$set <- factor(data_st$set, levels = c("MDA2", "MDA1"))
data_st$type <- factor(data_st$type, levels = c("smoker", "never"))
data_st$patient_id <- factor(data_st$patient_id)
data_st$SEX <- factor(data_st$SEX)
data_st$stage <- factor(data_st$stage)
data_st$EGFR <- factor(data_st$EGFR)
data_st$KRAS <- factor(data_st$KRAS)

data_st <- data_st %>% 
  arrange(set, type, SEX, stage, EGFR, KRAS) %>% 
  mutate(patient_id = factor(patient_id, levels = patient_id))

cat_vars  <- c("set","type","SEX","stage", "EGFR", "KRAS") 
long_cat  <- data_st %>%
  pivot_longer(all_of(cat_vars),
               names_to = "variable", values_to = "value") %>%
  mutate(variable = factor(variable,
                           levels = c('set',"type", "Age", "SEX","stage", "EGFR", "KRAS"))) 


p_tiles <- ggplot() +
  geom_tile(data = long_cat,
            aes(x = variable, y = patient_id, fill = value),
            colour = "white", width = 1, height = 0.8) +
  scale_fill_manual(values = pal_cat, na.value = "grey", name = NULL) +

  ggnewscale::new_scale_fill() +
  geom_tile(data = data_st,
            aes(x = "Age", y = patient_id, fill = Age),
            colour = "white", width = 1, height = 0.8) +
  scale_fill_gradientn(colours = age_pal, 
                       name = "Age") +
  
  scale_x_discrete(position = "top",
                   limits   = c("set","type","Age", "SEX","stage",
                                "EGFR", "KRAS"), 
                   labels = c("Set","Type","Age", "Sex","TNM stage",
                              "EGFR", "KRAS")) +
  theme_minimal(base_size = 7) +
  theme(
    axis.title  = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 0),
    axis.text.y = element_text(size = 7),
    panel.grid  = element_blank()
  )
print(p_tiles)

p_bar <- ggplot(data_st, aes(y = patient_id)) +
  geom_col(aes(x = spot),  fill = "#DFABC9", width = 0.8) +  # total spots
  geom_col(aes(x = pgmn_spot),   fill =  "#BC6892", width = 0.8) + # anthracosis
  labs(x = "#spots", y = NULL) +
  theme_minimal(base_size = 7) +
  theme(
    axis.text.y        = element_blank(),
    axis.ticks.y       = element_blank(),
    panel.grid.major.y = element_blank()
  )
print(p_bar)

fig4a <- (p_tiles + p_bar) + plot_layout(widths = c(1, 1))
print(fig4a)
ggsave(fig4a, file = "./figure4/fig4a-ST-sepCC_legend.pdf",width = 7, height = 16, units = "cm")



####Fig2e and FigS1d
in_house <- read.csv('../data/in_house_all.csv')
tcga <- read.csv('../data/tcga_all.csv')
cptac <-read.csv('../data/cptac_all.csv')

in_house_sub <- in_house[c('pgmnper', 'pgmn2tumor', 'pgmn2alveoli', 'type')]
tcga_sub <- tcga[c('pgmnper', 'pgmn2tumor', 'pgmn2alveoli','type')]
cptac_sub <- cptac[c('pgmnper', 'pgmn2tumor', 'pgmn2alveoli', 'type')]

in_house_sub$cohort <- 'internal'
tcga_sub$cohort <- 'tcga'
cptac_sub$cohort <- 'cptac'
data <- rbind(in_house_sub, tcga_sub, cptac_sub)

data$cohort <- factor(data$cohort, levels = c('internal', 'tcga', 'cptac'))
data_long <- gather(data, key = "variable", value = "value", pgmn2tumor, pgmn2alveoli)

data_long$variable <- factor(data_long$variable, levels = c('pgmn2tumor', 'pgmn2alveoli'))
data_long$cohort <- factor(data_long$cohort, levels = c('internal', 'tcga', 'cptac'))


fig2e <- ggplot(data_long, aes(x = cohort, y = value, fill = type)) +
  geom_violin(alpha = 0.5, position = position_dodge(width = 0.8)) +
  geom_boxplot(width = 0.1,  outlier.shape = NA, position = position_dodge(width = 0.8)) +
  scale_fill_manual(values = c("never" = "#009E73", "smoker" = "#E69F00")) +
  facet_wrap(~ variable) +
  scale_y_sqrt() +
  labs(x = "Cohort", y = "sqrt(LPI)", fill = "Group") +
  #ylim(0, 0.05)+
  theme_minimal() +
  theme(legend.position = "top") +
  stat_compare_means(aes(group = type))
print(fig2e)
ggsave(fig2e, file = "./figure2/fig2e-NS-S.pdf",width = 18, height = 10, units = "cm")

figS1d <- ggplot(data_long, aes(x = type, y = value, fill = cohort)) +
  geom_violin(alpha = 0.5, position = position_dodge(width = 0.8)) +  # Violin plot with dodge
  geom_boxplot(width = 0.1,  outlier.shape = NA, position = position_dodge(width = 0.8)) +
  scale_fill_brewer(palette='Dark2') +
  facet_wrap(~ variable) +  # Facet by variable (var1, var2)
  scale_y_sqrt() +
  labs(x = "Cohort", y = "sqrt(LPI)", fill = "Group") +
  #ylim(0, 0.01)+
  theme_minimal() +
  theme(legend.position = "top")+
  stat_compare_means(aes(group = cohort)) 
print(figS1d)
ggsave(figS1d, file = "./figure2/figS1d-cohort.pdf",width = 18, height = 10, units = "cm")



####Fig3c, correlation heatmap
in_house_sub <- in_house[c( 'pgmn2tumor', 'tumor_Tper', 'stroma_Tper','inflam_Tper', 'type')]
tcga_sub <- tcga[c( 'pgmn2tumor', 'tumor_Tper', 'stroma_Tper','inflam_Tper','type')]
cptac_sub <- cptac[c( 'pgmn2tumor',  'tumor_Tper', 'stroma_Tper','inflam_Tper', 'type')]

in_house_sub$cohort <- 'In-house'
tcga_sub$cohort <- 'TCGA'
cptac_sub$cohort <- 'CPTAC'

df <- rbind(in_house_sub, tcga_sub, cptac_sub)
df[c('type')] <- NULL

marker_map <- c(
  "Tumor"  = "tumor_Tper",
  "Stroma" = "stroma_Tper",
  "Inflam" = "inflam_Tper"
)

df <- df %>%
  mutate(cohort = factor(cohort, levels = c("In-house", "TCGA", "CPTAC")))

long <- df %>%
  pivot_longer(cols = all_of(unname(marker_map)),
               names_to = "marker_col", values_to = "value") %>%
  mutate(marker = names(marker_map)[match(marker_col, marker_map)])

res <- long %>%
  group_by(cohort, marker) %>%
  summarise(
    n = sum(complete.cases(pgmn2tumor, value)),
    rho = suppressWarnings(cor(pgmn2tumor, value, method = "spearman", use = "complete.obs")),
    p   = {
      ct <- suppressWarnings(cor.test(pgmn2tumor, value, method = "spearman", exact = FALSE))
      ct$p.value
    },
    .groups = "drop"
  )

# ---- Significance stars ----
res0 <- res %>%
  mutate(
    stars = case_when(
      is.na(p)            ~ "",
      p < 1e-4            ~ "****",
      p < 1e-3            ~ "***",
      p < 1e-2            ~ "**",
      p < 5e-2            ~ "*",
      TRUE                ~ ""
    ),
    marker = factor(marker, levels = c("Tumor", "Stroma", "Inflam"))
  ) #non-adjust

res <- res %>%
  group_by(cohort) %>%   # adjust within each cohort
  mutate(p_adj = p.adjust(p, method = "BH")) %>%
  ungroup() %>%
  mutate(
    stars = case_when(
      is.na(p_adj)        ~ "",
      p_adj < 1e-4        ~ "****",
      p_adj < 1e-3        ~ "***",
      p_adj < 1e-2        ~ "**",
      p_adj < 0.05        ~ "*",
      TRUE                ~ ""
    ),
    marker = factor(marker, levels = c( "Inflam", "Stroma","Tumor"))
  ) #adjust within each cohort

p_heat <- ggplot(res, aes(x = cohort, y = marker, fill = rho)) +
  geom_tile(color = "black", linewidth = 0.5) +
  # significance stars on top
  geom_text(aes(label = stars),  size = 6) +
  # (optional) show ρ as smaller text under the stars
  geom_text(aes(label = sprintf("%.2f", rho)), vjust = 2.0, size = 3.2, color = "black") +
  scale_fill_gradient2(
    low = "#2166ac", mid = "white", high = "#b2182b",
    limits = c(-1, 1), midpoint = 0, name = "Spearman \u03C1"
  ) +
  labs(x = NULL, y = NULL) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid = element_blank(),
    axis.text.x = element_text(angle = 0, hjust = 0.5),
    axis.text.y = element_text()
  )

p_heat
ggsave("./figure3/fig3c_corr_heatmap.pdf", plot = p_heat, width = 9, height = 7, units = "cm")



###tme% around pigment, Figure 3e
tissue_colors <- c(tumor        = "#843C39",
                   stroma       = "#fdae61",
                   inflammatory = "#CC79A7",
                   macrophage   = "#756BB1",
                   alveoli      = "#1a9850",
                   bronchi      = "#9EDAE5",
                   microvessel  = "cornflowerblue",
                   necrosis     = "#F0E442",
                   adipose      = "#8CA252",
                   muscle       = "darkblue")


###in_house
tme_pgmn_T_inhouse <- read.csv('../data/in_house_pm_tme.csv')

df <- tme_pgmn_T_inhouse [, c(3: 12)]  # where tissue_types is your 10 columns
tissue_avg <- colMeans(df) |> 
  as.data.frame() |> 
  rownames_to_column("tissue_type") 
colnames(tissue_avg)[2] <- "average_per"

tissue_avg <- tissue_avg %>%
  arrange(desc(average_per)) %>%
  mutate(
    label = paste0(round(average_per, 1), "%")
  )

tissue_type_order <- tissue_avg %>% 
  arrange(desc(average_per)) %>%
  pull(tissue_type)

tissue_avg$tissue_type <- factor(tissue_avg$tissue_type, levels = tissue_type_order)


fig3e1 <- ggplot(tissue_avg, aes(x = "", y = average_per, fill = tissue_type)) +
  geom_bar(stat = "identity") +
  coord_polar(theta = "y") +
  geom_text(aes(label = label),
            position = position_stack(vjust = 0.5),
            size = 3) +
  scale_fill_manual(values = tissue_colors) +
  theme_void() +
  theme(legend.position = 'none')
print(fig3e1)
ggsave(fig3e1, file='./figure3/fig3e_pgmn_neighbour_tmeper_Pie_inhouse.pdf', width = 5, height = 5, units = "cm") 


###statistical comparison with global tme per, Figure 3f
tme_pgmn_global_inhouse <- read.csv('../data/in_house_tme_pgmn20_global.csv')

cell_map <- c(
  tumor = "Tumor",       tumor_per = "Tumor",
  stroma = "Stroma",     stroma_per = "Stroma",
  inflammatory = "Inflammation",
  inflammation = "Inflammation",   # add if present
  inflam = "Inflammation",         # add if present
  inflam_per = "Inflammation",
  macrophage = "Macrophage", macro_per = "Macrophage"
)

long <- tme_pgmn_global_inhouse %>%
  dplyr::select(ID,
                tumor, tumor_per,
                stroma, stroma_per,
                inflammatory, inflam_per) %>%
  pivot_longer(
    cols = -ID,
    names_to = "var",
    values_to = "value"
  ) %>%
  mutate(
    cell = dplyr::recode(var, !!!cell_map),
    type = ifelse(str_detect(var, "_per$") | var == "macro_per",
                  "Global", "PM")
  ) %>%
  dplyr::select(ID, cell, type, value)

fig3f <- ggplot(long, aes(x = type, y = value, fill = type)) +
  geom_boxplot(width = 0.5, outlier.shape = NA, alpha = 0.8, color = "black") +
  geom_line(aes(group = ID), alpha = 0.5, color = "gray") +
  geom_point(size = 0.2, color = "black", alpha = 0.5,position = position_jitter(width = 0.05, seed = 1)) +
  facet_wrap(~ cell, scales = "free_y") +
  scale_fill_manual(values = c("PM" = "#F8B150FF", "Global" = "#BF9BDDFF")) +
  labs(
    x = NULL,
    y = "Tissue percentage"
  ) +
  theme_bw(base_size = 6) +
  theme(
    legend.position = "top",
    panel.grid.minor = element_blank()
  ) +
  stat_compare_means(
    aes(group = type),
    method = "wilcox.test",
    paired = TRUE,
    label = "p.format"
  )
print(fig3f)
ggsave(fig3f, file='./figure3/fig3f_pgmn_neighbour20_global_in_house.pdf', width = 8, height = 5, units = "cm")



###TCGA
tme_pgmn_T_tcga <- read.csv('/Volumes/yuan_lab/TIER2/anthracosis/0AI_PM_ST_code/data/tcga_pm_tme.csv')
df <- tme_pgmn_T_tcga[, c(3: 12)]  # where tissue_types is your 10 columns

tissue_avg <- colMeans(df) |> 
  as.data.frame() |> 
  rownames_to_column("tissue_type") 
colnames(tissue_avg)[2] <- "average_per"

tissue_avg <- tissue_avg %>%
  arrange(desc(average_per)) %>%
  mutate(
    label = paste0(round(average_per, 1), "%")
  )

tissue_type_order <- tissue_avg %>% 
  arrange(desc(average_per)) %>%
  pull(tissue_type)

tissue_avg$tissue_type <- factor(tissue_avg$tissue_type, levels = tissue_type_order)

fig3e2 <- ggplot(tissue_avg, aes(x = "", y = average_per, fill = tissue_type)) +
  geom_bar(stat = "identity") +
  coord_polar(theta = "y") +
  geom_text(aes(label = label),
            position = position_stack(vjust = 0.5),
            size = 3) +
  scale_fill_manual(values = tissue_colors) +
  theme_void() +
  theme(legend.position = 'none')
print(fig3e2)
ggsave(fig3e2, file='./figure3/fig2e_pgmn_neighbour20_tmeper_avgPie_tcga.pdf', width = 5, height = 5, units = "cm")


###statistical comparison with global tme per
tme_pgmn_global_tcga <- read.csv('../data/tcga_tme_pgmn20_global.csv')
cell_map <- c(
  tumor = "Tumor",       tumor_per = "Tumor",
  stroma = "Stroma",     stroma_per = "Stroma",
  inflammatory = "Inflammation",
  inflammation = "Inflammation",   # add if present
  inflam = "Inflammation",         # add if present
  inflam_per = "Inflammation",
  macrophage = "Macrophage", macro_per = "Macrophage"
)

long <- tme_pgmn_global_tcga %>%
  dplyr::select(ID,
                tumor, tumor_per,
                stroma, stroma_per,
                inflammatory, inflam_per) %>%
  pivot_longer(
    cols = -ID,
    names_to = "var",
    values_to = "value"
  ) %>%
  mutate(
    cell = dplyr::recode(var, !!!cell_map),
    type = ifelse(str_detect(var, "_per$") | var == "macro_per",
                  "Global", "PM")
  ) %>%
  dplyr::select(ID, cell, type, value)

fig3g <- ggplot(long, aes(x = type, y = value, fill = type)) +
  geom_boxplot(width = 0.5, outlier.shape = NA, alpha = 0.8, color = "black") +
  geom_line(aes(group = ID), alpha = 0.5, color = "gray") +
  geom_point(size = 0.2, color = "black", alpha = 0.5,position = position_jitter(width = 0.05, seed = 1)) +
  facet_wrap(~ cell, scales = "free_y") +
  scale_fill_manual(values = c("PM" = "#F8B150FF", "Global" = "#BF9BDDFF")) +
  labs(
    x = NULL,
    y = "Tissue percentage"
  ) +
  theme_bw(base_size = 6) +
  theme(
    legend.position = "top",
    panel.grid.minor = element_blank()
  ) +
  stat_compare_means(
    aes(group = type),
    method = "wilcox.test",
    paired = TRUE,
    label = "p.format"
  )
print(fig3g)
ggsave(fig3g, file='./figure3/fig3g_pgmn_neighbour20_global_box_TCGA.pdf', width = 8, height = 5, units = "cm")




###CPTAC
tme_pgmn_T_cptac <- read.csv('/Volumes/yuan_lab/TIER2/anthracosis/0AI_PM_ST_code/data/cptac_pm_tme.csv')
df <- tme_pgmn_T_cptac[, c(3: 12)]  # where tissue_types is your 10 columns
tissue_avg <- colMeans(df) |> 
  as.data.frame() |> 
  rownames_to_column("tissue_type") 
colnames(tissue_avg)[2] <- "average_per"

tissue_avg <- tissue_avg %>%
  arrange(desc(average_per)) %>%
  mutate(
    label = paste0(round(average_per, 1), "%")
  )

tissue_type_order <- tissue_avg %>% 
  arrange(desc(average_per)) %>%
  pull(tissue_type)

tissue_avg$tissue_type <- factor(tissue_avg$tissue_type, levels = tissue_type_order)

fig3e3 <- ggplot(tissue_avg, aes(x = "", y = average_per, fill = tissue_type)) +
  geom_bar(stat = "identity") +
  coord_polar(theta = "y") +
  geom_text(aes(label = label),
            position = position_stack(vjust = 0.5),
            size = 3) +
  scale_fill_manual(values = tissue_colors) +
  theme_void() +
  theme(legend.position = 'none')
print(fig3e3)
ggsave(fig3e3, file='./figure3/fig34_pgmn_neighbour20_tmeper_avgPie_cptac.pdf', width = 5, height = 5, units = "cm")


###statistical comparison with global tme per
tme_pgmn_global_cptac <- read.csv('../data/cptac_tme_pgmn20_global.csv')
cell_map <- c(
  tumor = "Tumor",       tumor_per = "Tumor",
  stroma = "Stroma",     stroma_per = "Stroma",
  inflammatory = "Inflammation",
  inflammation = "Inflammation",   # add if present
  inflam = "Inflammation",         # add if present
  inflam_per = "Inflammation",
  macrophage = "Macrophage", macro_per = "Macrophage"
)

long <- tme_pgmn_global_cptac %>%
  dplyr::select(ID,
                tumor, tumor_per,
                stroma, stroma_per,
                inflammatory, inflam_per) %>%
  pivot_longer(
    cols = -ID,
    names_to = "var",
    values_to = "value"
  ) %>%
  mutate(
    cell = dplyr::recode(var, !!!cell_map),
    type = ifelse(str_detect(var, "_per$") | var == "macro_per",
                  "Global", "PM")
  ) %>%
  dplyr::select(ID, cell, type, value)

fig3h <- ggplot(long, aes(x = type, y = value, fill = type)) +
  geom_boxplot(width = 0.5, outlier.shape = NA, alpha = 0.8, color = "black") +
  geom_line(aes(group = ID), alpha = 0.5, color = "gray") +
  geom_point(size = 0.2, color = "black", alpha = 0.5,position = position_jitter(width = 0.05, seed = 1)) +
  facet_wrap(~ cell, scales = "free_y") +
  scale_fill_manual(values = c("PM" = "#F8B150FF", "Global" = "#BF9BDDFF")) +
  labs(
    x = NULL,
    y = "Tissue percentage"
  ) +
  theme_bw(base_size = 6) +
  theme(
    legend.position = "top",
    panel.grid.minor = element_blank()
  ) +
  stat_compare_means(
    aes(group = type),
    method = "wilcox.test",
    paired = TRUE,
    label = "p.format"
  )
print(fig3h)
ggsave(fig3h, file='./figure3/fig3h_pgmn_neighbour20_global_box_CPTAC.pdf', width = 8, height = 5, units = "cm")
