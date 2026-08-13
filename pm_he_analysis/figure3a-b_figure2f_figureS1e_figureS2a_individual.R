


######Modern Pathology######
##step1 - to compile tme and pgmn data
tme <- read_excel('/Volumes/yuan_lab/TIER2/anthracosis/LungAdenocarcinomaEvolutionHE_PingMP/USA/tme_pix.xlsx')
pgmn <- read_excel('/Volumes/yuan_lab/TIER2/anthracosis/LungAdenocarcinomaEvolutionHE_PingMP/USA/rawRes_pigment.xlsx')

pixel_cols <- setdiff(names(tme), "ID")

tme <- tme %>%
  rowwise() %>%
  mutate(
    tme_pix = sum(c_across(all_of(pixel_cols))),
    across(all_of(pixel_cols),
           ~ .x / tme_pix,
           .names = "{.col}_pct")
  ) %>%
  ungroup()

tme[c(13:22)] <- NULL
tme_pgmn <- tme %>%
  left_join(pgmn, by='ID')
#china_tme_pgmn$pgmnper <- china_tme_pgmn$pigment8/ china_tme_pgmn$tme_pix
#china_tme_pgmn[c('pigment8')] <- NULL
write.csv(tme_pgmn, '/Users/xiaoxipan/Library/CloudStorage/OneDrive-Insidein_housenderson/yuanlab/Projects/Anthracosis/datain_house/modernP/pgmnUSA.csv', row.names = FALSE)


##step2 - add clinic and computational data
comp <- read.csv('/Users/xiaoxipan/Library/CloudStorage/OneDrive-Insidein_housenderson/yuanlab/Projects/Anthracosis/datain_house/modernP/CombineROIFeatures.csv')
pgmnChina <- read.csv('/Users/xiaoxipan/Library/CloudStorage/OneDrive-Insidein_housenderson/yuanlab/Projects/Anthracosis/datain_house/modernP/pgmnChina.csv')
pgmnJapan <- read.csv('/Users/xiaoxipan/Library/CloudStorage/OneDrive-Insidein_housenderson/yuanlab/Projects/Anthracosis/datain_house/modernP/pgmnJapan.csv')
pgmnUSA <- read.csv('/Users/xiaoxipan/Library/CloudStorage/OneDrive-Insidein_housenderson/yuanlab/Projects/Anthracosis/datain_house/modernP/pgmnUSA.csv')
pgmn <- rbind(pgmnUSA, pgmnJapan, pgmnChina)
pgmn[c(2:11)] <- NULL

colnames(comp)[colnames(comp) == 'Lesions'] <- 'ID'
pgmn$pgmnperROI <- pgmn$pigment8/ pgmn$tme_pix
comp <- comp %>%
  left_join(pgmn, by = 'ID')

comp <- comp %>%
  mutate(
    PatientID = case_when(
      str_starts(ID, "H") ~ str_extract(ID, "^[^-]+-[^-]+"),   # before 2nd "-"
      TRUE                ~ sub("-.*", "", ID)                 # before 1st "-"
    )
  )
length(unique(comp$PatientID))

###step3- forest plot
clinicChina <- read_excel('/Users/xiaoxipan/Library/CloudStorage/OneDrive-Insidein_housenderson/yuanlab/Projects/Anthracosis/datain_house/modernP/clinicChina.xlsx')
clinicJapan <- read_excel('/Users/xiaoxipan/Library/CloudStorage/OneDrive-Insidein_housenderson/yuanlab/Projects/Anthracosis/datain_house/modernP/clinicJapan.xlsx')
clinicUSA <- read_excel('/Users/xiaoxipan/Library/CloudStorage/OneDrive-Insidein_housenderson/yuanlab/Projects/Anthracosis/datain_house/modernP/clinicUSA.xlsx')
clinicChina$dataset <- 'China'
clinicChina$region <- 'Asia'
clinicChina$Race <- 'Asian'
clinicJapan$dataset <- 'Japan'
clinicJapan$region <- 'Asia'
clinicJapan$Race <- 'Asian'
clinicUSA$dataset <- 'USA'
clinicUSA$region <- 'America'
clinicUSA$Race <- 'White'
clinic <- rbind(clinicUSA, clinicJapan, clinicChina)

rm(clinicUSA, clinicJapan, clinicChina, pgmnUSA, pgmnJapan, pgmnChina, pgmn, compT)
save.image("~/Library/CloudStorage/OneDrive-Insidein_housenderson/yuanlab/Projects/Anthracosis/datain_house/modernP/pgmn_clinic_Aug23.RData")

load('~/Library/CloudStorage/OneDrive-Insidein_housenderson/yuanlab/Projects/Anthracosis/datain_house/modernP/pgmn_clinic_Aug23.RData')
compAAH <- comp %>%
  filter(Stages == 'AAH')
compAIS <- comp %>%
  filter(Stages == 'AIS')
compMIA <- comp %>%
  filter(Stages == 'MIA')

comp %>%
  group_by(PatientID) %>% 
  summarise(pgmn_all = sum(pigment8), tme_all = sum(tme_pix)) ->compT

compT$pgmnper <- compT$pgmn_all / compT$tme_all

compT <- compT %>%
  right_join(clinic, by='PatientID')
compT$type <- compT$SmokerType
compT$type[compT$type == 'Current/Former'] <- 'Smoker'

priority <- c("Normal","AAH","AIS","MIA","ADC")  # lowest → highest priority

clinicS <- comp %>%
  mutate(Stages = factor(Stages, levels = priority, ordered = TRUE)) %>%
  group_by(PatientID) %>%
  summarise(stage = as.character(max(Stages, na.rm = TRUE)),
            .groups = "drop")

compT <- compT %>%
  right_join(clinicS, by='PatientID')

setwd('/Users/xiaoxipan/Library/CloudStorage/OneDrive-Insidein_housenderson/yuanlab/Manuscripts/LPI/fig/v12/fig2')
compT$type <- factor(compT$type)
compT$Gender <- factor(compT$Gender)
compT$stage <- factor(compT$stage, levels = c("ADC", "MIA", "AIS","AAH", "Normal"))
compT$Race <- factor(compT$Race)
compT$dataset <- factor(compT$dataset, levels = c('USA', 'Japan', 'China'))
panels <- list(
  list(width = 0.01),
  # Variable names
  list(width = 0.1, display = ~variable, fontface = "bold", heading = "Variable"),
  list(width = 0.1, display = ~level),
  list(width = 0.05, display = ~n, hjust = 1, heading = "N"),
  list(width = 0.03, item = "vline", hjust = 0.5),
  list(width = 0.3, item = "forest", hjust = 0.5, linetype = "dashed",
       line_x = 0),   # <– dots & CIs go here
  list(width = 0.03, item = "vline", hjust = 0.5),
  
  # Estimate (95 % CI)
  #list(
  #  width   = 0.2,
  #  display = ~ ifelse(reference,"Reference", sprintf("%0.5f (%0.5f, %0.5f)", trans(estimate),trans(conf.low),trans(conf.high))),
  #  heading = "Estimate (95% CI)"),
  
  #list(width = 0.01, item = "vline", hjust = 0.5),
  # p-value
  list(
    width   = 0.03,
    display = ~ ifelse(reference, "",
                       format.pval(p.value, digits = 2, eps = 1e-5)),
    hjust   = 1,
    heading = "p"
  ),
  list(width = 0.02)
)

compTre <- compT %>%
  filter(stage != 'Normal')
model <- lm(pgmnper ~ Age+ Gender +  stage+ type + dataset, data = compT)
summary(model)
Anova(model, type='III', test.statistic = 'Wald')
pdf("fig2gMVA_MP_pgmnper_stage.pdf", width = 6, height = 5)
forest_model(model, panels)
dev.off()
