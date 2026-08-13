

#####inhouse set####
setwd('/Volumes/yuan_lab/TIER2/anthracosis/0AI_PM_ST_code/pm_he_analysis')
in_house <- read.csv('../data/in_house_all.csv')
in_house <- in_house %>%
  mutate(sk_heavy = case_when(
    Pack.Year >= 40 ~ 'Heavy',
    Pack.Year  < 40 & Pack.Year > 0 ~ 'Moderate',
    Pack.Year == 0 ~ 'Never',
    TRUE ~ NA_character_
  ),
  sk_heavy = factor(sk_heavy, levels = c('Never', 'Moderate', 'Heavy')))


in_houseI <- in_house %>%
  filter(!is.na(sk_heavy)) %>%
  filter(sk_heavy %in% c('Never', 'Heavy'))

figS1e1 <- ggplot(in_houseI, aes(x=sk_heavy, y=pgmnper, fill=sk_heavy)) +
  geom_violin(alpha = 0.5, position = position_dodge(width = 0.8)) +
  geom_boxplot(width = 0.1,  outlier.shape = NA, position = position_dodge(width = 0.8)) +
  scale_y_sqrt()+
  scale_fill_manual(values = c("Never" = "#009E73",'Moderate'='cornflowerblue' ,"Heavy" = "#E69F00"))+  
  theme_classic() +
  theme(legend.position="none",plot.title = element_text(size=11)) +
  stat_compare_means()+
  xlab("")
print(figS1e1)
ggsave(figS1e1, file = "./figure1/figS1e_Heavy40_pgmnper_inhouse.pdf",width = 4, height = 5, units = "cm") 
in_houseI$sk_heavy <- factor(in_houseI$sk_heavy, levels = c("Never", "Heavy"))
wilcox.test(pgmnper ~ sk_heavy, data = in_houseI) #n=132, pgmnper_all, p=0.126; pgmnper:p-value = 0.3851


###MVA
in_house$type <- factor(in_house$type)
in_house$SEX <- factor(in_house$SEX)
in_house$stage <- factor(in_house$stage)
in_house$Race_ <- factor(in_house$Race_, levels = c("Caucasian", "African American", "Asian",  "Hispanic"))
in_house$EGFR <- factor(in_house$EGFR)
in_house$KRAS <- factor(in_house$KRAS)
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


model_inhouse <- lm(pgmnper ~ Age+ SEX + stage + Race_ + type+EGFR + KRAS, data = in_house)
summary(model_inhouse)
pdf("./figure2/figS2a_MVA_inhouse.pdf", width = 6, height = 5)
forest_model(model_inhouse, panels)
dev.off()





#######TCGA#######
tcga <- read.csv('../data/tcga_all.csv')
immune <- read.csv('../data/Immunity2018_LUAD.csv')
tcga <- tcga %>%
  left_join(immune, by='patient_id')

tcgare <- tcga %>%
  mutate(
    pgmnper.tile = factor(ntile(pgmnper, 2)),
    nssLPI = case_when(
      pgmnper.tile == "1" & type == 'never' ~ 'NS_L',
      pgmnper.tile == "2" & type == 'never' ~ 'NS_H',
      pgmnper.tile == "1" & type == 'smoker' ~ 'S_L',
      pgmnper.tile == "2" & type == 'smoker' ~ 'S_H',
      TRUE ~ NA_character_
    ),
    nssLPI = factor( nssLPI, level=c('NS_L', 'NS_H', 'S_L', 'S_H'))
  )%>%
  filter(!is.na(Plasma.Cells)) %>%
  filter(!is.na(Leukocyte.Fraction)) 
#n=333, smoker: 287, never-smoker:46
tcgare$Plasma.Cells_overall <- tcgare$Plasma.Cells * tcgare$Leukocyte.Fraction

figS3g1 <- ggplot(tcgare, aes(x=type, y=Plasma.Cells_overall, fill=type)) +
  geom_boxplot(width=0.6,  outliers=F) +
  geom_jitter(color="black", size=0.3, width=0.2) +
  #scale_y_sqrt()+
  scale_fill_manual(values = c("never" = "#009E73","smoker" = "#E69F00"))+  
  theme_classic() +
  theme(legend.position="none",plot.title = element_text(size=11)) +
  #ylim(0, 0.01)+
  stat_compare_means()+
  xlab("")
print(figS3g1)
###p-val between S and NS with regards to plasma%_overall is 0.028

my_comparison = list(c("NS_L", "NS_H"), c("NS_H", "S_L"), c("S_L", "S_H"))
figS3g<- ggplot(tcgare, aes(x=nssLPI, y=Plasma.Cells_overall, fill=nssLPI)) +
  geom_violin(alpha = 0.5, position = position_dodge(width = 0.8)) +
  geom_boxplot(width = 0.1,  outlier.shape = NA, position = position_dodge(width = 0.8)) +
  #geom_boxplot(width=0.6, outliers=F) +
  #geom_jitter(color="black", size=0.1, width=0.2) +
  #scale_y_sqrt()+
  scale_fill_manual(values = c("NS_L" = "#a6dba0","NS_H" = "#009E73", "S_L" = "#fdb863", "S_H" = "#b35806"))+  
  theme_classic() +
  theme(legend.position="none",plot.title = element_text(size=11)) +
  #ylim(0, 0.01)+
  stat_compare_means(comparisons = my_comparison, method='wilcox.test')+
  xlab("")
print(figS3g)
ggsave(figS3g, file = "./figure3/figS3g_pgmnper_tcga.pdf", width = 5, height = 4, units = "cm")

tcgare$nssLPI <- factor(tcgare$nssLPI, level=c('NS_L', 'NS_H'))
wilcox.test( Plasma.Cells_overall ~nssLPI, data = tcgare) #p-value = 0.03964


tcga <- tcga %>%
  mutate(sk_heavy = case_when(
    Pack.Year >= 40 ~ 'Heavy',
    Pack.Year  < 40 & Pack.Year > 0 ~ 'Moderate',
    Pack.Year == 0 ~ 'Never',
    TRUE ~ NA_character_
  ),
  sk_heavy = factor(sk_heavy, levels = c('Never', 'Moderate', 'Heavy')))

tcgaI <- tcga %>%
  filter(!is.na(sk_heavy)) %>%
  filter(sk_heavy %in% c('Never', 'Heavy'))

figS1e2 <- ggplot(tcgaI, aes(x=sk_heavy, y=pgmnper, fill=sk_heavy)) +
  geom_violin(alpha = 0.5, position = position_dodge(width = 0.8)) +
  geom_boxplot(width = 0.1,  outlier.shape = NA, position = position_dodge(width = 0.8)) +
  scale_y_sqrt()+
  scale_fill_manual(values = c("Never" = "#009E73" ,"Heavy" = "#E69F00"))+  
  theme_classic() +
  theme(legend.position="none",plot.title = element_text(size=11)) +
  stat_compare_means()+
  xlab("")
print(figS1e2)
ggsave(figS1e2, file = "./figure1/figS1e_Heavy40_pgmnper_tcga.pdf", width = 4, height = 5, units = "cm") 
tcgaI$sk_heavy <- factor(tcgaI$sk_heavy, levels = c("Never", "Heavy"))
wilcox.test(pgmnper ~ sk_heavy, data = tcgaI) #n=193, p=0.084


####confident S and NS as per JCO
tcgaI <- tcga %>%
  filter(Smoker.high.confidence == 'Yes' | Never.smoker.high.confidence == 'Yes') %>%
  mutate(confidence.smoker = case_when(
    Smoker.high.confidence == 'Yes' ~ 'Yes',
    Never.smoker.high.confidence == 'Yes' ~ 'No',
    TRUE ~ NA_character_
  ))

fig2f1 <- ggplot(tcgaI, aes(x=confidence.smoker, y=pgmnper, fill=confidence.smoker)) +
  geom_violin(alpha = 0.5, position = position_dodge(width = 0.8)) +
  geom_boxplot(width = 0.15,  outlier.shape = NA, position = position_dodge(width = 0.8)) +
  scale_y_sqrt()+
  scale_fill_brewer(palette='Set2')+
  theme_classic() +
  theme(legend.position="none",plot.title = element_text(size=11)) +
  stat_compare_means()+
  xlab("")
print(fig2f1)
ggsave(fig2f1, file = "./figure2/fig2f_pgmnper_JCOtcga.pdf", width = 4, height = 4, units = "cm") 
tcgaI$confidence.smoker <- factor(tcgaI$confidence.smoker, levels = c("No", "Yes"))
wilcox.test(pgmnper ~ confidence.smoker, data = tcgaI) #n=231, p=0.04755

tcga$type <- factor(tcga$type)
tcga$gender <- factor(tcga$gender)
tcga$stage <- factor(tcga$stage)
tcga$Race_ <- factor(tcga$Race_, levels = c("White", "African American", "Asian", "Other", "Unknown"))
tcga$EGFR <- factor(tcga$EGFR)
tcga$KRAS <- factor(tcga$KRAS)
tcga$TP53 <- factor(tcga$TP53, levels = c("TP53", "none"))
tcga$STK11 <- factor(tcga$STK11, levels = c("STK11", "none"))
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
  list(
    width   = 0.03,
    display = ~ ifelse(reference, "",
                       format.pval(p.value, digits = 2, eps = 1e-5)),
    hjust   = 1,
    heading = "p"
  ),
  list(width = 0.02)
)

model_tcga <- lm(pgmnper ~ Age +  gender + stage  + type+ Race_ + EGFR + KRAS, data = tcga)
summary(model_tcga)
pdf("./figure2/figS2a_MVA_tcga.pdf", width = 6, height = 5)
forest_model(model_tcga, panels)
dev.off()




#####CPTAC######
cptac <-read.csv('../data/cptac_all.csv')

cptac <- cptac %>%
  mutate(sk_heavy = case_when(
    Pack.Year >= 40 ~ 'Heavy',
    Pack.Year  < 40 & Pack.Year > 0 ~ 'Moderate',
    Pack.Year == 0 ~ 'Never',
    TRUE ~ NA_character_
  ),
  sk_heavy = factor(sk_heavy, levels = c('Never', 'Moderate', 'Heavy')))

cptacI <- cptac %>%
  filter(!is.na(sk_heavy)) %>%
  filter(sk_heavy %in% c('Never', 'Heavy'))

figS1e3 <- ggplot(cptacI, aes(x=sk_heavy, y=pgmnper, fill=sk_heavy)) +
  geom_violin(alpha = 0.5, position = position_dodge(width = 0.8)) +
  geom_boxplot(width = 0.1,  outlier.shape = NA, position = position_dodge(width = 0.8)) +
  scale_y_sqrt()+
  scale_fill_manual(values = c("Never" = "#009E73", "Heavy" = "#E69F00"))+  
  theme_classic() +
  theme(legend.position="none",plot.title = element_text(size=11)) +
  #ylim(0, 0.01)+
  stat_compare_means()+
  xlab("")
print(figS1e3)
ggsave(figS1e3, file = "./figure2/figS1e_Heavy40_pgmnper_cptac.pdf",width = 4, height = 5, units = "cm") 
cptacI$sk_heavy <- factor(cptacI$sk_heavy, levels = c("Never", "Heavy"))
wilcox.test(pgmnper ~ sk_heavy, data = cptacI) #n=95, pgmnper_all:p=0.006667, pgmnper: p=0.015

####confident S and NS as per JCO
cptacI <- cptac %>%
  filter(Smoker.high.confidence == 'Yes' | Never.smoker.high.confidence == 'Yes') %>%
  mutate(confidence.smoker = case_when(
    Smoker.high.confidence == 'Yes' ~ 'Yes',
    Never.smoker.high.confidence == 'Yes' ~ 'No',
    TRUE ~ NA_character_
  ))

fig2f2 <- ggplot(cptacI, aes(x=confidence.smoker, y=pgmnper, fill=confidence.smoker)) +
  geom_violin(alpha = 0.5, position = position_dodge(width = 0.8)) +
  geom_boxplot(width = 0.15,  outlier.shape = NA, position = position_dodge(width = 0.8)) +
  scale_y_sqrt()+
  scale_fill_brewer(palette='Set2')+
  theme_classic() +
  theme(legend.position="none",plot.title = element_text(size=11)) +
  stat_compare_means()+
  xlab("")
print(fig2f2)
ggsave(fig2f2, file = "./figure2/fig2f_pgmnper_JCOcptac.pdf", width = 4, height = 4, units = "cm") 
cptacI$confidence.smoker <- factor(cptacI$confidence.smoker, levels = c("No", "Yes"))
wilcox.test(pgmnper ~ confidence.smoker, data = cptacI) #n=67, pgmnper_all:p=0.6736; pgmnper: p-value = 0.3689


cptac$type <- factor(cptac$type)
cptac$gender <- factor(cptac$gender)
cptac$stage <- factor(cptac$stage)
cptac$Race_ <- factor(cptac$Race_, levels = c("White", "African American", "Asian", "Other",  "Unknown"))
cptac$region <- factor(cptac$region, levels = c("america", "asia", "europe",  "unknown"))
cptac$EGFR <- factor(cptac$EGFR)
cptac$KRAS <- factor(cptac$KRAS)
cptac$TP53 <- factor(cptac$TP53, level=c('TP53', 'none'))
cptac$STK11 <- factor(cptac$STK11, level=c('STK11', 'none'))

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

  list(
    width   = 0.03,
    display = ~ ifelse(reference, "",
                       format.pval(p.value, digits = 3, eps = 1e-5)),
    hjust   = 1,
    heading = "p"
  ),
  list(width = 0.02)
)


model_cptac <- lm(pgmnper ~  Age +  gender + stage + type+region+EGFR + KRAS , data = cptac)
summary(model_cptac)
#car::Anova(model_cptac, type='III', test.statistic = 'Wald')
pdf("./figure3/fig3a_MVA_cptac_region.pdf", width = 6, height = 5) # 6, 5
forest_model(model_cptac, panels)
dev.off()

model_cptac <- lm(pgmnper ~  Age +  gender + stage + type+Race_+EGFR + KRAS , data = cptac)
summary(model_cptac)
#car::Anova(model_cptac, type='III', test.statistic = 'Wald')
pdf("./figure3/fig3a_MVA_cptac_race.pdf", width = 6, height = 5) # 6, 5
forest_model(model_cptac, panels)
dev.off()





######Modern Pathology######
load('../data/MP_precancer_Aug23.RData')
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
car::Anova(model, type='III', test.statistic = 'Wald')
pdf("./figure3/fig3a_MVA_precancer_region.pdf", width = 6, height = 5)
forest_model(model, panels)
dev.off()
