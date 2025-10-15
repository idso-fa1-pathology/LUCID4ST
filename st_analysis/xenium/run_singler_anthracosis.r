lapply(c("dplyr", "Seurat", "ggplot2", "SingleR", "SingleCellExperiment", "zellkonverter", "scuttle", "scrapper", "crunch"), library, character.only = T)
options(repr.plot.width=15, repr.plot.height=15)



flex_data.obj <- readRDS(file = "./dat/lung_xenium_flexdata_with_tcell_states.rds")


xenium_path <- "./dat/Xenium_5k_humanLung_Cancer_FFPE/"
xenium.obj <- LoadXenium(xenium_path, fov = "fov", molecule.coordinates = FALSE)
DefaultAssay(xenium.obj) <- "Xenium"

# Add log1p_nCount_RNA, log1p_nFeatures
xenium.obj@meta.data$nCount_Xenium_log <- log1p(xenium.obj@meta.data$nCount_Xenium)
xenium.obj@meta.data$nFeature_Xenium_log <- log1p(xenium.obj@meta.data$nFeature_Xenium)

# Remove any empty cells
xenium.obj <- subset(xenium.obj, subset = nCount_Xenium > 40 & nFeature_Xenium > 15)


DefaultAssay(flex_data.obj) <- "originalexp"
flex_data.obj <- NormalizeData(flex_data.obj) %>%
                 FindVariableFeatures() %>%
                 ScaleData()

DefaultAssay(xenium.obj) <- "Xenium"
xenium.obj <- NormalizeData(xenium.obj) %>%
                ScaleData()

sceX <- as.SingleCellExperiment(xenium.obj)
sceX <- logNormCounts(sceX)

sceS <- as.SingleCellExperiment(flex_data.obj)
sceS <- logNormCounts(sceS)

# add cell type labels to the SingleCellExperiment objects
celltype <- flex_data.obj$Visium_celltype_label
names(celltype) <- Cells(flex_data.obj)
sceS$Visium_celltype_label <- celltype

pred <- SingleR(test=sceX, ref=sceS, labels=sceS$Visium_celltype_label, de.method="wilcox")

# save pred as rds
saveRDS(pred, "./out/lung_xenium_singler_visium_labels_tcell_states.rds")

write.csv.gz(pred, "./out/lung_xenium_singler_visium_labels_tcell_states.csv.gz", row.names = TRUE)

pred_labels <- as.data.frame(pred$labels)
rownames(pred_labels) <- rownames(pred)
head(pred_labels)

colnames(pred) <- "group"
# save to csv
write.csv.gz(pred_labels, "./out/lung_xenium_singler_pred_label_only_visium_labels_tcell_states.csv.gz", row.names = TRUE)