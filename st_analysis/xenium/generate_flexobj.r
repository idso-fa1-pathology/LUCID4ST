suppressPackageStartupMessages({
	library(dplyr)
	library(Seurat)
	library(SingleCellExperiment)
	library(ProjecTILs)
	library(scCustomize)
})

xenium_path <- "./dat/Xenium_5k_humanLung_Cancer_FFPE/"

# ---- Build flex_data.obj from lung scRNAseq reference ----
# First, go download the reference data from the publication of Parzanowska & Lim Scientific Data, 2023: https://figshare.com/collections/An_integrated_single-cell_transcriptomic_dataset_for1_non-small_cell_lung_cancer/6222221/3
# Two files are needed: (1) RNA_rawcounts_matrix.rds and (2) metadata.csv. Place them in the directory "./dat/".
ref_raw <- readRDS("./dat/RNA_rawcounts_matrix.rds")
metadata <- read.csv("./dat/metadata.csv", row.names = 1)

sce <- SingleCellExperiment(
	assays = list(counts = ref_raw),
	colData = metadata
)

flex_data.obj <- as.Seurat(sce, counts = "counts", data = NULL)
flex_data.obj[["percent.mt"]] <- PercentageFeatureSet(flex_data.obj, pattern = "^MT-")

flex_data.obj <- subset(flex_data.obj, subset = Subtype == "adenocarcinoma")
flex_data.obj <- subset(
	flex_data.obj,
	subset = nCount_RNA > 200 & nCount_RNA < 10000 & percent.mt < 10
)

flex_data.obj$Visium_celltype_label <- flex_data.obj$Cell_Cluster_level2

# Convert cell type labels in the reference set to match the deconvoluted cell types in the visium data
map_to_malignant <- c(
	"CDKN2A Cancer", "SOX2 Cancer", "CXCL1 Cancer", "LAMC2 Cancer", "Proliferating Cancer"
)
map_to_cdc <- c("cDC2/moDCs")
map_to_macrophage <- c(
	"Monocytes", "Low quality Mac", "Lipid-associated Mac", "Alveolar Mac", "Proliferating Mac"
)
map_to_unidentifiable <- c("Pathological Alveolar", "Alveolar", "Ciliated")

flex_data.obj$Visium_celltype_label[
	flex_data.obj$Cell_Cluster_level2 %in% map_to_malignant
] <- "Malignant"
flex_data.obj$Visium_celltype_label[
	flex_data.obj$Cell_Cluster_level2 %in% map_to_cdc
] <- "cDC"
flex_data.obj$Visium_celltype_label[
	flex_data.obj$Cell_Cluster_level2 %in% map_to_macrophage
] <- "Macrophage"
flex_data.obj$Visium_celltype_label[
	flex_data.obj$Cell_Cluster_level2 %in% map_to_unidentifiable
] <- "Unidentifiable"

# ---- Infer T-cell states (ProjecTILs reference + transfer) ----
tcells.obj <- subset(
	flex_data.obj,
	subset = Visium_celltype_label %in% c("CD4+ Treg", "CD8+ Tem", "NK", "Naive T", "Proliferating T/NK")
)

t.ref <- get.reference.maps(reference = c("CD4", "CD8"))
t.int.ref <- merge(
	x = t.ref$human$CD4,
	y = t.ref$human$CD8,
	add.cell.ids = c("CD4", "CD8"),
	project = "Tcells_ref"
)

t.int.ref <- Convert_Assay(t.int.ref, assay = "RNA", "V5") # from package scCustomize

# ---- The following section is adapted from the ProjecTILs (https://github.com/carmonalab/ProjecTILs) main function, with some modifications to fix existing bugs ----
t.int.ref <- SCTransform(t.int.ref, assay = "RNA", verbose = FALSE)
t.int.ref <- FindVariableFeatures(t.int.ref, assay = "RNA")
t.int.ref <- ScaleData(t.int.ref, assay = "RNA")
t.int.ref <- RunPCA(t.int.ref, assay = "RNA")
t.int.ref <- FindNeighbors(t.int.ref, dims = 1:30, assay = "RNA")
t.int.ref <- FindClusters(t.int.ref, graph.name = "RNA_snn")
t.int.ref <- RunUMAP(t.int.ref, dims = 1:30)
tcells.obj <- SCTransform(tcells.obj, assay = "originalexp", verbose = FALSE)

tcells.anchors <- FindTransferAnchors(reference = t.int.ref, query = tcells.obj, dims = 1:30, reference.reduction = "pca")
predictions <- TransferData(anchorset = tcells.anchors, refdata = t.int.ref$functional.cluster, dims = 1:30)
tcells.obj <- AddMetaData(tcells.obj, metadata = predictions)

pred_max_cellstate <- apply(predictions[,2:dim(predictions)[2]], 1, function(x) names(x)[which.max(x)])

tcells.obj <- AddMetaData(tcells.obj, metadata = pred_max_cellstate, col.name = "t_cell_state")

# ---- Simplify T-cell states into CD4, CD8, and Others to match the Visium labels ----
tcells.obj@meta.data$t_state_simple <- "Other"
tcells.obj@meta.data$t_state_simple[grepl("CD4", tcells.obj@meta.data$t_cell_state)] <- "CD4_Tcell"
tcells.obj@meta.data$t_state_simple[grepl("CD8", tcells.obj@meta.data$t_cell_state)] <- "CD8_Tcell"
table(tcells.obj@meta.data$t_state_simple)

# Add to the original flex_data.obj
t_cell_states <- tcells.obj@meta.data$t_state_simple; names(t_cell_states) <- Cells(tcells.obj)
flex_data.obj <- AddMetaData(flex_data.obj, metadata = t_cell_states, col.name = "t_cell_state")

# Merge with Visium level
flex_data.obj@meta.data$Visium_celltype_label[!is.na(flex_data.obj@meta.data$t_cell_state)] <- flex_data.obj@meta.data$t_cell_state[!is.na(flex_data.obj@meta.data$t_cell_state)]

# ---- Output final flex_data.obj ----
saveRDS(flex_data.obj, file = "./dat/lung_xenium_flexdata_with_tcell_states.rds")
