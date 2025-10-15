library(Seurat)
library(ggplot2)
library(SpaCET)
library(Matrix)

# Done "TMA5-034", "TMA5-073", "TMA5-084", "TMA5-103", "TMA5-104", "TMA5-142", "TMA5-157", "TMA5-161"
sample_list <- c("TMA5-023")
spotclean_output <- "./dat/spatial_spot_cleaned"
st_base_dir <- "./dat/spatial-transcriptome"
spacet_output <- "./dat/withQCumi350_MT10_diffusion/spacet_rds"

markers <- read.csv("./dat/spacet_deconv/marker_list/marker_gene_score_2025May.csv")

for (stsample in sample_list) {

    SpaCET_obj <- create.SpaCET.object.10X(visiumPath = file.path("./dat/spatial_spot_cleaned", stsample))

    # Perform quality control on SpaCET object
    SpaCET_obj <- SpaCET.quality.control(SpaCET_obj, min.gene = 10)

    # Extract count matrix
    counts <- SpaCET_obj@input$counts 
    # Step 1: Compute QC metrics
    umi_counts <- Matrix::colSums(counts)          
    gene_counts <- Matrix::colSums(counts > 0)            

    # Step 2: Apply spot-level filters
    spot_filter <- umi_counts > 350 & gene_counts > 10
    counts_filtered <- counts[, spot_filter]

    # Step 3: Apply gene-level filter (e.g., gene detected in >= 3 spots)
    min.cells <- 5
    gene_filter <- Matrix::rowSums(counts_filtered > 0) >= min.cells
    counts_filtered <- counts_filtered[gene_filter, ]


    # Step 4: Update SpaCET object
    SpaCET_obj@input$counts <- counts_filtered

    # Update spot coordinates
    SpaCET_obj@input$spotCoordinates <- SpaCET_obj@input$spotCoordinates[
    rownames(SpaCET_obj@input$spotCoordinates) %in% colnames(counts_filtered),
    ]

    # (Optional) Add computed QC metrics into the object
    SpaCET_obj@input$metaData <- data.frame(
    barcode = colnames(counts_filtered),
    nCount_Spatial = umi_counts[colnames(counts_filtered)],
    nFeature_Spatial = gene_counts[colnames(counts_filtered)],
    row.names = colnames(counts_filtered)
    )

    SpaCET_obj <- SpaCET.deconvolution(SpaCET_obj, cancerType="LUAD", coreNo=15)

    saveRDS(SpaCET_obj, file = file.path(spacet_output, paste0(stsample, "_spacet.rds")))

    # save spacet result object
    row_col_2_spot_coordintes <- SpaCET_obj@input$spotCoordinates
    row_col_2_spot_coordintes$coor <- rownames(row_col_2_spot_coordintes)
    row_col_2_spot_coordintes <- row_col_2_spot_coordintes[row_col_2_spot_coordintes$coor %in% colnames(counts_filtered), ]

    cell_prop_mat <- SpaCET_obj@results$deconvolution$propMat
    cell_prop_mat <- as.data.frame(cell_prop_mat)
    # match colnames of cell_prop_mat to row_col_2_spot_coordintes
    colnames(cell_prop_mat) <- row_col_2_spot_coordintes$barcode[match(row_col_2_spot_coordintes$coor, colnames(cell_prop_mat))]

    print(cell_prop_mat[1:15,1:6])
    write.csv(cell_prop_mat, file = file.path("./out/spacet_deconv/withQCumi350_MT10_diffusion", paste0(stsample, "_spacet.csv")), row.names = TRUE, quote = FALSE)

    SpaCET_obj <- SpaCET.GeneSetScore(SpaCET_obj, GeneSets=markers)
    gene_set_mat <- SpaCET_obj@results$GeneSetScore
    gene_set_mat <- as.data.frame(gene_set_mat)
    colnames(gene_set_mat) <- row_col_2_spot_coordintes$barcode[match(row_col_2_spot_coordintes$coor, colnames(gene_set_mat))]

    print(gene_set_mat[1:15,1:6])
    write.csv(gene_set_mat, file = file.path("./out/spacet_deconv/withQCumi350_MT10_diffusion", paste0(stsample, "_spacet_markers.csv")), row.names = TRUE, quote = FALSE)
}