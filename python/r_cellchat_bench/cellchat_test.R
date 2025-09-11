library(dplyr)
library(Seurat)
library(ggplot2)
library(Matrix)
library(hdf5r)
library(dittoSeq)
library(SplineDV)
library(presto)
library(ggpubr)
library(tidyr)
library(stringr)
library(tibble)
library(cowplot)
library(openxlsx)
library(patchwork)
library(CellChat)

# --- SETUP AND DATA PROCESSING ---
base_dir <- "C:\\Users\\ssromerogon\\Documents\\vscode_working_dir\\QuantumXCT\\python\\r_benchmark"
# base_dir <- "/mnt/SCDC/Optimus/selim_working_dir/2023_nr4a1_colon/results"
setwd(base_dir)

# Read Seurat object
data <- readRDS(file.path(base_dir, "dataset_co_mo2.rds"))
table(data$CellType)

# Function to process Seurat object
process_rna <- function(data, assay_name = "RNA", num_hvg = 2000, dims_pca = 50, resolution = 1.0) {
  DefaultAssay(data) <- assay_name
  data <- FindVariableFeatures(data, selection.method = "vst", nfeatures = num_hvg)
  data <- ScaleData(data)
  data <- RunPCA(data)
  data <- RunUMAP(data, dims = 1:dims_pca, n.epochs = 500)
  data <- FindNeighbors(data, dims = 1:dims_pca)
  data <- FindClusters(data, resolution = resolution)
  return(data)
}

data <- process_rna(data)

# DimPlot visualization
plot <- DimPlot(object = data, reduction = "umap", group.by = c("CellType"),
                label = TRUE, repel = TRUE, label.size = 3, label.box = TRUE, alpha = 1, raster=FALSE) +
  NoLegend()
print(plot)

# Create 'Condition' metadata column
data$Condition <- as.character(data$BatchID)
data$Condition[grepl("\\(Co\\)", data$BatchID)] <- "Co"
data$Condition[grepl("\\(Mo\\)", data$BatchID)] <- "Mo"

# Subset data and create CellChat objects
data_co <- subset(data, Condition == 'Co')
data_co$CellType <- factor(data_co$CellType)
table(data_co$CellType)

data_mo <- subset(data, Condition == 'Mo')
data_mo$CellType <- factor(data_mo$CellType)
table(data_mo$CellType)

cellchat_Mo <- createCellChat(object = data_mo, meta = data_mo@meta.data, group.by = "CellType", assay = "RNA")
cellchat_Co <- createCellChat(object = data_co, meta = data_co@meta.data, group.by = "CellType", assay = "RNA")

rm(data_mo, data_co, data)

# Assign CellChat database
CellChatDB <- CellChatDB.human
cellchat_Mo@DB <- CellChatDB
cellchat_Co@DB <- CellChatDB

cellchat_Mo <- setIdent(cellchat_Mo, ident.use = "CellType")
cellchat_Co <- setIdent(cellchat_Co, ident.use = "CellType")


# --- CELLCHAT INFERENCE FOR EACH CONDITION ---
# Mo
cellchat_Mo <- subsetData(cellchat_Mo)
cellchat_Mo <- identifyOverExpressedGenes(cellchat_Mo)
cellchat_Mo <- identifyOverExpressedInteractions(cellchat_Mo)
cellchat_Mo <- computeCommunProb(cellchat_Mo)
cellchat_Mo <- filterCommunication(cellchat_Mo, min.cells = 50)
cellchat_Mo <- computeCommunProbPathway(cellchat_Mo)
cellchat_Mo <- aggregateNet(cellchat_Mo)

# Co
cellchat_Co <- subsetData(cellchat_Co)
cellchat_Co <- identifyOverExpressedGenes(cellchat_Co)
cellchat_Co <- identifyOverExpressedInteractions(cellchat_Co)
cellchat_Co <- computeCommunProb(cellchat_Co)
cellchat_Co <- filterCommunication(cellchat_Co, min.cells = 50)
cellchat_Co <- computeCommunProbPathway(cellchat_Co)
cellchat_Co <- aggregateNet(cellchat_Co)

# Save or load CellChat objects
# saveRDS(cellchat_Mo, file = "cellchat_Mo.rds")
# saveRDS(cellchat_Co, file = "cellchat_Co.rds")

# Assuming you've already run the above steps and saved the files
cellchat_Mo <- readRDS("cellchat_Mo.rds")
cellchat_Co <- readRDS("cellchat_Co.rds")

df <- as.data.frame(cellchat_Co@LR$LRsig)
df
cellchat_merged <- mergeCellChat(list(MO = cellchat_Mo, CO = cellchat_Co), add.names = c("MO", "CO"))

gg3 <- netVisual_heatmap(cellchat_merged, font.size = 14, font.size.title = 17) # Default "count"

# --- COMPARISON AND VISUALIZATION ---
# Merge objects for comparison
cellchat_merged <- mergeCellChat(list(Mo = cellchat_Mo, Co = cellchat_Co), add.names = c("Mo", "Co"))

# Heatmaps for interaction comparison
gg3 <- netVisual_heatmap(cellchat_merged, font.size = 14, font.size.title = 17) # Default "count"
gg4 <- netVisual_heatmap(cellchat_merged, measure = "weight", font.size = 15, font.size.title = 17) # measure "weight"
print(gg3)
print(gg4)

# --- DIFFERENTIAL GENE EXPRESSION AND MAPPING ---
# Define parameters for differential analysis
# 'Co' is your KO condition, so we set it as the positive dataset.
pos.dataset <- "Co"
features.name <- "differential_genes"

# Run differential gene expression analysis
cellchat_merged <- identifyOverExpressedGenes(
  cellchat_merged,
  group.dataset = "datasets",
  pos.dataset = pos.dataset,
  features.name = features.name,
  only.pos = FALSE,
  thresh.pc = 0.1,
  thresh.fc = 0.05,
  thresh.p = 0.05,
  group.DE.combined = FALSE
)

# Map the results to the communication networks
net <- netMappingDEG(cellchat_merged, features.name = features.name, variable.all = TRUE)
net.up <- subsetCommunication(cellchat_merged, net = net, datasets = "Co",ligand.logFC = 0.05, receptor.logFC = NULL)
write.csv(net, "net_up_Co_mapping_DEG_results.csv", row.names = FALSE)



# Your updated gene lists
gl_source <- c('STAT3','IL6RorST','TGFBR1or2','IL6ST', 'IL6R', 'TGFBR1', 'TGFBR2','PDGFB')
gl_recv <- c('PDGFRB', 'TGFB1', 'IL6')
merged_genes <- c(gl_source, gl_recv)

# Filter the 'net' data frame to keep only interactions where
# the ligand is in gl_source OR the receptor is in gl_recv.
filtered_net_by_genes <- net.up %>%
  filter(ligand %in% merged_genes | receptor %in% merged_genes)

# View the first few rows of the filtered results
head(filtered_net_by_genes)

# Save this filtered table to a new CSV file
# It will be saved in your current working directory (base_dir)
write.csv(filtered_net_by_genes, "net_up_filtered_by_specific_genes.csv", row.names = FALSE)

