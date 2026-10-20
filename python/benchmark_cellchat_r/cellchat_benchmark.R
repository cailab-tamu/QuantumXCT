# cellchat_benchmark.R  — CellChat DIFFERENTIAL (Co vs Mo), following the canonical
# CellChat comparison workflow. Refactored so a single reusable function builds each
# isolated per-condition CellChat object; the rest (merge -> differential DEG ->
# netMappingDEG -> net.up / net.dn) runs once on the merged object.
# Writes into results/ccc_comparison/:
#   cellchat_{Mo,Co}_all.csv        per-condition significant LR interactions
#   cellchat_net_mapping_DEG.csv    full DEG-mapped network (both conditions)
#   cellchat_co_vs_mo_up.csv        interactions UP in Co  (gained co-culture)
#   cellchat_co_vs_mo_down.csv      interactions DOWN in Co (lost co-culture)
# ENV: R (>=4.2) with Seurat + CellChat (see CCC_SETUP_README.md).
#
# Input: dataset_co_mo2.rds (Seurat) with meta columns `CellType` and `BatchID`
#        (BatchID like "Cancer Cells (Co)"). If you only have the .h5ad, run
#        convert_h5ad_to_seurat.R first.

suppressPackageStartupMessages({
  library(dplyr); library(Seurat); library(CellChat); library(stringr)
})

this_dir     <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile)), error = function(e) getwd())
PYTHON_ROOT  <- normalizePath(file.path(this_dir, ".."), mustWork = FALSE)
OUT          <- file.path(PYTHON_ROOT, "results", "ccc_comparison")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
rds_path     <- file.path(PYTHON_ROOT, "dataset_co_mo2.rds")
MIN_CELLS    <- 50    # matches cellchat_test.R; lower to ~10 for small subsets
LOGFC        <- 0.05  # ligand logFC threshold for up/down calls

stopifnot(file.exists(rds_path))
data <- readRDS(rds_path)

data$Condition <- as.character(data$BatchID)
data$Condition[grepl("\\(Co\\)", data$BatchID)] <- "Co"
data$Condition[grepl("\\(Mo\\)", data$BatchID)] <- "Mo"

# --- REUSABLE: build one isolated CellChat object for a single condition ------
compute_cellchat <- function(obj, min_cells = MIN_CELLS) {
  obj$CellType <- factor(obj$CellType)
  cc <- createCellChat(object = obj, meta = obj@meta.data, group.by = "CellType", assay = "RNA")
  cc@DB <- CellChatDB.human
  cc <- setIdent(cc, ident.use = "CellType")
  cc <- subsetData(cc)
  cc <- identifyOverExpressedGenes(cc)
  cc <- identifyOverExpressedInteractions(cc)
  cc <- computeCommunProb(cc)
  cc <- filterCommunication(cc, min.cells = min_cells)
  cc <- computeCommunProbPathway(cc)
  cc <- aggregateNet(cc)
  cc
}

cellchat_Mo <- compute_cellchat(subset(data, Condition == "Mo"))
cellchat_Co <- compute_cellchat(subset(data, Condition == "Co"))

write.csv(subsetCommunication(cellchat_Mo), file.path(OUT, "cellchat_Mo_all.csv"), row.names = FALSE)
write.csv(subsetCommunication(cellchat_Co), file.path(OUT, "cellchat_Co_all.csv"), row.names = FALSE)

# --- DIFFERENTIAL (identical to the proven workflow) -------------------------
merged <- mergeCellChat(list(Mo = cellchat_Mo, Co = cellchat_Co), add.names = c("Mo", "Co"))

features.name <- "differential_genes"
merged <- identifyOverExpressedGenes(
  merged,
  group.dataset = "datasets",
  pos.dataset   = "Co",            # Co = the perturbed/interacting condition
  features.name = features.name,
  only.pos      = FALSE,
  thresh.pc     = 0.1,
  thresh.fc     = 0.05,
  thresh.p      = 0.05,
  group.DE.combined = FALSE
)

net <- netMappingDEG(merged, features.name = features.name, variable.all = TRUE)
write.csv(net, file.path(OUT, "cellchat_net_mapping_DEG.csv"), row.names = FALSE)

# UP in Co: positive ligand logFC in the Co dataset
net.up <- subsetCommunication(merged, net = net, datasets = "Co",
                              ligand.logFC =  LOGFC, receptor.logFC = NULL)
# DOWN in Co: negative ligand logFC (read from the Mo dataset side)
net.dn <- subsetCommunication(merged, net = net, datasets = "Mo",
                              ligand.logFC = -LOGFC, receptor.logFC = NULL)

write.csv(net.up, file.path(OUT, "cellchat_co_vs_mo_up.csv"),   row.names = FALSE)
write.csv(net.dn, file.path(OUT, "cellchat_co_vs_mo_down.csv"), row.names = FALSE)
cat(sprintf("CellChat differential: %d up-in-Co, %d down-in-Co interactions saved.\n",
            nrow(net.up), nrow(net.dn)))
